"""E018 M1: ルート農場 + 全品目リアクティブ市場.

ルート (720ステップの farmer/hands 行動) をそのまま再生し、市場オーダーだけを
全品目対応のリアクティブロジックに置き換える (E016b の品目ミスマッチ修正)。

リアクティブ市場の設計:
  - 売却: MELON/WHEAT/STRAWBERRY/CARROT/MILK/WOOL/EGG/FERTILIZER 全部 (売却最優先)
  - 種: ルートの将来の植え付け (lookahead ウィンドウ内の PLANT 数) を満たすよう全作物購入
  - 動物: ルートの累積購入ペースを目標に資金ゲート付きで購入 (配置 PICKUP 前に届ける)
  - 土地: ルートの購入回数を目標に同じタイミングで購入 (遅延は農場スケジュール崩壊)
  - 雇用: ルートの1日あたり雇用数を上限に hour1-2 で資金ゲート付きで雇用
  - フィード: shed 在庫ターゲット管理 (ルートが一日中 PICKUP するため常時補充)
  - 10件制限: 売却 > 雇用 > 動物/土地 > 種/製品 の優先順でトリム
  - 売却順序: 価格影響の大きい売りを先に (Kaito 方式)
"""
import math
CROPS4 = ["WHEAT", "CARROT", "STRAWBERRY", "MELON"]
SEED_COST = {"WHEAT": 10, "CARROT": 20, "STRAWBERRY": 100, "MELON": 80}
ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
SELL_ORDER = ["MELON", "WHEAT", "STRAWBERRY", "CARROT", "MILK", "WOOL", "EGG", "FERTILIZER"]
SELL_CHUNK = {"MELON": 15, "WHEAT": 20, "STRAWBERRY": 10, "CARROT": 10,
              "MILK": 10, "WOOL": 10, "EGG": 10, "FERTILIZER": 10}
FEED_RESERVE_PER_ANIMAL = 2
FEED_RESERVE_MIN = 6
WHEAT_STOCK_TARGET = 8
LOOKAHEAD = 96
SEED_BUFFER = 1
ANIMAL_BUFFER = 0
LAND_BUFFER = 100
FERT_BUFFER = 250
SEED_MAX_BUY = 8
RANK_SELL_ORDERS = False
# M5 資金フロー修正: 種の大量前買い (96ステップ先読み) が d0-9 の資金を枯渇させ、
# ルートの重要購入 (d4 牛・d6 土地) を遅らせて動物滞留・土地遅延を連鎖させていた
# (LB 実戦リプレイ分析: d6 にイチゴ種23個=$2.3k を前買い vs 元プレイヤーは4個)。
M5_JIT_SEEDS = True
SEED_LOOKAHEAD_JIT = 12
SEED_MAX_BUY_JIT = 16
# フィード小麦の補充は資金に余裕があるときだけ (元プレイヤーは d4-6 に小麦購入0)。
M5_FEED_FLOOR = True
FEED_CASH_FLOOR = 250
# M5b ステップ同期修正: ルートの steps[k].action は obs[k-1] に応答した行動
# (リプレイは事後状態を記録)。ローカルでは obs[k] に route[k+1] を提出しないと
# 農場が1ステップ遅れ、日境界直後の PICKUP (d7h1 COW3 など) が shed 非隣接で失敗し
# 動物が滞留する (位置比較で 0/240 一致することを実証済み)。
M5_OFFSET = True
# M5b HIRE 同期: ルートの HIRE オーダーを orig と同時刻に再現する (ハンドの
# スポーン位置を orig と一致させ、ハンド依存の農場作業のずれを防ぐ)。
# ギャップフィラーはルート雇用が完了した後 (hour > その日の最終 HIRE hour) のみ。
M5_ROUTE_HIRES = True
M5_GAP_HIRE_LATE = True
# M3: ギャップフィラーハンド。ルートの雇用数に加えて最大 GAP_HIRE_MAX 人を
# 追加雇用し、水やり・雑草除去をリアクティブに実行させる (ルートのハンド位置
# ドリフトで水やりが漏れた作物の枯死を防ぐ)
GAP_HIRE_MAX = 2

# 価格影響スコア用の市場パラメータ (kaggriculture.py の MARKET_PARAMS と同一)
_PRICE_PARAMS = {
    "WHEAT":      {"base": 25, "I0": 10000, "T": 400, "below_func": "sqrt", "below_target": 0.80, "above_func": "log", "above_target": 0.20},
    "CARROT":     {"base": 35, "I0": 10000, "T": 450, "below_func": "hinge", "below_target": 1.00, "above_func": "sqrt", "above_target": 0.70},
    "STRAWBERRY": {"base": 120, "I0": 10000, "T": 100, "below_func": "sqrt", "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":      {"base": 250, "I0": 10000, "T": 300, "below_func": "log", "below_target": 0.20, "above_func": "sq", "above_target": 3.60},
    "EGG":        {"base": 50, "I0": 10000, "T": 332, "below_func": "hinge", "below_target": 0.40, "above_func": "log", "above_target": 0.20},
    "MILK":       {"base": 160, "I0": 10000, "T": 122, "below_func": "sqrt", "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":       {"base": 200, "I0": 10000, "T": 105, "below_func": "log", "below_target": 0.20, "above_func": "sq", "above_target": 3.20},
    "FERTILIZER": {"base": 100, "I0": 10000, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}


def _price_shape(func, x, T=None):
    x = max(0.0, x)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return x ** 0.5
    if func == "log":
        return math.log(1.0 + x)
    if func == "hinge":
        if not T or T <= 0:
            return x
        u = x / T
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    return x


def _market_price(item, inventory):
    p = _PRICE_PARAMS[item]
    base, I0, T = p["base"], p["I0"], p["T"]
    if inventory < I0:
        amp = p["below_target"] * base / _price_shape(p["below_func"], T)
        price = base + amp * _price_shape(p["below_func"], I0 - inventory)
    else:
        amp = p["above_target"] * base / _price_shape(p["above_func"], T)
        price = base - amp * _price_shape(p["above_func"], inventory - I0)
    return max(1, int(round(price)))


def _sell_impact_score(obs, order):
    """売却による価格下落 × 数量 (Kaito の _impact_score 移植)。"""
    if not (isinstance(order, (list, tuple)) and len(order) >= 3 and order[0] == "SELL"):
        return float("-inf")
    item = order[1]
    if item not in _PRICE_PARAMS:
        return float("-inf")
    try:
        qty = max(0, int(order[2]))
    except (TypeError, ValueError):
        return 0.0
    if qty <= 0:
        return 0.0
    market = obs.get("market", {}) or {}
    inv = market.get("inventory", {}) or {}
    prices = market.get("prices", {}) or {}
    current = int(inv.get(item, 10000) or 10000)
    quote = prices.get(item, _market_price(item, current))
    later = _market_price(item, current + qty)
    return float(qty) * max(0.0, quote - later)


def _rank_sell_orders(obs, market):
    """売却オーダーを価格影響の大きい順に並べ替える (同じスロット内で高額売却を先行)。"""
    rows = [
        (_sell_impact_score(obs, o), -idx, list(o))
        for idx, o in enumerate(market)
        if o and o[0] == "SELL"
    ]
    if len(rows) < 2:
        return market
    rows.sort(key=lambda r: (r[0], r[1]), reverse=True)
    ranked = iter(r[2] for r in rows)
    return [next(ranked) if o and o[0] == "SELL" else o for o in market]


def _fib(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _count_plant(entry, crop):
    n = 0
    acts = [entry.get("farmer") or ["PASS"]]
    acts += [h or ["PASS"] for h in entry.get("hands", [])]
    for a in acts:
        if a and len(a) > 1 and a[0] == "PLANT" and a[1] == crop:
            n += 1
    return n


def _plants_window(route, crop, W):
    """各 step から W ステップ以内の PLANT crop 回数 (ルート計画の種需要)。"""
    n = len(route)
    counts = [0] * n
    winsum = 0
    for i in range(min(W, n)):
        winsum += _count_plant(route[i], crop)
    counts[0] = winsum
    for s in range(1, n):
        winsum -= _count_plant(route[s - 1], crop)
        add = s + W - 1
        if add < n:
            winsum += _count_plant(route[add], crop)
        counts[s] = winsum
    return counts


def _next_plant_step(route, crop):
    """各 step 以降で最初に PLANT crop が現れる step (無ければ n)。"""
    n = len(route)
    out = [n] * n
    nxt = n
    for s in range(n - 1, -1, -1):
        if _count_plant(route[s], crop) > 0:
            nxt = s
        out[s] = nxt
    return out


def _animal_demand(route):
    """ルート農場の PICKUP <animal> から必要動物を導出 (品目ミスマッチ防止)。

    戻り値: {animal: {"target": 合計, "first_step": 最初のPICKUP step,
                       "last_step": 最後のPICKUP step}}
    """
    demand = {}
    for s, r in enumerate(route):
        acts = [r.get("farmer") or ["PASS"]]
        acts += [h or ["PASS"] for h in r.get("hands", [])]
        for act in acts:
            if act and len(act) >= 3 and act[0] == "PICKUP" and act[1] in ANIMAL_COST:
                d = demand.setdefault(act[1], {"target": 0, "first_step": s, "last_step": s})
                d["target"] += int(act[2])
                d["last_step"] = s
    return demand


_MOVE = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
_SHED_TILES = [(4, 4), (4, 5), (5, 4), (5, 5)]


def _sim_positions(route, hire_target):
    """ルートの農場アクションの実行位置をシミュレーションする (M3c)。

    ファーマーの位置は完全に決定論的。ハンドのスポーンは占有状況依存のため
    近似 (hire_target 人雇う前提)。戻り値:
      - build_pos: step -> その後の BUILD_PASTURE 位置の集合
      - plant_pos: step -> その後の PLANT 位置の集合
      - build_events: [(step, position)] のリスト
    """
    n = len(route)
    build_events = []
    plant_events = []
    farmer = [4, 4]
    hands = []
    for s, r in enumerate(route):
        hour = s % 24
        if hour == 0:
            farmer = [4, 4]
            hands = []
        n_hands = hire_target[s // 24]
        acts = [r.get("farmer") or ["PASS"]]
        acts += [h or ["PASS"] for h in r.get("hands", [])]

        def move(p, op):
            dx, dy = _MOVE[op]
            return [p[0] + dx, p[1] + dy]

        fa = acts[0]
        if fa[0] in _MOVE:
            farmer = move(farmer, fa[0])
        elif fa[0] == "BUILD_PASTURE":
            build_events.append((s, tuple(farmer)))
        elif fa[0] == "PLANT":
            plant_events.append((s, tuple(farmer)))
        for i in range(min(len(acts) - 1, n_hands)):
            ha = acts[i + 1]
            while len(hands) <= i:
                occ = {}
                for p in _SHED_TILES:
                    occ[p] = sum(1 for q in [farmer] + hands if tuple(q) == p)
                hands.append(list(min(occ, key=lambda p: (occ[p], _SHED_TILES.index(p)))))
            if ha[0] in _MOVE:
                hands[i] = move(hands[i], ha[0])
            elif ha[0] == "BUILD_PASTURE":
                build_events.append((s, tuple(hands[i])))
            elif ha[0] == "PLANT":
                plant_events.append((s, tuple(hands[i])))

    build_pos = [set() for _ in range(n)]
    plant_pos = [set() for _ in range(n)]
    for s in range(n):
        for e in build_events:
            if s <= e[0] < s + 48:
                build_pos[s].add(e[1])
        for e in plant_events:
            if s <= e[0] < s + 48:
                plant_pos[s].add(e[1])
    return build_pos, plant_pos, build_events


def build_plan(route, lookahead=LOOKAHEAD):
    n = len(route)
    market_buys = {}
    animal_pace = {}  # animal -> day -> 購入ユニット数 (ルートのペース)
    for s, r in enumerate(route):
        day = s // 24
        for o in r.get("market", []):
            if o and o[0] == "BUY_ANIMAL" and len(o) >= 3 and o[1] in ANIMAL_COST:
                a = o[1]
                market_buys[a] = market_buys.get(a, 0) + int(o[2])
                pace = animal_pace.setdefault(a, {})
                pace[day] = pace.get(day, 0) + int(o[2])
    animals = {}
    for a, d in _animal_demand(route).items():
        # 数量は市場購入実績を上限 (PICKUP は失敗分を再試行するため過大に出る)
        animals[a] = {
            "target": min(d["target"], market_buys.get(a, 0)),
            "first_step": d["first_step"],
            "last_step": d["last_step"],
        }
    # 1日あたり購入数の累積 (資金が許せばルートと同じ日に、許さなければ後日キャッチアップ)
    animal_cum = {}
    for a, pace in animal_pace.items():
        cum = []
        total = 0
        for day in range(30):
            total += pace.get(day, 0)
            cum.append(total)
        animal_cum[a] = cum
    plan = {
        "plants": {c: _plants_window(route, c, lookahead) for c in CROPS4},
        "next_plant": {c: _next_plant_step(route, c) for c in CROPS4},
        "animals": animals,
        "animal_cum": animal_cum,
        "animal_total": sum(d["target"] for d in animals.values()),
        "seed_first_day": {c: None for c in CROPS4},
        "land_total": 0,
        "first_land_day": None,
        "hire_target": [0] * 30,
        "uses_fertilizer": False,
    }
    if M5_JIT_SEEDS:
        plan["plants_jit"] = {c: _plants_window(route, c, SEED_LOOKAHEAD_JIT) for c in CROPS4}
    plan["jit_seeds"] = M5_JIT_SEEDS
    plan["feed_floor"] = M5_FEED_FLOOR
    plan["offset"] = M5_OFFSET
    plan["route_hires"] = M5_ROUTE_HIRES
    plan["gap_late"] = M5_GAP_HIRE_LATE
    plan["gap_max"] = GAP_HIRE_MAX
    for s, r in enumerate(route):
        day = s // 24
        for o in r.get("market", []):
            if not o:
                continue
            op = o[0]
            if op == "HIRE":
                plan["hire_target"][day] += 1
            elif op == "BUY_LAND":
                plan["land_total"] += 1
                if plan["first_land_day"] is None:
                    plan["first_land_day"] = day
            elif op == "BUY_SEED" and o[1] in CROPS4 and plan["seed_first_day"][o[1]] is None:
                plan["seed_first_day"][o[1]] = day
    # M5b: ルートの HIRE オーダーのステップ分布 (orig と同時刻に雇用するため)
    hire_orders = [0] * n
    for s, r in enumerate(route):
        for o in r.get("market", []):
            if o and o[0] == "HIRE":
                hire_orders[s] += 1
    hire_cum = [0] * n
    c = 0
    for s in range(n):
        c += hire_orders[s]
        hire_cum[s] = c
    plan["hire_orders"] = hire_orders
    plan["hire_cum"] = hire_cum
    plan["gap_hour"] = [3] * 30
    for s, r in enumerate(route):
        for o in r.get("market", []):
            if o and o[0] == "HIRE":
                plan["gap_hour"][s // 24] = s % 24
    build_pos, plant_pos, build_events = _sim_positions(route, plan["hire_target"])
    plan["build_pos"] = build_pos
    plan["plant_pos"] = plant_pos
    plan["build_events"] = build_events
    pasture_steps = {}
    for s, pos in build_events:
        pasture_steps.setdefault(pos, s)
    plan["pasture_steps"] = pasture_steps
    plan["animal_ready_day"] = max(
        (plan["animals"][a]["last_step"] // 24 for a in plan["animals"]), default=99
    ) + 1
    for s, r in enumerate(route):
        for act in [r.get("farmer") or ["PASS"]] + [h or ["PASS"] for h in r.get("hands", [])]:
            if act and act[0] == "FERTILIZE":
                plan["uses_fertilizer"] = True
            if act and len(act) >= 3 and act[0] == "PICKUP" and act[1] == "FERTILIZER":
                plan["uses_fertilizer"] = True
        for o in r.get("market", []):
            if o and o[0] == "BUY_PRODUCT" and o[1] == "FERTILIZER":
                plan["uses_fertilizer"] = True
    return plan


def _is_weed(tile):
    return isinstance(tile, dict) and tile.get("kind") == "WEED"


def _step_toward(fx, fy, tx, ty):
    if fx > tx:
        return "WEST"
    if fx < tx:
        return "EAST"
    if fy > ty:
        return "NORTH"
    if fy < ty:
        return "SOUTH"
    return None


def _inventory_of(private, farm, pos):
    """ユニットの手持ち (ファーマー=0、ハンド=i+1)。"""
    idx = 0
    if tuple(pos) != tuple(farm["farmer"]):
        for i, h in enumerate(farm["hands"]):
            if tuple(h) == tuple(pos):
                idx = i + 1
                break
    invs = private.get("inventories", [])
    return invs[idx] if idx < len(invs) else {}


def _count_pastures(tiles):
    return sum(1 for row in tiles for t in row if isinstance(t, dict) and t.get("kind") == "PASTURE")


def _empty_pasture_pos(tiles):
    for y, row in enumerate(tiles):
        for x, t in enumerate(row):
            if isinstance(t, dict) and t.get("kind") == "PASTURE" and "animal" not in t:
                return (x, y)
    return None


def _shed_adjacent(pos):
    x, y = pos
    return x in (4, 5) and y in (4, 5)


def _reactive_animal_action(obs, farm, private, pos, plan):
    """ギャップフィラー (動物ロール): 給餌・配置をリアクティブに補完する。

    ルートの PICKUP ステップが資金遅延で失敗した場合、動物が shed に残り
    ミルク/羊毛を生まない。また、ルートの給餌スケジュールは自分の配置
    分しか想定しておらず、リアクティブ配置した動物は飢えて脱走する。
    処理順: (1) 未給餌動物への給餌 (脱走防止) (2) 滞留動物の配置 (3) 牧場建設
    """
    tiles = farm["tiles"]
    board = len(tiles)
    x, y = pos
    inv = _inventory_of(private, farm, pos)
    shed = private.get("shed", {}) or {}
    animal_total = plan.get("animal_total", 0)
    placed = sum(1 for row in tiles for t in row if isinstance(t, dict) and "animal" in t)
    carrying = [a for a in ("COW", "SHEEP", "GOOSE") if inv.get(a, 0) > 0]
    shed_animals = {a: shed.get(a, 0) for a in ("COW", "SHEEP", "GOOSE") if shed.get(a, 0) > 0}

    # --- (1) 手持ちの動物を空き牧場へ配置 (最優先: 持ったまま放置すると
    # 夜に shed へ戻されて配置が永遠に進まない) ---
    if carrying:
        a = carrying[0]
        ep = _empty_pasture_pos(tiles)
        if ep == pos:
            return ["PLACE", a, 1]
        if ep:
            step = _step_toward(x, y, ep[0], ep[1])
            if step:
                return [step]
        # 空き牧場が無ければ一旦 PASS (建設はルートの作物計画を圧迫するためしない。
        # ルート自身の牧場建設が後から配置機会を作る)
        return ["PASS"]

    # --- (2) 未給餌の配置動物への給餌 (脱走防止) ---
    unfed = [
        (tx, ty) for ty in range(board) for tx in range(board)
        if isinstance(tiles[ty][tx], dict) and "animal" in tiles[ty][tx]
        and not tiles[ty][tx].get("fed_today", False)
    ]
    if unfed:
        if inv.get("WHEAT", 0) > 0:
            ux, uy = min(unfed, key=lambda p: abs(x - p[0]) + abs(y - p[1]))
            if (ux, uy) == (x, y):
                return ["FEED"]
            step = _step_toward(x, y, ux, uy)
            if step:
                return [step]
        elif shed.get("WHEAT", 0) > 0:
            if _shed_adjacent(pos):
                return ["PICKUP", "WHEAT", 2]
            step = _step_toward(x, y, 4, 4)
            if step:
                return [step]

    # --- (3) 滞留動物のピックアップ (空き牧場 or 建設余地があるとき) ---
    if shed_animals:
        can_place = _empty_pasture_pos(tiles) is not None or _count_pastures(tiles) < max(animal_total, placed + sum(shed_animals.values()))
        if can_place:
            if _shed_adjacent(pos):
                a = max(shed_animals, key=lambda k: shed_animals[k])
                return ["PICKUP", a, 1]
            step = _step_toward(x, y, 4, 4)
            if step:
                return [step]

    # --- (4) 空きタイルに牧場を建設 (配置余地の確保) ---
    if shed_animals and _count_pastures(tiles) < placed + sum(shed_animals.values()):
        for ty in range(board):
            for tx in range(board):
                if tiles[ty][tx] is None:
                    if (tx, ty) == (x, y):
                        return ["BUILD_PASTURE"]
                    step = _step_toward(x, y, tx, ty)
                    if step:
                        return [step]
    return ["PASS"]


def _reactive_animal_action(obs, farm, private, pos, plan, day):
    """ギャップフィラー (動物専任・ルートの最終 PICKUP 後のみ使用)。

    滞留動物の配置と給餌を担当する。動物は2日連続の未給餌で脱走するため
    給餌を最優先する。
    """
    tiles = farm["tiles"]
    board = len(tiles)
    x, y = pos
    inv = _inventory_of(private, farm, pos)
    shed = private.get("shed", {}) or {}
    placed = sum(1 for row in tiles for t in row if isinstance(t, dict) and "animal" in t)
    carrying = [a for a in ("COW", "SHEEP", "GOOSE") if inv.get(a, 0) > 0]
    shed_animals = {a: shed.get(a, 0) for a in ("COW", "SHEEP", "GOOSE") if shed.get(a, 0) > 0}

    # 脱走リスクの高い動物 (1日以上未給餌) を最優先で給餌
    at_risk = [
        (tx, ty) for ty in range(board) for tx in range(board)
        if isinstance(tiles[ty][tx], dict) and "animal" in tiles[ty][tx]
        and not tiles[ty][tx].get("fed_today", False)
        and tiles[ty][tx].get("consecutive_unfed", 0) >= 1
    ]
    if at_risk:
        if inv.get("WHEAT", 0) > 0:
            ux, uy = min(at_risk, key=lambda p2: abs(x - p2[0]) + abs(y - p2[1]))
            if (ux, uy) == (x, y):
                return ["FEED"]
            step = _step_toward(x, y, ux, uy)
            if step:
                return [step]
        elif shed.get("WHEAT", 0) > 0:
            if _shed_adjacent(pos):
                return ["PICKUP", "WHEAT", 3]
            step = _step_toward(x, y, 4, 4)
            if step:
                return [step]

    # 手持ちの動物を配置 (空き牧場 → ルート意図位置に建設)
    if carrying:
        a = carrying[0]
        ep = _empty_pasture_pos(tiles)
        if ep == pos:
            return ["PLACE", a, 1]
        if ep:
            step = _step_toward(x, y, ep[0], ep[1])
            if step:
                return [step]
        plan_pos = None
        for (px, py), bstep in plan.get("pasture_steps", {}).items():
            if day * 24 <= bstep:
                continue
            t = tiles[py][px]
            if t is None or (isinstance(t, dict) and t.get("kind") == "WEED"):
                d = abs(x - px) + abs(y - py)
                if plan_pos is None or d < plan_pos[0]:
                    plan_pos = (d, px, py, t)
        if plan_pos:
            _, px, py, t = plan_pos
            if (px, py) == (x, y):
                return ["DIG"] if (isinstance(t, dict) and t.get("kind") == "WEED") else ["BUILD_PASTURE"]
            step = _step_toward(x, y, px, py)
            if step:
                return [step]
        return ["PASS"]

    # ピックアップ (空き牧場 or 建設余地あり)
    if shed_animals:
        can_place = _empty_pasture_pos(tiles) is not None or any(
            tiles[py][px] is None or (isinstance(tiles[py][px], dict) and tiles[py][px].get("kind") == "WEED")
            for (px, py) in plan.get("pasture_steps", {})
            if day * 24 > plan["pasture_steps"][(px, py)]
        )
        if can_place:
            if _shed_adjacent(pos):
                a = max(shed_animals, key=lambda k: shed_animals[k])
                return ["PICKUP", a, 1]
            step = _step_toward(x, y, 4, 4)
            if step:
                return [step]

    # 通常の未給餌動物も給餌 (配置済み動物の生存維持)
    unfed = [
        (tx, ty) for ty in range(board) for tx in range(board)
        if isinstance(tiles[ty][tx], dict) and "animal" in tiles[ty][tx]
        and not tiles[ty][tx].get("fed_today", False)
    ]
    if unfed:
        if inv.get("WHEAT", 0) > 0:
            ux, uy = min(unfed, key=lambda p2: abs(x - p2[0]) + abs(y - p2[1]))
            if (ux, uy) == (x, y):
                return ["FEED"]
            step = _step_toward(x, y, ux, uy)
            if step:
                return [step]
        elif shed.get("WHEAT", 0) > 0:
            if _shed_adjacent(pos):
                return ["PICKUP", "WHEAT", 3]
            step = _step_toward(x, y, 4, 4)
            if step:
                return [step]
    return ["PASS"]


def _reactive_hand_action(obs, farm, private, pos, day, plan=None, step=0):
    """ギャップフィラーハンド: 水やり・雑草除去を最優先し、手が空いたら
    滞留動物の配置を補完する (ルートの PICKUP 完了後のみ — 干渉防止)。

    ルートのハンドが位置ドリフトで水やりを漏らした作物の枯死を防ぐのが
    主目的。M3c: ルートが今後48ステップ以内に BUILD_PASTURE/PLANT する
    位置の雑草を優先除去し、牧場建設・植え付けの失敗を防ぐ。
    """
    tiles = farm["tiles"]
    board = len(tiles)
    x, y = pos
    tile = tiles[y][x]
    inv = _inventory_of(private, farm, pos)
    urgent = set()
    if plan and step < len(plan.get("build_pos", [])):
        urgent = plan["build_pos"][step] | plan["plant_pos"][step]

    # --- 現在地タイルの処理 (水やり > 雑草 > 作物収穫) ---
    if isinstance(tile, dict):
        if tile.get("kind") == "PLANT" and not tile.get("watered_today"):
            return ["WATER"]
        if tile.get("kind") == "WEED":
            return ["DIG"]
        if tile.get("kind") == "PLANT":
            crop = tile.get("crop")
            age = day - tile.get("planted_day", 0)
            if crop in ("STRAWBERRY", "WHEAT", "CARROT") and tile.get("yield_units", 0) > 0:
                fy = {"STRAWBERRY": 10, "WHEAT": 4, "CARROT": 3}[crop]
                if age >= fy:
                    return ["HARVEST"]
    best = None  # (dist, kind, urgent, tx, ty)
    for ty in range(board):
        for tx in range(board):
            t = tiles[ty][tx]
            if not isinstance(t, dict):
                continue
            if t.get("kind") == "PLANT" and not t.get("watered_today"):
                key = (abs(x - tx) + abs(y - ty), 0, 0, tx, ty)
            elif t.get("kind") == "WEED":
                key = (abs(x - tx) + abs(y - ty), 1, 0 if (tx, ty) in urgent else 1, tx, ty)
            else:
                continue
            if best is None or key < best:
                best = key
    if best:
        step = _step_toward(x, y, best[3], best[4])
        if step:
            return [step]

    # --- 収穫 (水やり/雑草が済んだら。メロンはルートのタイミングに任せる) ---
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        crop = tile.get("crop")
        age = day - tile.get("planted_day", 0)
        if crop in ("STRAWBERRY", "WHEAT", "CARROT") and tile.get("yield_units", 0) > 0:
            fy = {"STRAWBERRY": 10, "WHEAT": 4, "CARROT": 3}[crop]
            if age >= fy:
                return ["HARVEST"]
    for ty in range(board):
        for tx in range(board):
            t = tiles[ty][tx]
            if not (isinstance(t, dict) and t.get("kind") == "PLANT" and t.get("yield_units", 0) > 0):
                continue
            crop = t.get("crop")
            if crop not in ("STRAWBERRY", "WHEAT", "CARROT"):
                continue
            fy = {"STRAWBERRY": 10, "WHEAT": 4, "CARROT": 3}[crop]
            if day - t.get("planted_day", 0) >= fy:
                step = _step_toward(x, y, tx, ty)
                if step:
                    return [step]
                return ["HARVEST"]

    # --- 動物タスク (ルートの最終 PICKUP を過ぎてから。牧場はルート建設分のみ) ---
    if plan and plan.get("animals"):
        shed = private.get("shed", {}) or {}
        carrying = [a for a in ("COW", "SHEEP", "GOOSE") if inv.get(a, 0) > 0]
        shed_animals = {a: shed.get(a, 0) for a in ("COW", "SHEEP", "GOOSE") if shed.get(a, 0) > 0}
        last_pickup_day = max(
            (plan["animals"][a]["last_step"] // 24 for a in plan["animals"]), default=-1
        )
        if day > last_pickup_day:
            if carrying:
                a = carrying[0]
                ep = _empty_pasture_pos(tiles)
                if ep == pos:
                    return ["PLACE", a, 1]
                if ep:
                    step = _step_toward(x, y, ep[0], ep[1])
                    if step:
                        return [step]
                # 空き牧場が無ければルートの意図した牧場位置に建設
                # (建設ステップを過ぎて、タイルが空/雑草の場合のみ)
                plan_pos = None
                for (px, py), bstep in plan.get("pasture_steps", {}).items():
                    if step <= bstep:
                        continue
                    t = tiles[py][px]
                    if t is None or (isinstance(t, dict) and t.get("kind") == "WEED"):
                        d = abs(x - px) + abs(y - py)
                        if plan_pos is None or d < plan_pos[0]:
                            plan_pos = (d, px, py, t)
                if plan_pos:
                    _, px, py, t = plan_pos
                    if (px, py) == (x, y):
                        return ["DIG"] if (isinstance(t, dict) and t.get("kind") == "WEED") else ["BUILD_PASTURE"]
                    step_ = _step_toward(x, y, px, py)
                    if step_:
                        return [step_]
                return ["PASS"]
            if shed_animals and _empty_pasture_pos(tiles) is not None:
                if _shed_adjacent(pos):
                    a = max(shed_animals, key=lambda k: shed_animals[k])
                    return ["PICKUP", a, 1]
                step = _step_toward(x, y, 4, 4)
                if step:
                    return [step]
    return ["PASS"]


def _repair(action, pos, tiles):
    if not action:
        return ["PASS"]
    op = action[0]
    fx, fy = pos
    tile = tiles[fy][fx]
    if op in ("PLANT", "BUILD_COOP", "BUILD_PASTURE") and _is_weed(tile):
        return ["DIG"]
    return action


def _count_placed_animals(tiles):
    cnt = {}
    for row in tiles:
        for t in row:
            if isinstance(t, dict) and "animal" in t:
                cnt[t["animal"]] = cnt.get(t["animal"], 0) + 1
    return cnt


def _total_animals(shed, invs, placed):
    n = 0
    for a in ("GOOSE", "COW", "SHEEP"):
        n += shed.get(a, 0)
        n += sum(inv.get(a, 0) for inv in invs)
    n += sum(placed.values())
    return n


def _reactive_market(obs, farm, private, plan, step, day, hour):
    prices = (obs.get("market", {}) or {}).get("prices", {})
    shed = private.get("shed", {}) or {}
    seeds = private.get("seeds", {}) or {}
    invs = private.get("inventories", []) or []
    money = farm.get("money", 0)
    placed = _count_placed_animals(farm["tiles"])
    n_animals = _total_animals(shed, invs, placed)
    market = []

    # --- 売却 (最優先) ---
    for item in SELL_ORDER:
        n = shed.get(item, 0)
        if n <= 0:
            continue
        if item == "WHEAT":
            # ルートのユニットが毎日10-36個を PICKUP するため、shed 在庫を
            # 残して売る (実在動物数のフィードも確保)。売りすぎない範囲で
            # d5-9 の資金源にする (雇用が止まると水やり崩壊で作物が枯れる)
            reserve = max(6, n_animals * FEED_RESERVE_PER_ANIMAL)
            n = max(0, n - reserve)
            if n <= 0:
                continue
        market.append(["SELL", item, min(SELL_CHUNK[item], n)])

    # --- 雇用 (hour1-3 が上位のパターン。ルートの1日当たり雇用数が上限。
    # hour1 で資金が足りなくても hour2 で再試行される。
    # M3: ルート目標に達したらギャップフィラーを最大2人追加) ---
    # M5b: ルートの HIRE オーダーを orig と同じステップで再現する
    # (ハンドのスポーン位置一致)。ギャップフィラーはルートの当日分の雇用が
    # 完了する hour を過ぎてから。
    if plan.get("route_hires") and "hire_cum" in plan:
        n_route = len(plan["hire_cum"])
        rs = min(step + 1, n_route - 1)
        day_start = day * 24
        cum_before = plan["hire_cum"][day_start - 1] if day_start > 0 else 0
        target = plan["hire_cum"][rs] - cum_before
        n_today = farm.get("hires_today", 0)
        while n_today < target:
            cost = _fib(n_today)
            if money < cost:
                break
            market.append(["HIRE"])
            n_today += 1
            money -= cost
        if plan.get("gap_late") and hour > plan["gap_hour"][day]:
            cap = plan["hire_target"][day] + plan["gap_max"]
            while n_today < cap:
                cost = _fib(n_today)
                if money < cost:
                    break
                market.append(["HIRE"])
                n_today += 1
                money -= cost
    elif hour in (1, 2, 3) and day < len(plan["hire_target"]):
        n_today = farm.get("hires_today", 0)
        cap = plan["hire_target"][day] - n_today + GAP_HIRE_MAX
        while cap > 0 and n_today < plan["hire_target"][day] + GAP_HIRE_MAX:
            cost = _fib(n_today)
            if money < cost:
                break
            market.append(["HIRE"])
            n_today += 1
            cap -= 1
            money -= cost

    # --- フィード用小麦の購入 (shed 在庫ターゲット管理: ルートの PICKUP 需要を満たす。
    # ルートは一日中 PICKUP するため常時補充が必要) ---)
    # M5: 資金フロアを下回る間は購入を止める (収穫小麦に任せる)。元プレイヤーは
    # d4-6 の資金難期に小麦購入0で、購入は資金に余裕がある日に集中している。
    shed_wheat = shed.get("WHEAT", 0)
    if shed_wheat < WHEAT_STOCK_TARGET and (not plan.get("feed_floor") or money >= FEED_CASH_FLOOR):
        wheat_price = prices.get("WHEAT", 25)
        buy = min(WHEAT_STOCK_TARGET - shed_wheat, 4, int(money) // max(1, wheat_price))
        if buy > 0:
            market.append(["BUY_PRODUCT", "WHEAT", buy])
            money -= buy * wheat_price

    # --- 動物購入 (ルート農場が必要とする品目。累積ペースで購入、種より優先:
    # 動物は日次ゲートで逃すと配置機会が失われる (E018-M3c の検証) ---)
    if plan["animals"]:
        for a in sorted(plan["animals"], key=lambda x: plan["animals"][x]["first_step"]):
            d = plan["animals"][a]
            if day < d["first_step"] // 24:
                continue
            need_by_today = plan["animal_cum"][a][day] if day < 30 else d["target"]
            owned = shed.get(a, 0) + placed.get(a, 0) + sum(inv.get(a, 0) for inv in invs)
            while owned < min(need_by_today, d["target"]) and money >= ANIMAL_COST[a] + ANIMAL_BUFFER:
                market.append(["BUY_ANIMAL", a, 1])
                owned += 1
                money -= ANIMAL_COST[a]

    # --- 種の補充 (hour2 以降: hour0-1 の種購入が雇用資金を食い潰し、
    # ルートと雇用数がずれてハンドのスポーン位置がドリフトするのを防ぐ) ---
    # 購入日はルートの最初の種購入日以降。植え付けが近い作物から優先。
    # M5 JIT: 需要ウィンドウを「次の SEED_LOOKAHEAD_JIT ステップ」に縮小。
    # 種は今ステップのユニット行動後に購入されるため、s+1 以降の植え付けに
    # 間に合えばよい (大量前買いで d0-9 の資金を枯渇させない)。
    if hour >= 2 and step < len(plan["plants"]["WHEAT"]):
        n = len(plan["plants"]["WHEAT"])
        for crop in sorted(CROPS4, key=lambda c: (plan["next_plant"][c][step], c)):
            if day < plan["seed_first_day"][crop]:
                continue
            if plan.get("jit_seeds"):
                off = 2 if plan.get("offset") else 1
                need = plan["plants_jit"][crop][min(step + off, n - 1)] + SEED_BUFFER
                max_buy = SEED_MAX_BUY_JIT
            else:
                need = plan["plants"][crop][step] + SEED_BUFFER
                max_buy = SEED_MAX_BUY
            have = seeds.get(crop, 0)
            if have >= need or money < SEED_COST[crop]:
                continue
            buy = min(need - have, max_buy, int(money) // SEED_COST[crop])
            if buy > 0:
                market.append(["BUY_SEED", crop, buy])
                money -= buy * SEED_COST[crop]

    # --- 土地購入 (day5+ のタイミング、資金ゲート付き) ---
    if plan["land_total"] > 0 and plan["first_land_day"] is not None:
        n_unlocked = len(farm.get("unlocked_quadrants", ["NW"])) - 1
        if n_unlocked < plan["land_total"] and day >= plan["first_land_day"] + n_unlocked * 5:
            from kaggle_environments.envs.kaggriculture.kaggriculture import LAND_PRICES
            if money >= LAND_PRICES[n_unlocked] + LAND_BUFFER:
                market.append(["BUY_LAND"])
                money -= LAND_PRICES[n_unlocked]

    # --- 肥料の仕入れ (ルートが肥料を使う場合。セットアップ期は資金を優先) ---
    if plan["uses_fertilizer"] and day >= 5 and shed.get("FERTILIZER", 0) < 2 and money >= FERT_BUFFER:
        market.append(["BUY_PRODUCT", "FERTILIZER", 1])

    # --- 10件制限のトリム + 売却順序の最適化 (価格影響の大きい売りを先に) ---
    market = market[:10]
    # E018-M2 検証: 同シード A/B で -$3.6k (ノイズ範囲内) のため既定オフ
    return _rank_sell_orders(obs, market) if RANK_SELL_ORDERS else market


def make_adaptive_agent(route, lookahead=LOOKAHEAD):
    plan = build_plan(route, lookahead)

    def agent(obs):
        try:
            farms = obs.get("farms", [])
            player = obs.get("player", 0)
            private = obs.get("private", {}) or {}
            if not farms or player >= len(farms):
                return {"farmer": ["PASS"], "hands": [], "market": []}
            farm = farms[player]
            step = obs.get("step", obs.get("day", 0) * 24 + obs.get("hour", 0))
            day = obs.get("day", 0)
            hour = obs.get("hour", 0)
            if step >= len(route):
                return {"farmer": ["PASS"], "hands": [], "market": []}
            # M5b: ルートの steps[k].action は obs[k-1] に応答した行動のため、
            # obs[k] では route[k+1] を提出する (1ステップ遅れの修正)。
            rs = min(step + 1, len(route) - 1) if plan.get("offset") else step
            r = route[rs] if rs < len(route) else {"farmer": ["PASS"], "hands": []}
            farmer = _repair(r.get("farmer", ["PASS"]), farm["farmer"], farm["tiles"])
            actual_hands = farm.get("hands", [])
            hands = []
            for i, ha in enumerate(r.get("hands", [])):
                if i >= len(actual_hands):
                    break
                hands.append(_repair(ha, actual_hands[i], farm["tiles"]))
            # M3: ルート計画外の余剰ハンドはギャップフィラー
            # (水やり/雑草優先、手が空いたら動物配置を補完)
            while len(hands) < len(actual_hands):
                hands.append(_reactive_hand_action(obs, farm, private, actual_hands[len(hands)], day, plan, step))
            market = _reactive_market(obs, farm, private, plan, step, day, hour)
            return {"farmer": farmer, "hands": hands, "market": market}
        except Exception:
            return {"farmer": ["PASS"], "hands": [], "market": []}
    return agent