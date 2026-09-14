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
# M5e: ルートの BUY_ANIMAL を orig と同時刻に再現 (PICKUP 空振り防止)。
M5_ROUTE_ANIMALS = False
# M5f: ルートの小麦購入も orig と同時刻に再現 (給餌用小麦の枯渇防止)。
# FEED_CASH_FLOOR は資金難期に購入を止めすぎ、19頭の給餌需要に追いつかず
# GOOSE が脱走していた (orig は d7 に8個購入 vs 我々1個)。
M5_ROUTE_WHEAT = False
# M5h: 最終日 (d29) の小麦リザーブ解放 (当日給餌分のみ残して売却)
M5_ENDGAME_LIQUIDATE = True
# M5i: 価格暴落時は売却を控え、価格回復を待つ (実戦で WOOL が $1 まで暴落しても
# 売り続けていた。town 需要で $144 まで回復するため、保有が大幅有利)。
M5_PRICE_FLOOR = False
SELL_PRICE_FLOOR_FRAC = 0.5
# M6: 終盤イチゴ植え替え — ルートの PASS ターンを「その場の PLANT STRAWBERRY」に
# 静的に書き換える (移動なし=位置ドリフトなし。種は JIT が自動購入)。
# d14-18 に植えると d24-28 に収穫でき、ルート自体が終盤に畳むイチゴ畑を延長する
# (E018-M5g の敗戦分析: 敵は終盤も15株維持し $16-18k 稼いだ)。
M6_GAP_STRAWBERRY = False
M6_GAP_STRAWBERRY_DAYS = (14, 18)
M6_GAP_STRAWBERRY_MAX = 10
# M6b: ルートの d15-17 の WHEAT PLANT を STRAWBERRY に置換 (同一ユニット・同一位置・
# 同一タイミング)。ルート自身の水やり/施肥/収穫スケジュールがそのまま効くため
# 専任プランター方式 (M6) と違い枯死しない。終盤 (d24-28) のイチゴ生産を延長する
M6_SWAP_WHEAT = False
M6_SWAP_DAYS = (15, 18)
M6_SWAP_MAX = 12
# M5k: 肥料の購入を止める (最大のボトルネック)。動物19頭から毎日無料で
# 肥料が採れるのに、市場が「shed<2 なら常に買う」ため $100 で買って $60 で売る
# 往復損失を 454個/試合 ($45.4k) も続けていた。ルート自身の計画でも購入ゼロ。
M5_NO_FERT_BUY = True
# M6d: ギャップフィラーは農場が大きい時期 (d5-24) のみ雇用。
# 終盤 (d25-29) は作物が枯渇し水やり負荷が下がるため、fib(12)+fib(13)=$610/日の
# フィラー2人は過剰 (ルート生成・労働効率化の第一歩)
M6_GAP_WINDOW = True
M6_GAP_WINDOW_DAYS = (3, 22)
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


def _gap_strawberry_tile_ok(pos, s, plant_pos, build_pos):
    x, y = pos
    if not (0 <= x < 10 and 0 <= y < 10):
        return False
    if x in (4, 5) and y in (4, 5):
        return False
    if x >= 5 and y >= 5:
        return False
    if pos in plant_pos[s] or pos in build_pos[s]:
        return False
    return True


def _rewrite_gap_strawberries(route):
    """終盤の PASS を「その場の PLANT STRAWBERRY」に書き換える (M6)。

    位置シミュレーション上で d14-18 の PASS を見つけ、そのタイルが
    (a) ルートの今後48ステップの PLANT/BUILD 位置でない (b) shed 隣接でない
    (c) SE 区画でない 場合に書き換える。PLANT は移動を伴わないため位置ドリフト
    ゼロ。タイルが実際は占有されていても無音で失敗するだけ (安全)。
    """
    if not M6_GAP_STRAWBERRY:
        return route
    d0, d1 = M6_GAP_STRAWBERRY_DAYS
    n = len(route)
    hire = [0] * 30
    for s, r in enumerate(route):
        for o in r.get("market", []):
            if o and o[0] == "HIRE":
                hire[s // 24] += 1
    build_pos, plant_pos, _ = _sim_positions(route, hire)
    new_route = [
        {
            "farmer": list(r.get("farmer") or ["PASS"]),
            "hands": [list(h) for h in r.get("hands", [])],
            "market": r.get("market", []),
        }
        for r in route
    ]
    farmer = [4, 4]
    hands = []
    count = 0
    for s in range(n):
        hour = s % 24
        day = s // 24
        if hour == 0:
            farmer = [4, 4]
            hands = []
        n_hands = hire[day]

        def move(p, op):
            dx, dy = _MOVE[op]
            return [p[0] + dx, p[1] + dy]

        fa = new_route[s]["farmer"]
        if d0 <= day < d1 and fa[0] == "PASS" and count < M6_GAP_STRAWBERRY_MAX:
            if _gap_strawberry_tile_ok(tuple(farmer), s, plant_pos, build_pos):
                new_route[s]["farmer"] = ["PLANT", "STRAWBERRY"]
                fa = new_route[s]["farmer"]
                count += 1
        if fa[0] in _MOVE:
            farmer = move(farmer, fa[0])
        acts = new_route[s]["hands"]
        for i in range(min(len(acts), n_hands)):
            while len(hands) <= i:
                occ = {}
                for p in _SHED_TILES:
                    occ[p] = sum(1 for q in [farmer] + hands if tuple(q) == p)
                hands.append(list(min(occ, key=lambda p: (occ[p], _SHED_TILES.index(p)))))
            ha = acts[i]
            if d0 <= day < d1 and ha[0] == "PASS" and count < M6_GAP_STRAWBERRY_MAX:
                if _gap_strawberry_tile_ok(tuple(hands[i]), s, plant_pos, build_pos):
                    new_route[s]["hands"][i] = ["PLANT", "STRAWBERRY"]
                    ha = new_route[s]["hands"][i]
                    count += 1
            if ha[0] in _MOVE:
                hands[i] = move(hands[i], ha[0])
    return new_route


def _rewrite_swap_wheat_strawberry(route):
    """ルートの d15-17 の WHEAT PLANT を STRAWBERRY に置換 (M6b)。

    同一ユニット・同一位置・同一ステップで作物だけ入れ替えるため、位置ドリフト
    ゼロ。置換後のタイルはルート自身の水やりスケジュールで維持される。
    """
    if not M6_SWAP_WHEAT:
        return route
    d0, d1 = M6_SWAP_DAYS
    new_route = [
        {
            "farmer": list(r.get("farmer") or ["PASS"]),
            "hands": [list(h) for h in r.get("hands", [])],
            "market": r.get("market", []),
        }
        for r in route
    ]
    count = 0
    for s, r in enumerate(new_route):
        day = s // 24
        if not (d0 <= day < d1):
            continue
        fa = r["farmer"]
        if count < M6_SWAP_MAX and fa[0] == "PLANT" and fa[1] == "WHEAT":
            r["farmer"] = ["PLANT", "STRAWBERRY"]
            count += 1
        for i, h in enumerate(r["hands"]):
            if count < M6_SWAP_MAX and h[0] == "PLANT" and h[1] == "WHEAT":
                r["hands"][i] = ["PLANT", "STRAWBERRY"]
                count += 1
    return new_route


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
    plan["feed_cash_floor"] = FEED_CASH_FLOOR
    plan["offset"] = M5_OFFSET
    plan["route_hires"] = M5_ROUTE_HIRES
    plan["gap_late"] = M5_GAP_HIRE_LATE
    plan["gap_max"] = GAP_HIRE_MAX
    plan["route_animals"] = M5_ROUTE_ANIMALS
    plan["n"] = n
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
    # 一度も買われない作物は 99 (事実上無限) にする — None のままだと
    # 市場ロジックの `day < seed_first_day` 比較が TypeError になり agent 全体が
    # PASS フォールバックする (E018 強敵評価で Rocket Zech 等のルートが崩壊した原因)
    for c in CROPS4:
        if plan["seed_first_day"][c] is None:
            plan["seed_first_day"][c] = 99
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
    # M5e: ルートの BUY_ANIMAL オーダーのステップ分布 (orig と同じタイミングで
    # 購入し、PICKUP に間に合わせる。first_step 順のペース購入では品目間の
    # 優先が orig と逆転して PICKUP が空振りする — GOOSE の d6h3 PICKUP 等)
    animal_buy_steps = {}
    for s, r in enumerate(route):
        for o in r.get("market", []):
            if o and o[0] == "BUY_ANIMAL" and o[1] in ANIMAL_COST:
                animal_buy_steps.setdefault(s, []).append((o[1], int(o[2]) if len(o) > 2 else 1))
    plan["animal_buy_steps"] = animal_buy_steps
    # M5f: ルートの小麦購入 (BUY_PRODUCT WHEAT) のステップ分布
    wheat_buy_steps = {}
    for s, r in enumerate(route):
        for o in r.get("market", []):
            if o and o[0] == "BUY_PRODUCT" and o[1] == "WHEAT":
                wheat_buy_steps[s] = wheat_buy_steps.get(s, 0) + (int(o[2]) if len(o) > 2 else 1)
    plan["wheat_buy_steps"] = wheat_buy_steps
    plan["route_wheat"] = M5_ROUTE_WHEAT
    plan["endgame_liquidate"] = M5_ENDGAME_LIQUIDATE
    plan["price_floor"] = M5_PRICE_FLOOR
    plan["m6_gap"] = M6_GAP_STRAWBERRY
    plan["m6_swap"] = M6_SWAP_WHEAT
    plan["no_fert_buy"] = M5_NO_FERT_BUY
    plan["gap_window"] = M6_GAP_WINDOW
    plan["gap_window_days"] = M6_GAP_WINDOW_DAYS
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


def _reactive_hand_action(obs, farm, private, pos, day, plan=None, step=0, is_planter=False):
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
    # M6: 植え付け専任ハンド — 水やりより植えを優先 (最後のギャップハンドのみ)。
    if is_planter and plan and plan.get("m6_gap") and M6_GAP_STRAWBERRY_DAYS[0] <= day < M6_GAP_STRAWBERRY_DAYS[1] \
            and _m6_gap_count(farm) < M6_GAP_STRAWBERRY_MAX:
        # (1) ギャップイチゴの水やり最優先 (植えた当日に水をやらないと枯死する)
        need = [(tx, ty) for ty in range(board) for tx in range(board)
                if isinstance(tiles[ty][tx], dict)
                and tiles[ty][tx].get("kind") == "PLANT"
                and tiles[ty][tx].get("crop") == "STRAWBERRY"
                and tiles[ty][tx].get("planted_day", 0) >= 14
                and not tiles[ty][tx].get("watered_today")]
        if need:
            ux, uy = min(need, key=lambda p2: abs(x - p2[0]) + abs(y - p2[1]))
            if (ux, uy) == (x, y):
                return ["WATER"]
            step_ = _step_toward(x, y, ux, uy)
            if step_:
                return [step_]
        # (2) 新規植え
        best = None
        for ty in range(board):
            for tx in range(board):
                t = tiles[ty][tx]
                if t is not None:
                    continue
                if tx in (4, 5) and ty in (4, 5):
                    continue
                if tx >= 5 and ty >= 5:
                    continue
                if plan and step < len(plan.get("plant_pos", [])):
                    if (tx, ty) in plan["plant_pos"][step] or (tx, ty) in plan["build_pos"][step]:
                        continue
                d2 = abs(x - tx) + abs(y - ty)
                if best is None or d2 < best[0]:
                    best = (d2, tx, ty)
        if best:
            _, tx, ty = best
            if (tx, ty) == (x, y):
                return ["PLANT", "STRAWBERRY"]
            step_ = _step_toward(x, y, tx, ty)
            if step_:
                return [step_]
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

    # --- M6: ギャップイチゴ植え (水やり・雑草の次に優先。最寄りの空きタイルへ) ---
    # リアクティブハンドは自由移動のため、ルートの位置スケジュールを壊さない。
    if (plan and plan.get("m6_gap") and M6_GAP_STRAWBERRY_DAYS[0] <= day < M6_GAP_STRAWBERRY_DAYS[1]
            and _m6_gap_count(farm) < M6_GAP_STRAWBERRY_MAX):
        best = None
        for ty in range(board):
            for tx in range(board):
                t = tiles[ty][tx]
                if t is not None:
                    continue
                if tx in (4, 5) and ty in (4, 5):
                    continue
                if tx >= 5 and ty >= 5:
                    continue
                if plan and step < len(plan.get("plant_pos", [])):
                    if (tx, ty) in plan["plant_pos"][step] or (tx, ty) in plan["build_pos"][step]:
                        continue
                d2 = abs(x - tx) + abs(y - ty)
                if best is None or d2 < best[0]:
                    best = (d2, tx, ty)
        if best:
            _, tx, ty = best
            if (tx, ty) == (x, y):
                return ["PLANT", "STRAWBERRY"]
            step_ = _step_toward(x, y, tx, ty)
            if step_:
                return [step_]

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
    # M5i: 暴落中の商品は保有して回復を待つ (最終日は強制売却)。
    for item in SELL_ORDER:
        n = shed.get(item, 0)
        if n <= 0:
            continue
        if plan.get("price_floor") and day < 29 and item in _PRICE_PARAMS:
            base = _PRICE_PARAMS[item]["base"]
            if prices.get(item, base) < SELL_PRICE_FLOOR_FRAC * base:
                continue
        if item == "WHEAT":
            # ルートのユニットが毎日10-36個を PICKUP するため、shed 在庫を
            # 残して売る (実在動物数のフィードも確保)。売りすぎない範囲で
            # d5-9 の資金源にする (雇用が止まると水やり崩壊で作物が枯れる)
            # M5h: 最終日 (d29) は当日分の給餌だけ残して全て売却する。
            # 未給餌は脱走 (2日連続) せず基本生産も入るため、CARE ボーナスと
            # 当日卵の分だけ残せばよい (従来は 38 個が死蔵されていた = ~$1.5k)
            if plan.get("endgame_liquidate") and day >= 29:
                reserve = sum(placed.values())
            else:
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
        if (plan.get("gap_late") and hour > plan["gap_hour"][day]
                and (not plan.get("gap_window")
                     or plan["gap_window_days"][0] <= day <= plan["gap_window_days"][1])):
            cap = plan["hire_target"][day] + plan["gap_max"]
            if plan.get("m6_gap") and M6_GAP_STRAWBERRY_DAYS[0] <= day < M6_GAP_STRAWBERRY_DAYS[1]:
                cap += 1
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
    # M5f: ルートの小麦購入オーダーを orig と同時刻に再現 (給餌需要 19個/日に
    # 追いつかせ、GOOSE の脱走を防ぐ)。shed < 3 のときはフロア例外で枯渇防止。
    shed_wheat = shed.get("WHEAT", 0)
    if plan.get("route_wheat") and "wheat_buy_steps" in plan:
        n = plan.get("n", len(plan["plants"]["WHEAT"]))
        rs = min(step + 1, n - 1)
        qty = plan["wheat_buy_steps"].get(rs, 0)
        if qty > 0:
            wheat_price = prices.get("WHEAT", 25)
            buy = min(qty, int(money) // max(1, wheat_price))
            if buy > 0:
                market.append(["BUY_PRODUCT", "WHEAT", buy])
                money -= buy * wheat_price
    if shed_wheat < WHEAT_STOCK_TARGET and (not plan.get("feed_floor") or money >= plan["feed_cash_floor"]):
        wheat_price = prices.get("WHEAT", 25)
        buy = min(WHEAT_STOCK_TARGET - shed_wheat, 4, int(money) // max(1, wheat_price))
        if buy > 0:
            market.append(["BUY_PRODUCT", "WHEAT", buy])
            money -= buy * wheat_price

    # --- 動物購入 (ルート農場が必要とする品目。累積ペースで購入、種より優先:
    # 動物は日次ゲートで逃すと配置機会が失われる (E018-M3c の検証) ---)
    # M5e: ルートの BUY_ANIMAL オーダーを orig と同じステップで再現する
    # (first_step 順のペース購入は品目間の優先が orig と逆転し、GOOSE の
    # d6h3 PICKUP 等が空振りして配置が 18/18 → 15/18 に落ちていた)。
    # キャッチアップ (資金不足で逃した分) は hour>=12 に当日分のみ。
    if plan["animals"]:
        n = plan.get("n", len(plan["plants"]["WHEAT"]))
        if plan.get("route_animals") and "animal_buy_steps" in plan:
            owned = {a: shed.get(a, 0) + placed.get(a, 0) + sum(inv.get(a, 0) for inv in invs)
                     for a in plan["animals"]}
            rs = min(step + 1, n - 1)
            for a, qty in plan["animal_buy_steps"].get(rs, []):
                d = plan["animals"][a]
                while qty > 0 and owned.get(a, 0) < d["target"] and money >= ANIMAL_COST[a] + ANIMAL_BUFFER:
                    market.append(["BUY_ANIMAL", a, 1])
                    owned[a] = owned.get(a, 0) + 1
                    qty -= 1
                    money -= ANIMAL_COST[a]
            if hour >= 12:
                for a in sorted(plan["animals"], key=lambda x: plan["animals"][x]["first_step"]):
                    d = plan["animals"][a]
                    if day < d["first_step"] // 24:
                        continue
                    need_by_today = plan["animal_cum"][a][day] if day < 30 else d["target"]
                    o = owned.get(a, 0)
                    while o < min(need_by_today, d["target"]) and money >= ANIMAL_COST[a] + ANIMAL_BUFFER:
                        market.append(["BUY_ANIMAL", a, 1])
                        o += 1
                        money -= ANIMAL_COST[a]
        else:
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

    # --- M6: ギャップイチゴの種の補充 (窓内で種2個未満なら1個買う) ---
    if (plan.get("m6_gap") and M6_GAP_STRAWBERRY_DAYS[0] <= day < M6_GAP_STRAWBERRY_DAYS[1]
            and seeds.get("STRAWBERRY", 0) < 2
            and _m6_gap_count(farm) < M6_GAP_STRAWBERRY_MAX
            and money >= SEED_COST["STRAWBERRY"] + 50):
        market.append(["BUY_SEED", "STRAWBERRY", 1])
        money -= SEED_COST["STRAWBERRY"]

    # --- 土地購入 (day5+ のタイミング、資金ゲート付き) ---
    if plan["land_total"] > 0 and plan["first_land_day"] is not None:
        n_unlocked = len(farm.get("unlocked_quadrants", ["NW"])) - 1
        if n_unlocked < plan["land_total"] and day >= plan["first_land_day"] + n_unlocked * 5:
            from kaggle_environments.envs.kaggriculture.kaggriculture import LAND_PRICES
            if money >= LAND_PRICES[n_unlocked] + LAND_BUFFER:
                market.append(["BUY_LAND"])
                money -= LAND_PRICES[n_unlocked]

    # --- 肥料の仕入れ (ルートが肥料を使う場合。セットアップ期は資金を優先) ---
    # M5k: 購入停止 — 動物由来の肥料 (19個/日) でルートの施肥需要は賄える。
    # 購入すると $100 で仕入れて $60 で売る往復損失になる (実測 $45.4k/試合)
    if (not plan.get("no_fert_buy") and plan["uses_fertilizer"] and day >= 5
            and shed.get("FERTILIZER", 0) < 2 and money >= FERT_BUFFER):
        market.append(["BUY_PRODUCT", "FERTILIZER", 1])

    # --- 10件制限のトリム + 売却順序の最適化 (価格影響の大きい売りを先に) ---
    market = market[:10]
    # E018-M2 検証: 同シード A/B で -$3.6k (ノイズ範囲内) のため既定オフ
    return _rank_sell_orders(obs, market) if RANK_SELL_ORDERS else market


def _m6_gap_count(farm):
    return sum(1 for row in farm["tiles"] for t in row
               if isinstance(t, dict) and t.get("kind") == "PLANT"
               and t.get("crop") == "STRAWBERRY" and t.get("planted_day", 0) >= 14)


def _m6_plant_override(pos, farm, plan, step):
    """PASS の代わりにその場に PLANT STRAWBERRY (移動なし=ドリフトなし)。"""
    x, y = pos
    tile = farm["tiles"][y][x]
    if tile is not None:
        return None
    if x in (4, 5) and y in (4, 5):
        return None
    if x >= 5 and y >= 5:
        return None
    if step < len(plan.get("plant_pos", [])):
        if (x, y) in plan["plant_pos"][step] or (x, y) in plan["build_pos"][step]:
            return None
    return ["PLANT", "STRAWBERRY"]


def make_adaptive_agent(route, lookahead=LOOKAHEAD):
    if M6_SWAP_WHEAT:
        route = _rewrite_swap_wheat_strawberry(route)
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
            m6_day = M6_GAP_STRAWBERRY_DAYS
            m6_active = (M6_GAP_STRAWBERRY and m6_day[0] <= day < m6_day[1]
                         and _m6_gap_count(farm) < M6_GAP_STRAWBERRY_MAX)
            if m6_active and farmer[0] == "PASS":
                ov = _m6_plant_override(farm["farmer"], farm, plan, step)
                if ov:
                    farmer = ov
            for i, ha in enumerate(r.get("hands", [])):
                if i >= len(actual_hands):
                    break
                act = _repair(ha, actual_hands[i], farm["tiles"])
                if m6_active and act[0] == "PASS":
                    ov = _m6_plant_override(actual_hands[i], farm, plan, step)
                    if ov:
                        act = ov
                hands.append(act)
            # M3: ルート計画外の余剰ハンドはギャップフィラー
            # (水やり/雑草優先、手が空いたら動物配置を補完)
            while len(hands) < len(actual_hands):
                is_last = len(hands) == len(actual_hands) - 1
                hands.append(_reactive_hand_action(obs, farm, private, actual_hands[len(hands)], day, plan, step, is_planter=is_last))
            market = _reactive_market(obs, farm, private, plan, step, day, hour)
            return {"farmer": farmer, "hands": hands, "market": market}
        except Exception:
            return {"farmer": ["PASS"], "hands": [], "market": []}
    return agent