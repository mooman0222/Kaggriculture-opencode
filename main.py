"""Kaggriculture agent — Melon farm with market-aware trading.

ベース戦略は Kaggle "Kaggriculture: Getting Started" (bovard/kaggriculture-getting-started)
の melon_maxxer を土台に、ノートブックで指摘されている課題を改良した版:

  * 収穫: first_yield_day (10日目) で収量が十分なら即収穫 → シーズン中に3サイクル目が可能
  * 優先順位: 収穫 > 水やり > 雑草除去 > 植え付け、近いタスクへ巡回
  * 売却: 1ターンあたり SELL_CHUNK 個まで分割売却 (価格クラッシュを緩和)
  * 売却: 納屋が溢れそうなら閾値以下でも売却 / 終盤 (day >= 28) は強制換金
  * 植え付け: メロン価格が閾値を下回ったら停止 (供給過多を自己規制)
  * 拡張: 余裕資金があればファームハンドを雇用 (1日最大2人)
  * 小麦フェーズ (E008): メロン価格が閾値を割ったら空きタイルを小麦へ転換し、
    価格安定の小麦を常時分割売却 (メロン市場頭打ち後の安定収入)
  * 役割分担 (E008): ファーマーは植え付け優先、ハンドは水やり優先で分散。
    小麦モードではハンドを常時雇用 (1日2ドル) し、メロンは MAX_MELON_TILES に制限
  * 動物 (E009): COW 2頭 + SHEEP 1頭を飼育し、ミルク/羊毛/肥料 (1頭/日で無料生成) を
    売却。動物タスクはハンドが最優先で担当 (フィード > 収穫 > 肥料回収 > ケア)。
    vs base +$12.9k / mirror +$10k の最大の改善

検証済みの知見 (ローカル A/B テスト、5試合平均):
  * メロン市場は約70個で頭打ち → 土地購入は売上を増やさず損になるため OFF
  * ファームハンドは水やり/収穫のカバー率を上げ +$3k 程度の効果 → ON
  * 肥料は水やりを欠かさなければメロンには無意味 → OFF (トマト・イチゴ用)

提出形式: このファイルの `agent(obs)` を main.py としてそのまま提出できる。
"""
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS, LAND_PRICES, ANIMALS

CROP = "MELON"
SEED_COST = CROPS[CROP]["seed"]
FIRST_YIELD_DAY = CROPS[CROP]["first_yield_day"]
MAX_YIELD_DAY = CROPS[CROP]["max_yield_day"]
WINDOW_START = (MAX_YIELD_DAY + 1) // 2

WHEAT = "WHEAT"
WHEAT_SEED_COST = CROPS[WHEAT]["seed"]
WHEAT_MAX_YIELD_DAY = CROPS[WHEAT]["max_yield_day"]
WHEAT_SELL_CHUNK = 10
# メロンのタイル上限: メロン市場の消化量は両者合計 ~150個で頭打ちのため、
# 25タイル全メロンは価格を暴落させる (旧エージェントの自然平衡 ~12タイルを固定化)。
MAX_MELON_TILES = 12

SELL_THRESHOLD = 200
SELL_CHUNK = 10
SHED_FORCE_SELL = 90
SHED_KEEP_AFTER_FORCE_SELL = 40
CASHOUT_DAY = 28
HIRE_MAX = 2
HIRE_MIN_MONEY = 300
# 仮説 E012b: ハンドに小麦植えを許可する。E012 (ハンド増強) は「ハンドが植えられず
# やることがない」ため効果がなかった。小麦は種が安い (10) ので PLANT ブロッキングの
# リスクが低く、ハンドの余剰ターンを小麦畑に回せる。
USE_HAND_WHEAT_PLANT = False
LAND_BUFFER = 500
FERTILIZER_MAX_STOCK = 3
FERTILIZER_MIN_MONEY = 400

USE_HANDS = True
# 注意: メロン専業では土地購入は逆効果 (E001)。ただし多品種+動物なら再評価の価値がある (E015)。
# E015a 検証済み: 無制限の土地購入 (SE含む) は労働力が追いつかず -$13k。NE+SWのみ・小麦タイル上限が条件。
USE_LAND = False
MAX_LAND_BUYS = 2
MAX_WHEAT_TILES = 25
# メモ: 肥料は 1回限り作物 (メロン) には水やりを欠かさなければ無意味
# (水やりだけで max_yield=6 に到達する)。オンニング作物 (トマト・イチゴ) の
# 2倍収穫に有効なので、多品種化したら ON にすると良い。
USE_FERTILIZER = False
# 仮説 E007: タウンデマンド消費ターン (step%4==0 ショップ / step%24==0 タウンセンター)
# には売却しない。消費はターン終了時に発生し価格は翌ターンに反映されるため、
# 消費ターンに売るのは翌ターンより不利 (boatlee v16 のフロントラン機構の簡易版)。
# 検証済み (2026-08-28, E007): メロンにはショップ需要がなく (タウンセンター1個/日のみ)、
# 価格ブースト +1〜4 は売り逃しリスクで相殺され**無効**。多品種化 (ショップ需要商品) で再評価。
DEMAND_AWARE_SELL = False
# 仮説 E008: 小麦への多品種化。メロン価格が閾値を割った後 (メロン市場頭打ち) や
# 納屋満杯・タイル上限でメロンが植えられないときは、空きタイルを小麦
# (seed $10, 4日で4個, age>=4 収穫) に転換し、価格安定の小麦を常時分割売却する。
# 小麦は glut を吸収 (log 曲線) するため閾値なしで売れる。
# 検証済み (2026-08-28, E008): vs base +$710 / vs random +$471 / mirror +$3k。
# メロンは MAX_MELON_TILES に制限 (市場の消化量を超えると価格暴落するため)。
USE_WHEAT = True
# 仮説 E009: 動物 (COW/SHEEP) + 肥料売り。Kaito v27 の最大の構造差。
# 動物1頭/日で肥料1個が無料生成 (COLLECT_FERTILIZER) → 4頭 = 1日4個 ≈ $400/日。
# フィードは小麦4個/日 (自前生産 + BUY_PRODUCT で補充)。タイル: 牧場4 + メロン12 + 小麦9。
USE_ANIMALS = True
# E009 検証済み: 2COW+1SHEEP (3頭) が最適 (4頭は初期資金 $3,000 で種+フィードと競合して資金ショート)。
# ローカル実測 (vs random 10試合): 2COW $35.6k / 2COW+1SHEEP $39.4k / 2COW+2SHEEP $25.3k / 動物なし $26.1k
# 動物1頭/日で肥料1個が無料生成。肥料4個/日 + ミルク + 羊毛の三重収入。
ANIMAL_PLAN = {"COW": 2, "SHEEP": 1}
ANIMAL_PRODUCT = {a: ANIMALS[a]["product"] for a in ANIMAL_PLAN}
FEED_RESERVE = sum(ANIMAL_PLAN.values()) * 2
MAX_ANIMAL_BUYS_PER_DAY = 2
# 仮説 E011: 小麦の保有戦略。小麦価格は動物メタでは 25→43 に上昇 (#1 Crop Dusta の
# リプレイ実測)。常時売却は安値で手放すため、閾値以上 or 納屋圧迫 or 終盤にだけ売る。
USE_WHEAT_HOLD = False
WHEAT_HOLD_THRESHOLD = 35
WHEAT_HOLD_SHED_PRESSURE = 85


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


def _is_demand_turn(step):
    """タウンデマンド消費ターンかどうか (消費はターン終了時に発生)。

    ショップは step%4==0、タウンセンターは step%24==0 に消費する。
    消費はターン終了時に起きるため、消費ターンに売るのは翌ターンより不利。
    """
    return step % 4 == 0 or step % 24 == 0


def _is_crop(tile, crop):
    return isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == crop


def _is_melon(tile):
    return _is_crop(tile, CROP)


def _inventory_of(private, farm, pos):
    idx = 0
    if tuple(pos) != tuple(farm["farmer"]):
        for i, h in enumerate(farm["hands"]):
            if tuple(h) == tuple(pos):
                idx = i + 1
                break
    invs = private.get("inventories", [])
    return invs[idx] if idx < len(invs) else {}


def _is_shed_adjacent(pos):
    x, y = pos
    return x in (4, 5) and y in (4, 5)


def _any_plant_in_window(tiles, day):
    for row in tiles:
        for tile in row:
            if _is_melon(tile):
                age = day - tile["planted_day"]
                if WINDOW_START <= age <= MAX_YIELD_DAY - 1 and tile.get("fertilized_until_day", -1) < day:
                    return True
    return False


def _count_pastures(tiles):
    return sum(1 for row in tiles for t in row if isinstance(t, dict) and t.get("kind") == "PASTURE")


def _count_placed_animals(tiles):
    cnt = {}
    for row in tiles:
        for t in row:
            if isinstance(t, dict) and "animal" in t:
                cnt[t["animal"]] = cnt.get(t["animal"], 0) + 1
    return cnt


def _any_animal_unfed(tiles):
    return any(
        isinstance(t, dict) and "animal" in t and not t.get("fed_today", False)
        for row in tiles for t in row
    )


def _has_empty_pasture(tiles):
    return any(
        isinstance(t, dict) and t.get("kind") == "PASTURE" and "animal" not in t
        for row in tiles for t in row
    )


def _nearest_shed_tile(x0, y0):
    return (4 if x0 < 5 else 5, 4 if y0 < 5 else 5)


def _scan_tasks(tiles, board_size, x0, y0, day, want_melon_plant, want_wheat_plant, melon_seeds, wheat_seeds, plant_first=False, inv=None, shed=None):
    """Return nearest tasks sorted by priority.

    役割分担: plant_first=True (ファーマー) は植え付け・セットアップ優先、
    False (ハンド) は動物世話・水やり優先。
    ファーマー: メロン収穫 > 動物収穫 > 配置 > 牧場建設 > 雑草 > メロン植え > メロン水やり >
               小麦収穫 > 小麦植え > 小麦水やり > 動物拾い > 動物世話 (フォールバック)
    ハンド:     動物フィード > 動物収穫 > 肥料回収 > ケア > 作物収穫 > 小麦拾い >
               メロン水やり > 小麦水やり > 雑草
    """
    tasks = []
    any_unfed = False
    carrying_animal = USE_ANIMALS and inv is not None and any(inv.get(a, 0) > 0 for a in ANIMAL_PLAN)
    need_pastures = USE_ANIMALS and _count_pastures(tiles) < sum(ANIMAL_PLAN.values())
    for y in range(board_size):
        for x in range(board_size):
            tile = tiles[y][x]
            dist = abs(x - x0) + abs(y - y0)
            if _is_melon(tile):
                age = day - tile["planted_day"]
                if tile["yield_units"] > 0 and age >= FIRST_YIELD_DAY and (
                    tile["yield_units"] >= 5 or age >= MAX_YIELD_DAY
                ):
                    tasks.append((0 if plant_first else 4, dist, x, y))
                elif not tile["watered_today"]:
                    tasks.append((6 if plant_first else 6, dist, x, y))
            elif USE_WHEAT and _is_crop(tile, WHEAT):
                age = day - tile["planted_day"]
                if tile["yield_units"] > 0 and age >= WHEAT_MAX_YIELD_DAY:
                    tasks.append((7 if plant_first else 4, dist, x, y))
                elif not tile["watered_today"]:
                    tasks.append((9 if plant_first else 7, dist, x, y))
            elif USE_ANIMALS and isinstance(tile, dict) and "animal" in tile:
                if not tile.get("fed_today", False):
                    any_unfed = True
                    # 小麦を持っていないときはフィードタスクを発行しない
                    # (納屋への PICKUP タスクが誘導する。距離0のフィードタスクが
                    # PASS になるバグの回避)
                    if inv is None or inv.get(WHEAT, 0) > 0:
                        tasks.append((11 if plant_first else 0, dist, x, y))
                elif tile.get("yield_units", 0) > 0:
                    tasks.append((1, dist, x, y))
                elif tile.get("fertilizer_available"):
                    tasks.append((12 if plant_first else 2, dist, x, y))
                elif not tile.get("cared_today", False):
                    tasks.append((13 if plant_first else 3, dist, x, y))
            elif isinstance(tile, dict) and tile.get("kind") == "WEED":
                tasks.append((4 if plant_first else 8, dist, x, y))
            elif tile is None:
                if USE_ANIMALS and need_pastures and plant_first:
                    tasks.append((3, dist, x, y))
                elif want_melon_plant and melon_seeds > 0:
                    tasks.append((5, dist, x, y))
                elif want_wheat_plant and wheat_seeds > 0:
                    tasks.append((8, dist, x, y))
    if USE_ANIMALS and inv is not None and shed is not None:
        sx, sy = _nearest_shed_tile(x0, y0)
        shed_dist = abs(x0 - sx) + abs(y0 - sy)
        # 空き牧場への配置 (手持ちに動物があるとき)
        if carrying_animal:
            for y in range(board_size):
                for x in range(board_size):
                    t = tiles[y][x]
                    if isinstance(t, dict) and t.get("kind") == "PASTURE" and "animal" not in t:
                        tasks.append((2 if plant_first else 8, abs(x - x0) + abs(y - y0), x, y))
        elif _has_empty_pasture(tiles) and any(shed.get(a, 0) > 0 for a in ANIMAL_PLAN):
            # 配置用動物の受け取り (納屋へ)
            tasks.append((10 if plant_first else 9, shed_dist, sx, sy))
        # フィード用小麦の受け取り (納屋へ)
        if inv.get(WHEAT, 0) == 0 and shed.get(WHEAT, 0) > 0 and any_unfed:
            tasks.append((14 if plant_first else 5, shed_dist, sx, sy))
    tasks.sort(key=lambda t: (t[0], t[1]))
    return tasks


def _unit_action(obs, farm, private, pos, day, role, can_plant):
    tiles = farm["tiles"]
    board_size = len(tiles)
    x, y = pos
    tile = tiles[y][x]
    seeds = private.get("seeds", {})
    inv = _inventory_of(private, farm, pos)
    shed = private.get("shed", {})
    melon_price = obs["market"]["prices"][CROP]

    should_plant_melon = can_plant and melon_price >= SELL_THRESHOLD and shed.get(CROP, 0) < 80
    if USE_WHEAT:
        # メロンのタイル上限 (市場の消化量を超えて植えると価格暴落するため)
        melon_tiles = sum(1 for row in tiles for t in row if _is_crop(t, CROP))
        should_plant_melon = should_plant_melon and melon_tiles < MAX_MELON_TILES
    # 小麦は「メロンが植えられないとき」の代替: 価格 < 閾値 だけでなく、
    # 納屋満杯・タイル上限でも空きタイルを小麦で埋める (E008 で検証)
    # E012b: ハンドにも小麦植えを許可 (小麦の余剰ターンを活用)
    # E015: 小麦タイル上限 (労働力が追いつかない過剰拡散を防ぐ)
    should_plant_wheat = USE_WHEAT and (
        can_plant or (USE_HAND_WHEAT_PLANT and role == "hand")
    ) and not should_plant_melon
    if should_plant_wheat:
        wheat_tiles = sum(1 for row in tiles for t in row if _is_crop(t, WHEAT))
        should_plant_wheat = should_plant_wheat and wheat_tiles < MAX_WHEAT_TILES

    # --- 納屋での受け取り (フィード用小麦 / 配置用動物) ---
    if USE_ANIMALS and _is_shed_adjacent(pos):
        if inv.get(WHEAT, 0) == 0 and shed.get(WHEAT, 0) > 0 and _any_animal_unfed(tiles):
            return ["PICKUP", WHEAT, 1]
        if (
            not any(inv.get(a, 0) > 0 for a in ANIMAL_PLAN)
            and _has_empty_pasture(tiles)
            and any(shed.get(a, 0) > 0 for a in ANIMAL_PLAN)
        ):
            for a in ANIMAL_PLAN:
                if shed.get(a, 0) > 0:
                    return ["PICKUP", a, 1]

    if _is_melon(tile):
        age = day - tile["planted_day"]
        if tile["yield_units"] > 0 and age >= FIRST_YIELD_DAY and (
            tile["yield_units"] >= 5 or age >= MAX_YIELD_DAY
        ):
            return ["HARVEST"]
        if not tile["watered_today"]:
            return ["WATER"]
        if (
            role == "farmer"
            and USE_FERTILIZER
            and WINDOW_START <= age <= MAX_YIELD_DAY - 1
            and tile.get("fertilized_until_day", -1) < day
            and inv.get("FERTILIZER", 0) > 0
            and melon_price >= SELL_THRESHOLD
        ):
            return ["FERTILIZE"]
    elif USE_WHEAT and _is_crop(tile, WHEAT):
        age = day - tile["planted_day"]
        if tile["yield_units"] > 0 and age >= WHEAT_MAX_YIELD_DAY:
            return ["HARVEST"]
        if not tile["watered_today"]:
            return ["WATER"]
    elif USE_ANIMALS and isinstance(tile, dict) and "animal" in tile:
        if not tile.get("fed_today", False):
            if inv.get(WHEAT, 0) > 0:
                return ["FEED"]
        elif tile.get("yield_units", 0) > 0:
            return ["HARVEST"]
        elif tile.get("fertilizer_available"):
            return ["COLLECT_FERTILIZER"]
        elif not tile.get("cared_today", False):
            return ["CARE"]
    elif USE_ANIMALS and isinstance(tile, dict) and tile.get("kind") == "PASTURE" and "animal" not in tile:
        for a in ANIMAL_PLAN:
            if inv.get(a, 0) > 0:
                return ["PLACE", a, 1]
    elif isinstance(tile, dict) and tile.get("kind") == "WEED":
        return ["DIG"]
    elif tile is None:
        if USE_ANIMALS and _count_pastures(tiles) < sum(ANIMAL_PLAN.values()):
            return ["BUILD_PASTURE"]
        if should_plant_melon and seeds.get(CROP, 0) > 0:
            return ["PLANT", CROP]
        if should_plant_wheat and seeds.get(WHEAT, 0) > 0:
            return ["PLANT", WHEAT]

    # 肥料の受け取り: 納屋隣接タイルに居て、手持ちが無く、納屋に在庫があり、
    # ボーナス期間の植物が存在するとき。朝は shed タイル (4,4) で毎日発生する。
    if (
        role == "farmer"
        and USE_FERTILIZER
        and _is_shed_adjacent(pos)
        and inv.get("FERTILIZER", 0) == 0
        and shed.get("FERTILIZER", 0) > 0
        and melon_price >= SELL_THRESHOLD
        and _any_plant_in_window(tiles, day)
    ):
        return ["PICKUP", "FERTILIZER", 1]

    tasks = _scan_tasks(
        tiles, board_size, x, y, day,
        should_plant_melon, should_plant_wheat,
        seeds.get(CROP, 0), seeds.get(WHEAT, 0),
        plant_first=(role == "farmer" and USE_WHEAT),
        inv=inv, shed=shed,
    )
    if tasks:
        step = _step_toward(x, y, tasks[0][2], tasks[0][3])
        if step:
            return [step]
    return ["PASS"]


def agent(obs):
    try:
        farms = obs.get("farms", [])
        player = obs.get("player", 0)
        private = obs.get("private", {}) or {}
        if not farms or player >= len(farms):
            return {"farmer": ["PASS"], "hands": [], "market": []}

        farm = farms[player]
        day = obs.get("day", 0)
        hour = obs.get("hour", 0)
        seeds = private.get("seeds", {})
        shed = private.get("shed", {})
        prices = (obs.get("market", {}) or {}).get("prices", {})
        melon_price = prices.get(CROP, 0)
        melons_in_shed = shed.get(CROP, 0)

        market = []

        # --- 売却: 閾値以上なら分割売却、納屋が溢れそうなら強制売却、終盤は換金 ---
        n_sell = 0
        if melons_in_shed > 0:
            price_ok = melon_price >= SELL_THRESHOLD
            if DEMAND_AWARE_SELL and _is_demand_turn(day * 24 + hour):
                price_ok = False
            if price_ok or day >= CASHOUT_DAY:
                n_sell = min(SELL_CHUNK, melons_in_shed)
            elif melons_in_shed >= SHED_FORCE_SELL:
                n_sell = min(SELL_CHUNK, max(1, melons_in_shed - SHED_KEEP_AFTER_FORCE_SELL))
        if n_sell > 0:
            market.append(["SELL", CROP, n_sell])

        # --- 小麦の売却 ---
        # E011: 常時売却ではなく、閾値以上 (高値) or 納屋圧迫 or 終盤のみ売る。
        # 小麦価格は動物メタで上昇するため、保有して高値で売るのが有利のはず。
        wheat_in_shed = shed.get(WHEAT, 0)
        feed_reserve = FEED_RESERVE if USE_ANIMALS else 0
        sellable_wheat = max(0, wheat_in_shed - feed_reserve)
        if USE_WHEAT and sellable_wheat > 0:
            wheat_price = prices.get(WHEAT, 0)
            if (
                not USE_WHEAT_HOLD
                or wheat_price >= WHEAT_HOLD_THRESHOLD
                or sum(shed.values()) >= WHEAT_HOLD_SHED_PRESSURE
                or day >= CASHOUT_DAY
            ):
                market.append(["SELL", WHEAT, min(WHEAT_SELL_CHUNK, sellable_wheat)])

        # --- 動物製品と肥料の売却 (常時・分割。供給量が小さく価格への影響は僅少) ---
        if USE_ANIMALS:
            for item in ("MILK", "WOOL", "FERTILIZER"):
                n = shed.get(item, 0)
                if n > 0:
                    market.append(["SELL", item, min(SELL_CHUNK, n)])

        # --- 動物の購入 (セットアップ期、1日最大2頭。手持ちも所有数に含める) ---
        # 種より先に処理する: 初期資金 ($3,000) は動物 $1,800 + メロン種 $960 で
        # ほぼ使い切るため、動物が後回しだと購入できない (E009 の失敗事例)
        if USE_ANIMALS:
            placed = _count_placed_animals(farm["tiles"])
            inv_animals = {a: 0 for a in ANIMAL_PLAN}
            for inv in private.get("inventories", []):
                for a in ANIMAL_PLAN:
                    inv_animals[a] += inv.get(a, 0)
            buys = 0
            for a, target in ANIMAL_PLAN.items():
                owned = shed.get(a, 0) + placed.get(a, 0) + inv_animals.get(a, 0)
                while owned < target and buys < MAX_ANIMAL_BUYS_PER_DAY and farm["money"] >= ANIMALS[a]["cost"]:
                    market.append(["BUY_ANIMAL", a, 1])
                    buys += 1
                    owned += 1

        # --- フィード用小麦の購入 (自前生産が追いつくまでのつなぎ、2日分の予備) ---
        if USE_ANIMALS and farm["money"] >= 150:
            total_wheat = shed.get(WHEAT, 0) + sum(inv.get(WHEAT, 0) for inv in private.get("inventories", []))
            if total_wheat < FEED_RESERVE:
                market.append(["BUY_PRODUCT", WHEAT, min(2, FEED_RESERVE - total_wheat)])

        # --- 種の補充 (植え付けが有効な場合のみ) ---
        should_plant = melon_price >= SELL_THRESHOLD and melons_in_shed < 80
        if USE_WHEAT:
            melon_tiles = sum(1 for row in farm["tiles"] for t in row if _is_crop(t, CROP))
            should_plant = should_plant and melon_tiles < MAX_MELON_TILES
        if should_plant and seeds.get(CROP, 0) < 2 and farm["money"] >= SEED_COST:
            # まとめ買い (12個) が植え付け速度に重要: 少ないと植え付けが遅れ、
            # メロン高値期を逃す (E009 の失敗事例)
            market.append(["BUY_SEED", CROP, min(12, int(farm["money"]) // SEED_COST)])

        # --- 小麦の種の補充 (メロン植えが止まっているとき) ---
        if (
            USE_WHEAT
            and not should_plant
            and seeds.get(WHEAT, 0) < 2
            and farm["money"] >= WHEAT_SEED_COST
        ):
            market.append(["BUY_SEED", WHEAT, min(15, int(farm["money"]) // WHEAT_SEED_COST)])

        # --- 土地購入: メロン価格が良いときだけ拡張 (E015: NE+SW のみ、SE は不買) ---
        if USE_LAND and melon_price >= SELL_THRESHOLD:
            n_extra = len(farm.get("unlocked_quadrants", ["NW"])) - 1
            if n_extra < MAX_LAND_BUYS and farm["money"] >= LAND_PRICES[n_extra] + LAND_BUFFER:
                market.append(["BUY_LAND"])

        # --- ファームハンド雇用: 朝の1回で、余裕があるときだけ ---
        # 小麦モード (USE_WHEAT) では常時雇用 (コストは1日2ドルと僅少。
        # 種購入で資金が枯渇しても水やり維持が最優先)。それ以外は従来ゲート。
        hire_min_money = 10 if USE_WHEAT else HIRE_MIN_MONEY
        if (
            USE_HANDS
            and hour == 0
            and farm.get("hires_today", 0) < HIRE_MAX
            and (melon_price >= SELL_THRESHOLD or USE_WHEAT)
            and farm["money"] >= hire_min_money
        ):
            market.append(["HIRE"])

        # --- 肥料の仕入れ ---
        if (
            USE_FERTILIZER
            and melon_price >= SELL_THRESHOLD
            and shed.get("FERTILIZER", 0) < FERTILIZER_MAX_STOCK
            and farm["money"] >= FERTILIZER_MIN_MONEY
        ):
            market.append(["BUY_PRODUCT", "FERTILIZER", 1])

        # --- 各ユニットの行動 ---
        hands = []
        if USE_HANDS:
            for h in farm.get("hands", []):
                hands.append(_unit_action(obs, farm, private, h, day, "hand", False))
        else:
            hands = [["PASS"]] * len(farm.get("hands", []))

        farmer = _unit_action(obs, farm, private, farm["farmer"], day, "farmer", True)
        return {"farmer": farmer, "hands": hands, "market": market}
    except Exception:
        return {"farmer": ["PASS"], "hands": [], "market": []}


melon_maxxer_improved = agent