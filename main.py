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
# E017: イチゴ (継続作物、ショップ需要あり、上位は 24-43種・181-442個売却)。
# 小麦よりタイル価値が高く、施肥で 2倍収穫 (Michael Timbs は肥料を 248-324個購入)。
STRAWBERRY = "STRAWBERRY"
STRAWBERRY_SEED_COST = CROPS[STRAWBERRY]["seed"]
STRAWBERRY_FIRST_YIELD_DAY = CROPS[STRAWBERRY]["first_yield_day"]
STRAWBERRY_MAX_YIELD_DAY = CROPS[STRAWBERRY]["max_yield_day"]
MAX_STRAWBERRY_TILES = 12

SELL_THRESHOLD = 200
SELL_CHUNK = 10
SHED_FORCE_SELL = 90
SHED_KEEP_AFTER_FORCE_SELL = 40
CASHOUT_DAY = 28
HIRE_MAX = 2
HIRE_MIN_MONEY = 300
# E017: スケールアップ (top5 メタ優先)。上位は土地2区画 + 動物10-19頭 + ~10ハンド + イチゴ。
# このフラグを ON にすると上記の構成へ拡張する (E015/E013 の単発失敗からセットで再挑戦)。
# 個別フラグ: USE_LAND / USE_HAND_WHEAT_PLANT / HIRE_MAX / ANIMAL_PLAN / MAX_WHEAT_TILES / USE_STRAWBERRY
USE_SCALE = False
# 仮説 E012b: ハンドに小麦/イチゴ植えを許可 (スケール時は必須。E012 の「タスク不足」は
# 土地+動物増強で解消されるはず)
USE_HAND_WHEAT_PLANT = False
# E017: イチゴ導入 (継続作物・ショップ需要・施肥で2倍)
USE_STRAWBERRY = False
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
# E009 検証済み: 2COW+1SHEEP (3頭) が 25タイル・2ハンド の最適。
# E017 (top5 メタ優先): 上位は 10-19頭 + 土地2区画 + ~10ハンド で運用。スケール時はこの plan を拡大。
ANIMAL_PLAN = {"COW": 2, "SHEEP": 1}
ANIMAL_PRODUCT = {a: ANIMALS[a]["product"] for a in ANIMAL_PLAN}
FEED_RESERVE = sum(ANIMAL_PLAN.values()) * 2
MAX_ANIMAL_BUYS_PER_DAY = 2
# 仮説 E013: 収入が回ったら GOOSE を買い増す。卵市場は glut 吸収力が高い
# (T=332, log 曲線) ため、ミルク/羊毛が暴落する COW/SHEEP の追加より安全。
# GOOSE 1頭 = 卵1個/日 ($50) + 肥料1個/日 ($90) で COW 並みの収入。
USE_GOOSE = False
GOOSE_TARGET = 3
GOOSE_BUY_DAY = 5
GOOSE_BUY_MONEY = 600
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
            if _is_crop(tile, STRAWBERRY):
                age = day - tile["planted_day"]
                if STRAWBERRY_FIRST_YIELD_DAY - 2 <= age <= STRAWBERRY_MAX_YIELD_DAY and tile.get("fertilized_until_day", -1) < day:
                    return True
    return False


def _count_pastures(tiles):
    return sum(1 for row in tiles for t in row if isinstance(t, dict) and t.get("kind") == "PASTURE")


def _count_structs(tiles, kind):
    return sum(1 for row in tiles for t in row if isinstance(t, dict) and t.get("kind") == kind)


def _count_placed_animals(tiles):
    cnt = {}
    for row in tiles:
        for t in row:
            if isinstance(t, dict) and "animal" in t:
                cnt[t["animal"]] = cnt.get(t["animal"], 0) + 1
    return cnt


def _total_animals(shed, invs, placed):
    """納屋 + 手持ち + 配置済みの動物総数。"""
    n = sum(shed.get(a, 0) for a in _animals_all())
    n += sum(inv.get(a, 0) for a in _animals_all() for inv in invs)
    n += sum(placed.values())
    return n


def _animals_all():
    if USE_GOOSE:
        return set(ANIMAL_PLAN) | {"GOOSE"}
    return set(ANIMAL_PLAN)


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


def _has_empty_struct(tiles, kind):
    return any(
        isinstance(t, dict) and t.get("kind") == kind and "animal" not in t
        for row in tiles for t in row
    )


def _nearest_shed_tile(x0, y0):
    return (4 if x0 < 5 else 5, 4 if y0 < 5 else 5)


def _scan_tasks(tiles, board_size, x0, y0, day, want_melon_plant, want_wheat_plant, want_strawberry_plant, melon_seeds, wheat_seeds, strawberry_seeds, plant_first=False, inv=None, shed=None):
    """Return nearest tasks sorted by priority.

    役割分担: plant_first=True (ファーマー) は植え付け・セットアップ優先、
    False (ハンド) は動物世話・水やり優先。
    ファーマー: メロン収穫 > 動物収穫 > 配置 > 牧場建設 > 雑草 > メロン植え > メロン水やり >
               イチゴ/小麦収穫 > イチゴ植え > イチゴ水やり > 小麦植え > 小麦水やり >
               動物拾い > 動物世話 (フォールバック)
    ハンド:     動物フィード > 動物収穫 > 肥料回収 > ケア > 作物収穫 > 小麦拾い >
               メロン水やり > イチゴ水やり > 小麦水やり > 雑草 > イチゴ/小麦植え
    """
    tasks = []
    any_unfed = False
    carrying_animal = USE_ANIMALS and inv is not None and any(inv.get(a, 0) > 0 for a in _animals_all())
    # E017: 牧場は「購入済み動物数」駆動で建設 (plan 全頭分を先に作らない —
    # 植え付け前の牧場乱立が農場を壊す)
    if USE_ANIMALS and shed is not None:
        placed = _count_placed_animals(tiles)
        need_pastures = _count_pastures(tiles) < _total_animals(shed, [inv] if inv is not None else [], placed)
    else:
        need_pastures = False
    need_coops = USE_GOOSE and _count_structs(tiles, "COOP") < GOOSE_TARGET
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
                    tasks.append((11 if plant_first else 8, dist, x, y))
            elif USE_STRAWBERRY and _is_crop(tile, STRAWBERRY):
                age = day - tile["planted_day"]
                if tile["yield_units"] > 0 and age >= STRAWBERRY_FIRST_YIELD_DAY:
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
                        tasks.append((13 if plant_first else 0, dist, x, y))
                elif tile.get("yield_units", 0) > 0:
                    tasks.append((1, dist, x, y))
                elif tile.get("fertilizer_available"):
                    tasks.append((14 if plant_first else 2, dist, x, y))
                elif not tile.get("cared_today", False):
                    tasks.append((15 if plant_first else 3, dist, x, y))
            elif isinstance(tile, dict) and tile.get("kind") == "WEED":
                tasks.append((4 if plant_first else 9, dist, x, y))
            elif tile is None:
                if USE_ANIMALS and need_pastures and plant_first:
                    tasks.append((3, dist, x, y))
                elif USE_GOOSE and need_coops and plant_first:
                    tasks.append((3, dist, x, y))
                elif want_melon_plant and melon_seeds > 0:
                    tasks.append((5, dist, x, y))
                elif USE_STRAWBERRY and want_strawberry_plant and strawberry_seeds > 0:
                    tasks.append((8 if plant_first else 10, dist, x, y))
                elif want_wheat_plant and wheat_seeds > 0:
                    tasks.append((10 if plant_first else 11, dist, x, y))
    if USE_ANIMALS and inv is not None and shed is not None:
        sx, sy = _nearest_shed_tile(x0, y0)
        shed_dist = abs(x0 - sx) + abs(y0 - sy)
        # 空き構造への配置 (手持ちに動物があるとき)
        if carrying_animal:
            for y in range(board_size):
                for x in range(board_size):
                    t = tiles[y][x]
                    if isinstance(t, dict) and "animal" not in t and t.get("kind") in ("PASTURE", "COOP"):
                        tasks.append((2 if plant_first else 8, abs(x - x0) + abs(y - y0), x, y))
        elif (
            (_has_empty_pasture(tiles) or _has_empty_struct(tiles, "COOP"))
            and any(shed.get(a, 0) > 0 for a in _animals_all())
        ):
            # 配置用動物の受け取り (納屋へ)
            tasks.append((12 if plant_first else 9, shed_dist, sx, sy))
        # フィード用小麦の受け取り (納屋へ)
        if inv.get(WHEAT, 0) == 0 and shed.get(WHEAT, 0) > 0 and any_unfed:
            tasks.append((16 if plant_first else 5, shed_dist, sx, sy))
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
    # E017: イチゴはメロンの次に優先 (小麦よりタイル価値が高い)
    should_plant_strawberry = USE_STRAWBERRY and (
        can_plant or (USE_HAND_WHEAT_PLANT and role == "hand")
    ) and not should_plant_melon
    if should_plant_strawberry:
        strawberry_tiles = sum(1 for row in tiles for t in row if _is_crop(t, STRAWBERRY))
        should_plant_strawberry = should_plant_strawberry and strawberry_tiles < MAX_STRAWBERRY_TILES

    # --- 納屋での受け取り (フィード用小麦 / 配置用動物) ---
    if USE_ANIMALS and _is_shed_adjacent(pos):
        # 小麦は1個ずつピックアップ (一括6個は回帰の原因だった: 納屋の小麦を
        # 引き抜くと売却・予備のバランスが崩れる — E017 の失敗事例)
        if inv.get(WHEAT, 0) == 0 and shed.get(WHEAT, 0) > 0 and _any_animal_unfed(tiles):
            return ["PICKUP", WHEAT, 1]
        if (
            not any(inv.get(a, 0) > 0 for a in _animals_all())
            and (_has_empty_pasture(tiles) or _has_empty_struct(tiles, "COOP"))
            and any(shed.get(a, 0) > 0 for a in _animals_all())
        ):
            for a in sorted(_animals_all()):
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
    elif USE_STRAWBERRY and _is_crop(tile, STRAWBERRY):
        age = day - tile["planted_day"]
        if tile["yield_units"] > 0 and age >= STRAWBERRY_FIRST_YIELD_DAY:
            return ["HARVEST"]
        if not tile["watered_today"]:
            return ["WATER"]
        # E017: イチゴ (継続作物) への施肥 — 生産日に 2倍収穫
        if (
            USE_FERTILIZER
            and STRAWBERRY_FIRST_YIELD_DAY - 2 <= age <= STRAWBERRY_MAX_YIELD_DAY
            and tile.get("fertilized_until_day", -1) < day
            and inv.get("FERTILIZER", 0) > 0
        ):
            return ["FERTILIZE"]
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
    elif USE_GOOSE and isinstance(tile, dict) and tile.get("kind") == "COOP" and "animal" not in tile:
        if inv.get("GOOSE", 0) > 0:
            return ["PLACE", "GOOSE", 1]
    elif isinstance(tile, dict) and tile.get("kind") == "WEED":
        return ["DIG"]
    elif tile is None:
        if USE_ANIMALS and _count_pastures(tiles) < _total_animals(shed, [inv], _count_placed_animals(tiles)):
            return ["BUILD_PASTURE"]
        if USE_GOOSE and _count_structs(tiles, "COOP") < GOOSE_TARGET:
            return ["BUILD_COOP"]
        if should_plant_melon and seeds.get(CROP, 0) > 0:
            return ["PLANT", CROP]
        if should_plant_strawberry and seeds.get(STRAWBERRY, 0) > 0:
            return ["PLANT", STRAWBERRY]
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
        and (melon_price >= SELL_THRESHOLD or USE_STRAWBERRY)
        and _any_plant_in_window(tiles, day)
    ):
        return ["PICKUP", "FERTILIZER", 1]

    tasks = _scan_tasks(
        tiles, board_size, x, y, day,
        should_plant_melon, should_plant_wheat, should_plant_strawberry,
        seeds.get(CROP, 0), seeds.get(WHEAT, 0), seeds.get(STRAWBERRY, 0),
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

        # --- 売却 (最優先: 収入の源泉。10件制限で切り捨てられると資金が枯渇する) ---
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
        if USE_ANIMALS:
            n_animals = _total_animals(shed, private.get("inventories", []), _count_placed_animals(farm["tiles"]))
            feed_reserve = max(6, n_animals * 2)
        if USE_GOOSE:
            feed_reserve += GOOSE_TARGET * 2
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
            for item in ("MILK", "WOOL", "FERTILIZER", "EGG"):
                n = shed.get(item, 0)
                if n > 0:
                    market.append(["SELL", item, min(SELL_CHUNK, n)])

        # --- イチゴの売却 (常時・分割) ---
        strawberry_in_shed = shed.get(STRAWBERRY, 0)
        if USE_STRAWBERRY and strawberry_in_shed > 0:
            market.append(["SELL", STRAWBERRY, min(SELL_CHUNK, strawberry_in_shed)])

        # --- 動物の購入 (セットアップ期、1日最大2頭。手持ちも所有数に含める) ---
        # 種より先に処理する: 初期資金 ($3,000) は動物 $1,800 + メロン種 $960 で
        # ほぼ使い切るため、動物が後回しだと購入できない (E009 の失敗事例)
        # E017: 買い増しには資金バッファ (800) を要求 — 農場運用 (種・ハンド・フィード) を
        # 枯渇させない (E017 初期実装の失敗: 動物に資金が吸われて植え付けが崩壊)
        if USE_ANIMALS:
            placed = _count_placed_animals(farm["tiles"])
            inv_animals = {a: 0 for a in ANIMAL_PLAN}
            for inv in private.get("inventories", []):
                for a in ANIMAL_PLAN:
                    inv_animals[a] += inv.get(a, 0)
            buys = 0
            animal_buffer = 800 if len(ANIMAL_PLAN) > 3 else 0
            for a, target in ANIMAL_PLAN.items():
                owned = shed.get(a, 0) + placed.get(a, 0) + inv_animals.get(a, 0)
                while owned < target and buys < MAX_ANIMAL_BUYS_PER_DAY and farm["money"] >= ANIMALS[a]["cost"] + animal_buffer:
                    market.append(["BUY_ANIMAL", a, 1])
                    buys += 1
                    owned += 1
            # E013: GOOSE の買い増し (収入が回った day>=5 以降、資金に余裕があるとき)
            if USE_GOOSE and day >= GOOSE_BUY_DAY:
                goose_owned = shed.get("GOOSE", 0) + placed.get("GOOSE", 0) + sum(
                    inv.get("GOOSE", 0) for inv in private.get("inventories", [])
                )
                if goose_owned < GOOSE_TARGET and farm["money"] >= ANIMALS["GOOSE"]["cost"] + GOOSE_BUY_MONEY:
                    market.append(["BUY_ANIMAL", "GOOSE", 1])

        # --- 雇用 (売却の次: 農場運用の要。残りスロット数に応じて調整) ---
        # 10件制限を考慮: 売却数+2 (土地等) を残して雇用数を決定
        hire_min_money = 10 if USE_WHEAT else HIRE_MIN_MONEY
        if USE_HANDS and hour == 0 and (melon_price >= SELL_THRESHOLD or USE_WHEAT) and farm["money"] >= hire_min_money:
            n_hired = farm.get("hires_today", 0)
            sell_count = len(market)
            hire_cap = min(HIRE_MAX, max(2, 10 - sell_count - 2))
            a, b = 1, 1
            for _ in range(n_hired):
                a, b = b, a + b
            for _ in range(hire_cap):
                if n_hired >= hire_cap or farm["money"] < a:
                    break
                market.append(["HIRE"])
                n_hired += 1
                a, b = b, a + b
        if USE_LAND and melon_price >= SELL_THRESHOLD:
            n_extra = len(farm.get("unlocked_quadrants", ["NW"])) - 1
            # E017: 土地は上位と同じタイミング (NE~day5 / SW~day10) に購入。
            # day0 に買うと初期資金を食い潰して農場運用が崩壊する
            land_day_gate = 5 + n_extra * 5
            if n_extra < MAX_LAND_BUYS and day >= land_day_gate and farm["money"] >= LAND_PRICES[n_extra] + LAND_BUFFER:
                market.append(["BUY_LAND"])

        # --- フィード用小麦の購入 (自前生産が追いつくまでのつなぎ、2日分の予備) ---
        # E017: 予備は「実際の動物数」ベース (PLAN 全頭分だと初期資金を食い潰す)
        if USE_ANIMALS and farm["money"] >= 150:
            n_animals = _total_animals(shed, private.get("inventories", []), _count_placed_animals(farm["tiles"]))
            feed_need = max(6, n_animals * 2)
            total_wheat = shed.get(WHEAT, 0) + sum(inv.get(WHEAT, 0) for inv in private.get("inventories", []))
            if total_wheat < feed_need:
                market.append(["BUY_PRODUCT", WHEAT, min(2, feed_need - total_wheat)])

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

        # --- イチゴの種の補充 (メロン植えが止まっているとき、イチゴが優先) ---
        # E017: 種が高価 (100) なため資金に余裕があるときだけ購入
        # (初期資金を枯渇させると農場運用が崩壊する)
        if (
            USE_STRAWBERRY
            and not should_plant
            and seeds.get(STRAWBERRY, 0) < 2
            and farm["money"] >= 500
        ):
            market.append(["BUY_SEED", STRAWBERRY, min(8, int(farm["money"] - 300) // STRAWBERRY_SEED_COST)])

        # --- 肥料の仕入れ (E017: イチゴがあれば購入。施肥で継続作物は2倍収穫) ---
        if (
            USE_FERTILIZER
            and (melon_price >= SELL_THRESHOLD or USE_STRAWBERRY)
            and shed.get("FERTILIZER", 0) < FERTILIZER_MAX_STOCK
            and farm["money"] >= FERTILIZER_MIN_MONEY
        ):
            market.append(["BUY_PRODUCT", "FERTILIZER", 1])

        # --- 10件制限のトリム (末尾の購入系が落ちても次のターンに再試行される) ---
        market = market[:10]

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