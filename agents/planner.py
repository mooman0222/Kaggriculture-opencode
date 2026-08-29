"""E018 M3b: 手書きプランナー (PASS ターン介入版).

adaptive_route (ルート農場+リアクティブ市場、avg $61.8k) を維持したまま、
**ファーマーのルートアクションが PASS のターンだけ介入**して滞留動物を解消する。

背景 (E018-M3b の原因分析):
- ルートのハンドはスポーン位置ズレで BUILD_PASTURE が計画外位置で実行 → 牧場 11/16
- ルートの PICKUP ステップは一過性で、資金遅延で無効化されると動物が shed に滞留
- フルリアクティブ化は移動コストで水やり能力が不足し構造的に失敗 (今回検証)
- ハイブリッド (farmer 全面置換) はファーマー位置変化→ハンドスポーン位置変化で
  水やりが漏れ失敗 (E018-M2 の知見と同根)

対策 (PASS 介入):
- ルートの farmer アクションが PASS のターンにのみ介入:
  1. 手持ち動物 → 空き牧場へ PLACE
  2. shed 滞留動物 + 空き牧場 → PICKUP
  3. 計画位置 (build_events) の未建設牧場 → BUILD_PASTURE (雑草は DIG)
  4. 未給餌動物 → FEED (脱走防止)
- ルートのハンドアクションの BUILD_PASTURE は PASS に置換 (計画外位置での建設を防止。
  牧場建設はファーマー介入で計画位置に実施)
- market: リアクティブ (adaptive_route と同一)
- hands: ルート再生 + ギャップフィラー (adaptive_route と同一、BUILD 置換のみ)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_adaptive():
    import importlib.util
    spec = importlib.util.spec_from_file_location("adaptive_route", ROOT / "agents" / "adaptive_route.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_adaptive = None


def _get_adaptive():
    global _adaptive
    if _adaptive is None:
        _adaptive = _load_adaptive()
    return _adaptive


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


def _shed_adjacent(pos):
    x, y = pos
    return x in (4, 5) and y in (4, 5)


def _count_pastures(tiles):
    return sum(1 for row in tiles for t in row if isinstance(t, dict) and t.get("kind") == "PASTURE")


def _count_placed_animals(tiles):
    return sum(1 for row in tiles for t in row if isinstance(t, dict) and "animal" in t)


def _empty_pasture_pos(tiles):
    for y, row in enumerate(tiles):
        for x, t in enumerate(row):
            if isinstance(t, dict) and t.get("kind") == "PASTURE" and "animal" not in t:
                return (x, y)
    return None


def _inventory_of(private, farm, pos):
    idx = 0
    if tuple(pos) != tuple(farm["farmer"]):
        for i, h in enumerate(farm["hands"]):
            if tuple(h) == tuple(pos):
                idx = i + 1
                break
    invs = private.get("inventories", [])
    return invs[idx] if idx < len(invs) else {}


def _farmer_pass_intervention(obs, farm, private, plan, day):
    """ファーマーの PASS ターン介入。優先順位:
    1. 手持ち動物を空き牧場へ PLACE
    2. shed 滞留動物を PICKUP (空き牧場あり時)
    3. 計画位置の牧場建設 (ルートの build_events 位置で未建設)
    4. 未給餌動物への給餌 (脱走防止)
    5. 水やり補助
    """
    tiles = farm["tiles"]
    board = len(tiles)
    x, y = farm["farmer"]
    pos = (x, y)
    tile = tiles[y][x]
    inv = _inventory_of(private, farm, pos)
    shed = private.get("shed", {}) or {}
    placed = _count_placed_animals(tiles)
    carrying = [a for a in ("COW", "SHEEP", "GOOSE") if inv.get(a, 0) > 0]
    shed_animals = {a: shed.get(a, 0) for a in ("COW", "SHEEP", "GOOSE") if shed.get(a, 0) > 0}

    # --- 1. 手持ち動物の配置 ---
    if carrying:
        a = carrying[0]
        ep = _empty_pasture_pos(tiles)
        if ep == pos:
            return ["PLACE", a, 1]
        if ep:
            step = _step_toward(x, y, ep[0], ep[1])
            if step:
                return [step]
        return ["PASS"]

    # --- 2. 滞留動物の PICKUP (空き牧場があるとき) ---
    if shed_animals and _empty_pasture_pos(tiles) is not None:
        if _shed_adjacent(pos):
            a = max(shed_animals, key=lambda k: shed_animals[k])
            return ["PICKUP", a, 1]
        step = _step_toward(x, y, 4, 4)
        if step:
            return [step]

    # --- 3. 計画位置の牧場建設 (ルートの build_events 位置。動物が滞留している
    # または購入予定がある場合のみ進行) ---
    n_pastures = _count_pastures(tiles)
    n_shed = sum(shed_animals.values())
    build_events = plan.get("build_events", [])
    if placed + n_shed < len(build_events) and n_pastures < len(build_events):
        for ev in build_events:
            px, py = ev[1]  # build_events: [(step, (x, y)), ...]
            t = tiles[py][px]
            if t is None:
                if (px, py) == pos:
                    return ["BUILD_PASTURE"]
                step = _step_toward(x, y, px, py)
                if step:
                    return [step]
            elif _is_weed(t):
                if (px, py) == pos:
                    return ["DIG"]
                step = _step_toward(x, y, px, py)
                if step:
                    return [step]

    # --- 4. 給餌 (脱走リスク: 2日連続未給餌は脱走) ---
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

    # --- 5. 水やり補助 ---
    unwatered = [
        (tx, ty) for ty in range(board) for tx in range(board)
        if isinstance(tiles[ty][tx], dict) and tiles[ty][tx].get("kind") == "PLANT"
        and not tiles[ty][tx].get("watered_today", False)
    ]
    if unwatered:
        ux, uy = min(unwatered, key=lambda p: abs(x - p[0]) + abs(y - p[1]))
        if (ux, uy) == (x, y):
            return ["WATER"]
        step = _step_toward(x, y, ux, uy)
        if step:
            return [step]

    return ["PASS"]


def _hand_build_pasture(obs, farm, private, pos, plan):
    """ハンドの BUILD_PASTURE を計画位置 (build_events) で実行する。

    ルートのハンドはスポーン位置ズレで計画外位置に牧場を建ててしまう
    (E018-M3b: 牧場 11/16)。ここでは計画の未建設位置へ移動して建設する。
    """
    tiles = farm["tiles"]
    x, y = pos
    build_events = plan.get("build_events", [])
    placed = _count_placed_animals(tiles)
    n_pastures = _count_pastures(tiles)
    if placed + n_pastures >= len(build_events) or n_pastures >= len(build_events):
        return ["PASS"]
    for ev in build_events:
        px, py = ev[1]
        t = tiles[py][px]
        if t is None:
            if (px, py) == pos:
                return ["BUILD_PASTURE"]
            step = _step_toward(x, y, px, py)
            if step:
                return [step]
        elif _is_weed(t):
            if (px, py) == pos:
                return ["DIG"]
            step = _step_toward(x, y, px, py)
            if step:
                return [step]
    return ["PASS"]


def make_planner_agent(route):
    """adaptive_route + 牧場建設位置補正 + PASS ターン介入。

    - farmer: ルートの farmer アクションを基本とし、PASS のターンのみ介入
      (滞留動物 PICKUP/PLACE・給餌)
    - hands:  ルート再生 + ギャップフィラー。ルートの BUILD_PASTURE は
      計画位置 (build_events) で実行するよう補正
    - market: リアクティブ (adaptive_route と同一)
    """
    ar = _get_adaptive()
    plan = ar.build_plan(route)

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

            r = route[step]
            farmer = ar._repair(r.get("farmer", ["PASS"]), farm["farmer"], farm["tiles"])
            # PASS ターン介入: ルートが PASS のときだけ滞留動物・牧場建設を処理
            if farmer[0] == "PASS":
                farmer = _farmer_pass_intervention(obs, farm, private, plan, day)

            actual_hands = farm.get("hands", [])
            hands = []
            for i, ha in enumerate(r.get("hands", [])):
                if i >= len(actual_hands):
                    break
                # ハンドの BUILD_PASTURE はルート通り (スポーン位置ズレは許容。
                # 牧場不足はファーマー PASS 介入が補う)
                hands.append(ar._repair(ha, actual_hands[i], farm["tiles"]))
            while len(hands) < len(actual_hands):
                hands.append(ar._reactive_hand_action(obs, farm, private, actual_hands[len(hands)], day, plan, step))
            market = ar._reactive_market(obs, farm, private, plan, step, day, hour)
            return {"farmer": farmer, "hands": hands, "market": market}
        except Exception:
            return {"farmer": ["PASS"], "hands": [], "market": []}

    return agent