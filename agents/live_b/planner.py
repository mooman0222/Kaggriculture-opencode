"""Live farm planner (day 12+): per-turn task assignment from the observation, no tapes.

Tasks: FEED (carry wheat) > HARVEST > WATER > CARE > COLLECT_FERTILIZER > PLANT, assigned greedily by distance.
Market: hire at dawn, keep a feed-wheat reserve, replant harvested one-time crops, sell shed products by policy.
"""
from __future__ import annotations

import math, os
try:
    from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS
except Exception:  # fallback table (engine 1.32.7)
    CROPS = {"WHEAT": {"max_yield_day": 4, "ongoing": False}, "CARROT": {"max_yield_day": 3, "ongoing": False}, "MELON": {"max_yield_day": 10, "ongoing": False},
             "TOMATO": {"ongoing": True}, "STRAWBERRY": {"ongoing": True}}

SHED_TILES = [(4, 4), (5, 4), (4, 5), (5, 5)]
CROP_SEED = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
BASE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}
P = {
    "hands": int(os.environ.get("LB_HANDS", "10")),
    "feed_reserve": int(os.environ.get("LB_FEED", "40")),
    "sell_hold": float(os.environ.get("LB_HOLD", "0.0")),   # sell if price >= hold*base (0 = always sell)
    "replant_until_day": int(os.environ.get("LB_REPLANT", "27")),
}


def dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step_toward(pos, tgt):
    x, y = pos
    if tgt[0] != x: return ["EAST" if tgt[0] > x else "WEST"]
    if tgt[1] != y: return ["SOUTH" if tgt[1] > y else "NORTH"]
    return None


class Planner:
    def __init__(self):
        self.day = -1; self.target = {}

    def act(self, obs, tape_action=None):
        step, seat = int(obs["step"]), int(obs["player"])
        day, hour = step // 24, step % 24
        farm = obs["farms"][seat]; private = obs["private"]; tiles = farm["tiles"]
        prices = obs["market"]["prices"]; shed = private["shed"]; seeds = private["seeds"]
        units = [tuple(farm["farmer"]), *[tuple(h) for h in farm["hands"]]]
        invs = private["inventories"]
        inv = lambda i: invs[i] if i < len(invs) else {}
        money = farm["money"]
        market = []

        # ---------------- tasks from tiles (tier 0 = deadline today)
        tasks = {}  # (x,y,op) -> (tier, need)
        n_animals = 0; empties = []
        for y, row in enumerate(tiles):
            for x, tl in enumerate(row):
                if tl == "LOCKED": continue
                if tl is None:
                    empties.append((x, y)); continue
                if "animal" in tl:
                    n_animals += 1
                    if not tl["fed_today"]: tasks[(x, y, "FEED")] = (0, "WHEAT")
                    if tl.get("yield_units", 0) > 0: tasks[(x, y, "HARVEST")] = (0 if tl["yield_units"] >= 3 else 1, None)
                    if not tl["cared_today"]: tasks[(x, y, "CARE")] = (1, None)  # care bonus doubles the next production
                    if tl["fertilizer_available"]: tasks[(x, y, "COLLECT_FERTILIZER")] = (1, None)
                elif tl.get("kind") == "PLANT":
                    cd = CROPS[tl["crop"]]; age = day - tl["planted_day"]
                    if not tl["watered_today"]:
                        survive = tl.get("consecutive_unwatered", 0) >= 1          # a second dry day kills the plant
                        if cd.get("ongoing"):
                            pays = tl.get("fertilized_until_day", -1) >= day        # fertilizer doubles only a watered production
                        else:
                            pays = (cd.get("max_yield_day", 99) + 1) // 2 <= age <= cd.get("max_yield_day", 99)  # bonus window
                        if survive or pays:
                            tasks[(x, y, "WATER")] = (0 if survive or hour >= 14 else 1, None)
                    ripe = tl.get("yield_units", 0) > 0 and (cd.get("ongoing") or age >= cd.get("max_yield_day", 99) or day >= 29)
                    if ripe: tasks[(x, y, "HARVEST")] = (1, None)
                    # fertilizer doubles every watered production of an ongoing crop (3 days per unit)
                    if cd.get("ongoing") and tl.get("fertilized_until_day", -1) < day and day <= 27:
                        tasks[(x, y, "FERTILIZE")] = (1, "FERTILIZER")
                elif tl.get("kind") == "WEED":
                    tasks[(x, y, "DIG")] = (1 if day <= P["replant_until_day"] else 3, None)
        for i in range(len(units)):
            held = sum(v for k, v in inv(i).items() if k not in ("WHEAT", "FERTILIZER") and v > 0)
            near = dist(units[i], min(SHED_TILES, key=lambda s_: dist(units[i], s_)))
            if (held >= 6 and near <= 3) or (held > 0 and hour >= 21 and near <= 2) or held >= 12:
                tasks[("DROP", i, "DROP")] = (1, None)
        if day <= P["replant_until_day"]:
            budget = dict(seeds)
            for (x, y) in empties:
                for crop in ("TOMATO", "STRAWBERRY", "CARROT", "WHEAT"):
                    if budget.get(crop, 0) > 0:
                        tasks[(x, y, "PLANT")] = (1, crop); budget[crop] -= 1; break

        # ---------------- assignment: sticky targets + global nearest-first matching
        acts = [["PASS"] for _ in units]
        if hour <= 1: self.target = {}
        assigned = set()
        if hour <= 1 or len(getattr(self, "zone", {})) != len(units):
            work = [(x, y) for y, row in enumerate(tiles) for x, tl in enumerate(row) if tl != "LOCKED" and not (isinstance(tl, dict) and tl.get("kind") in ("COOP", "PASTURE") and "animal" not in tl)]
            K = max(1, len(units) - 1)
            # deterministic k-means (few iterations) seeded on a serpentine order of the work tiles
            work.sort(key=lambda t: (t[1] // 3, t[0] if (t[1] // 3) % 2 == 0 else -t[0]))
            cents = [work[int(k * len(work) / K)] for k in range(K)] if work else []
            groups = {}
            for _ in range(6):
                groups = {k: [] for k in range(K)}
                for t in work: groups[min(range(K), key=lambda k: dist(t, cents[k]))].append(t)
                cents = [(sum(a for a, _ in g_) / len(g_), sum(b for _, b in g_) / len(g_)) if g_ else cents[k] for k, g_ in groups.items()]
            self.zone_tiles = {i + 1: set(groups.get(i, [])) for i in range(K)}
            self.zone_tiles[0] = set()  # farmer: free agent
        if hour == 1 and n_animals and shed.get("WHEAT", 0) > 0:
            left = shed.get("WHEAT", 0)
            for i in range(1, len(units)):
                na = sum(1 for (x, y) in self.zone_tiles.get(i, ()) if isinstance(tiles[y][x], dict) and "animal" in tiles[y][x])
                if na and units[i] in SHED_TILES and left > 0:
                    q = min(na + 1, left); acts[i] = ["PICKUP", "WHEAT", q]; left -= q
            fert = shed.get("FERTILIZER", 0)
            for i in range(1, len(units)):
                if acts[i] != ["PASS"] or units[i] not in SHED_TILES or fert <= 0: continue
                nf = sum(1 for (x, y) in self.zone_tiles.get(i, ()) if isinstance(tiles[y][x], dict) and tiles[y][x].get("kind") == "PLANT"
                         and CROPS[tiles[y][x]["crop"]].get("ongoing") and tiles[y][x].get("fertilized_until_day", -1) < day)
                if nf: q = min(nf, 3, fert); acts[i] = ["PICKUP", "FERTILIZER", q]; fert -= q
        def can(i, key):
            op = key[2]
            if op == "FEED" and inv(i).get("WHEAT", 0) <= 0: return False
            if op == "FERTILIZE" and inv(i).get("FERTILIZER", 0) <= 0: return False
            return True
        # keep yesterday's-turn targets that are still live
        for i in list(getattr(self, "target", {}).keys()):
            key = self.target[i]
            if i >= len(units) or key not in tasks or key in assigned or not can(i, key) or acts[i] != ["PASS"]:
                self.target.pop(i, None); continue
            assigned.add(key)
        # global matching by (tier, distance)
        pairs = []
        for i in range(len(units)):
            if acts[i] != ["PASS"] or i in self.target: continue
            for key, (tier, need) in tasks.items():
                if key in assigned or not can(i, key): continue
                if key[0] == "DROP" and key[1] != i: continue
                zt = self.zone_tiles.get(i, set())
                zone_pen = 0 if (not zt or key[0] == "DROP" or key[:2] in zt) else 12
                d = dist(units[i], min(SHED_TILES, key=lambda s_: dist(units[i], s_))) if key[0] == "DROP" else dist(units[i], key[:2])
                pairs.append((-10 if d == 0 else d + 4 * tier + zone_pen, tier, i, str(key), key))  # finish work on the tile you stand on
        pairs.sort(key=lambda x: x[:4])
        for sc, tier, i, _k, key in pairs:
            if i in self.target or key in assigned: continue
            self.target[i] = key; assigned.add(key)
        def tgt_of(i, key):
            return min(SHED_TILES, key=lambda s_: dist(units[i], s_)) if key[0] == "DROP" else key[:2]
        for i, key in self.target.items():
            if i >= len(units) or acts[i] != ["PASS"]: continue
            mv = step_toward(units[i], tgt_of(i, key))
            acts[i] = mv if mv else (["DROP"] if key[0] == "DROP" else ([key[2], tasks[key][1]] if key[2] == "PLANT" else [key[2]]))
        done_now = {i for i, key in self.target.items() if i < len(units) and units[i] == tgt_of(i, key)}
        for i in done_now: self.target.pop(i, None)
        # feeders: refill wheat when animals still need feeding and nobody carries wheat
        need_feed = sum(1 for k in tasks if k[2] == "FEED")
        carried = sum(inv(i).get("WHEAT", 0) for i in range(len(units)))
        if need_feed > carried and shed.get("WHEAT", 0) > 0:
            for i in range(len(units)):
                if acts[i] != ["PASS"]: continue
                if units[i] in SHED_TILES: acts[i] = ["PICKUP", "WHEAT", min(8, shed.get("WHEAT", 0))]; break
                mv = step_toward(units[i], min(SHED_TILES, key=lambda s_: dist(units[i], s_)))
                if mv: acts[i] = mv; break
        # idle carriers drop produce at the shed so it can be sold today
        for i in range(len(units)):
            if acts[i] == ["PASS"] and any(k not in ("WHEAT", "FERTILIZER") for k, v in inv(i).items() if v > 0):
                if units[i] in SHED_TILES: acts[i] = ["DROP"]
                else:
                    mv = step_toward(units[i], min(SHED_TILES, key=lambda s_: dist(units[i], s_)))
                    if mv: acts[i] = mv

        # ---------------- market
        if hour == 0:
            market += [["HIRE"] for _ in range(P["hands"])]
        # feed wheat: keep a reserve in the shed
        wheat_need = n_animals + 4 - shed.get("WHEAT", 0) - sum(inv(i).get("WHEAT", 0) for i in range(len(units)))
        if wheat_need > 0 and hour in (0, 23):
            market.append(["BUY_PRODUCT", "WHEAT", min(wheat_need, 30)])  # grown wheat feeds the herd; buy only the shortfall
        # seeds for empties (cheap wheat backbone; premium crops handled by strategy later)
        if empties and day <= P["replant_until_day"] and hour in (0, 6, 12, 18):
            want = len(empties) + sum(1 for k in tasks if k[2] == "DIG") - sum(seeds.values())
            if want > 0 and money > 300:
                crop = "CARROT" if prices.get("CARROT", 0) >= 1.5 * BASE["CARROT"] and day <= 26 else "WHEAT"
                market.append(["BUY_SEED", crop, min(want, 30)])
        # sells
        for item in ("WOOL", "MILK", "STRAWBERRY", "MELON", "EGG", "TOMATO", "CARROT", "FERTILIZER", "WHEAT"):
            q = shed.get(item, 0)
            if item == "WHEAT": q = q - (2 * n_animals + 6)
            if item == "FERTILIZER": q = q - (12 if day <= 27 else 0)  # keep a stock for fertilizing
            if q <= 0: continue
            if prices.get(item, 0) < 2: continue
            if item != "WHEAT" and prices[item] < P["sell_hold"] * BASE[item] and day < 28: continue
            market.append(["SELL", item, int(q)])
            if len(market) >= 10: break
        return {"farmer": acts[0], "hands": acts[1:], "market": market[:10]}
