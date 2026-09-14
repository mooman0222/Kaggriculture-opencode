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
    "herd": int(os.environ.get("LB_HERD", "5")),
    "tomato_tiles": int(os.environ.get("LB_TOM", "12")), "tomato_until": int(os.environ.get("LB_TOM_DAY", "17")),
    "tom_sell_day": int(os.environ.get("LB_TOM_SELL", "27")), "tom_sell_px": int(os.environ.get("LB_TOM_PX", "250")),
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
        self.day = -1; self.target = {}; self.herders = set()

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
                    if not tl["fed_today"]: tasks[(x, y, "FEED")] = (-1, "WHEAT")  # an unfed animal skips production; two dry days and it escapes
                    if tl.get("yield_units", 0) > 0: tasks[(x, y, "HARVEST")] = (0 if tl["yield_units"] >= 3 else 1, None)
                    if not tl["cared_today"]: tasks[(x, y, "CARE")] = (0, None)  # every missed care day is a lost unit at the next production
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
                            tasks[(x, y, "WATER")] = (-1 if survive else 0, None)  # a missed survival day is a lost tile
                    yu = tl.get("yield_units", 0)
                    ripe = yu > 0 and (cd.get("ongoing") or age >= cd.get("max_yield_day", 99) or (yu >= 3 and age >= 3) or day >= 29)
                    if ripe: tasks[(x, y, "HARVEST")] = (0 if (cd.get("ongoing") and yu >= 2) or day >= 29 else 1, None)  # held yield caps at 4
                    # fertilizer: doubles a watered production (ongoing) or the daily bonus (one-time, in window)
                    in_window = (not cd.get("ongoing")) and (cd.get("max_yield_day", 99) + 1) // 2 <= age + 1 <= cd.get("max_yield_day", 99)
                    if (cd.get("ongoing") or in_window) and tl.get("fertilized_until_day", -1) < day and day <= 27:
                        tasks[(x, y, "FERTILIZE")] = (0 if cd.get("ongoing") else 1, "FERTILIZER")
                elif tl.get("kind") == "WEED":
                    tasks[(x, y, "DIG")] = (0 if day <= P["replant_until_day"] else 3, None)  # every weed tile-day is lost production
        for i in range(len(units)):
            held = sum(v for k, v in inv(i).items() if k not in ("WHEAT", "FERTILIZER") and v > 0)
            near = dist(units[i], min(SHED_TILES, key=lambda s_: dist(units[i], s_)))
            last_day = day >= 29 and hour >= 24 - 2 - near  # the last day: bring everything home in time to sell
            # the shed holds 100; everything still carried at midnight is dropped at once and the overflow is destroyed
            if (held >= 3 and near <= 2) or (held >= 6 and near <= 4) or (held > 0 and hour >= 19 and near <= 5) or held >= 10 or (last_day and held > 0):
                tasks[("DROP", i, "DROP")] = (-1 if last_day or hour >= 20 else 1, None)
        shops = obs["town"]["unlocked_shops"]
        tomato_world = any(sh in ("PIZZA_SHOP", "FARMERS_MARKET") for sh in shops[:5])
        n_tomato = sum(1 for row in tiles for tl in row if isinstance(tl, dict) and tl.get("crop") == "TOMATO")
        want_tomato = tomato_world and day <= P["tomato_until"] and n_tomato < P["tomato_tiles"]
        if day <= P["replant_until_day"]:
            budget = dict(seeds); tom_left = P["tomato_tiles"] - n_tomato if want_tomato else 0
            for (x, y) in empties:
                order = ("TOMATO", "WHEAT", "CARROT", "STRAWBERRY") if tom_left > 0 else ("WHEAT", "CARROT", "STRAWBERRY", "TOMATO")
                for crop in order:
                    if budget.get(crop, 0) > 0:
                        tasks[(x, y, "PLANT")] = (0, crop); budget[crop] -= 1
                        if crop == "TOMATO": tom_left -= 1
                        break

        # ---------------- assignment: sticky targets + global nearest-first matching
        acts = [["PASS"] for _ in units]
        if hour <= 1: self.target = {}
        assigned = set()
        if hour <= 1 or len(getattr(self, "zone_tiles", {})) != len(units):
            K = max(1, len(units) - 1)
            animals_t = sorted(((x, y) for y, row in enumerate(tiles) for x, tl in enumerate(row) if isinstance(tl, dict) and "animal" in tl), key=lambda t: (t[0] // 3, t[1], t[0]))
            plants_t = [(x, y) for y, row in enumerate(tiles) for x, tl in enumerate(row) if tl != "LOCKED" and (x, y) not in set(animals_t)
                        and not (isinstance(tl, dict) and tl.get("kind") in ("COOP", "PASTURE"))]
            M = min(K, max(1, math.ceil(len(animals_t) / P["herd"]))) if animals_t else 0  # <= 5 animals per herder (feed+care+collect+harvest fit in a day)
            self.zone_tiles = {0: set()}; self.herders = set(range(1, 1 + M))
            for m in range(M):
                self.zone_tiles[1 + m] = set(animals_t[m::M])
            Kp = K - M
            if Kp > 0 and plants_t:
                plants_t.sort(key=lambda t: (t[1] // 3, t[0] if (t[1] // 3) % 2 == 0 else -t[0]))
                cents = [plants_t[int(k * len(plants_t) / Kp)] for k in range(Kp)]
                groups = {}
                for _ in range(6):
                    groups = {k: [] for k in range(Kp)}
                    for t in plants_t: groups[min(range(Kp), key=lambda k: dist(t, cents[k]))].append(t)
                    cents = [(sum(a for a, _ in g_) / len(g_), sum(b for _, b in g_) / len(g_)) if g_ else cents[k] for k, g_ in groups.items()]
                for k in range(Kp): self.zone_tiles[1 + M + k] = set(groups.get(k, []))
            for i in range(len(units)): self.zone_tiles.setdefault(i, set())
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
                         and tiles[y][x].get("fertilized_until_day", -1) <= day and CROPS[tiles[y][x]["crop"]].get("ongoing"))
                if nf: q = min(nf, 8, fert); acts[i] = ["PICKUP", "FERTILIZER", q]; fert -= q
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
                herder = i in self.herders
                zone_pen = 0 if (not zt or key[0] == "DROP" or key[:2] in zt or (key[2] == "FEED" and hour >= 14)) else (4 if herder else 12)
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
            market += [["HIRE"] for _ in range(min(P["hands"], 10))]  # the 10-order cap: at most ten hires at dawn
        # feed wheat: keep a reserve in the shed
        wheat_need = n_animals + 4 - shed.get("WHEAT", 0) - sum(inv(i).get("WHEAT", 0) for i in range(len(units)))
        if wheat_need > 0 and hour in (0, 23):
            market.append(["BUY_PRODUCT", "WHEAT", min(wheat_need, 30)])  # grown wheat feeds the herd; buy only the shortfall
        # seeds for empties (cheap wheat backbone; premium crops handled by strategy later)
        if empties and day <= P["replant_until_day"] and hour in (0, 6, 12, 18):
            maturing = sum(1 for y, row in enumerate(tiles) for x, tl in enumerate(row) if isinstance(tl, dict) and tl.get("kind") == "PLANT"
                           and not CROPS[tl["crop"]].get("ongoing") and day - tl["planted_day"] >= CROPS[tl["crop"]].get("max_yield_day", 99) - 1)
            want = len(empties) + sum(1 for k in tasks if k[2] == "DIG") + maturing - sum(seeds.values())
            if want_tomato and seeds.get("TOMATO", 0) < 4 and money > 1000:
                market.append(["BUY_SEED", "TOMATO", min(8, P["tomato_tiles"] - n_tomato)])
            if want > 0 and money > 300:
                crop = "CARROT" if prices.get("CARROT", 0) >= 1.5 * BASE["CARROT"] and day <= 26 else "WHEAT"
                market.append(["BUY_SEED", crop, min(want, 30)])
        # sells
        final = step >= 716
        for item in ("WOOL", "MILK", "STRAWBERRY", "MELON", "EGG", "TOMATO", "CARROT", "FERTILIZER", "WHEAT"):
            q = shed.get(item, 0)
            if item == "TOMATO" and not final:
                # hold for the scarcity premium, then trickle 6 units at three hours a day so the drain keeps the price up;
                # release stock early only under shed pressure; the final day dumps the rest
                pressure = sum(shed.values()) > 85
                if day >= 29: q = min(q, 15)
                elif (day >= 26 and prices.get("TOMATO", 0) >= 150 and hour in (2, 10, 18)) or prices.get("TOMATO", 0) >= P["tom_sell_px"]: q = min(q, 6)
                elif pressure: q = min(q, 6)
                else: q = 0
            if item == "WHEAT" and not final: q = q - (n_animals + 4)  # one day of feed stays home
            if item == "FERTILIZER" and not final: q = q - (36 if day <= 27 else 0)  # keep a stock for fertilizing (33 strawberries / 3 days)
            if q <= 0: continue
            if prices.get(item, 0) < 2: continue
            if item != "WHEAT" and prices[item] < P["sell_hold"] * BASE[item] and day < 28: continue
            market.append(["SELL", item, int(q)])
            if len(market) >= 10: break
        return {"farmer": acts[0], "hands": acts[1:], "market": market[:10]}
