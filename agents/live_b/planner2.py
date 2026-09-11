"""Patrol planner (day 12+): each hand owns a zone and walks a fixed serpentine route through it every day,
finishing all pending work on a tile before moving on. Herders own <= 5 animals each (feed, care, collect,
harvest); planters own plant/empty tiles (water when it pays or saves the plant, fertilize, harvest, dig, plant).
The farmer is a free agent filling gaps. Market: hire at dawn, feed reserve, proactive seeds, sell everything else."""
from __future__ import annotations
import math, os
try:
    from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS
except Exception:
    CROPS = {"WHEAT": {"max_yield_day": 4, "ongoing": False}, "CARROT": {"max_yield_day": 3, "ongoing": False}, "MELON": {"max_yield_day": 10, "ongoing": False},
             "TOMATO": {"ongoing": True}, "STRAWBERRY": {"ongoing": True}}

SHED = [(4, 4), (5, 4), (4, 5), (5, 5)]
BASE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}
P = {"hands": int(os.environ.get("LB_HANDS", "10")), "herd": int(os.environ.get("LB_HERD", "5")), "fert_reserve": int(os.environ.get("LB_FRES", "36")),
     "replant_until": int(os.environ.get("LB_REPLANT", "27"))}


def dist(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step_toward(pos, tgt):
    x, y = pos
    if tgt[0] != x: return ["EAST" if tgt[0] > x else "WEST"]
    if tgt[1] != y: return ["SOUTH" if tgt[1] > y else "NORTH"]
    return None


def pending(tile, day, hour, inv, seeds, plant_crop):
    """Ordered list of ops still worth doing on this tile now (first = most urgent)."""
    ops = []
    if tile is None:
        if plant_crop and seeds.get(plant_crop, 0) > 0 and day <= P["replant_until"]: ops.append(["PLANT", plant_crop])
        return ops
    if not isinstance(tile, dict): return ops
    if "animal" in tile:
        if not tile["fed_today"] and inv.get("WHEAT", 0) > 0: ops.append(["FEED"])
        if tile.get("yield_units", 0) > 0: ops.append(["HARVEST"])
        if not tile["cared_today"]: ops.append(["CARE"])
        if tile["fertilizer_available"]: ops.append(["COLLECT_FERTILIZER"])
        return ops
    kind = tile.get("kind")
    if kind == "WEED":
        if day <= P["replant_until"]: ops.append(["DIG"])
        return ops
    if kind == "PLANT":
        cd = CROPS[tile["crop"]]; age = day - tile["planted_day"]; yu = tile.get("yield_units", 0)
        ripe = yu > 0 and (cd.get("ongoing") or age >= cd.get("max_yield_day", 99) or (yu >= 3 and age >= 3) or day >= 29)
        if ripe: ops.append(["HARVEST"])
        if not tile["watered_today"]:
            survive = tile.get("consecutive_unwatered", 0) >= 1
            if cd.get("ongoing"): pays = tile.get("fertilized_until_day", -1) >= day
            else: pays = (cd.get("max_yield_day", 99) + 1) // 2 <= age <= cd.get("max_yield_day", 99)
            if survive or pays: ops.append(["WATER"])
        if inv.get("FERTILIZER", 0) > 0 and tile.get("fertilized_until_day", -1) < day and day <= 27:
            in_window = (not cd.get("ongoing")) and (cd.get("max_yield_day", 99) + 1) // 2 <= age + 1 <= cd.get("max_yield_day", 99)
            if cd.get("ongoing") or in_window: ops.append(["FERTILIZE"])
    return ops


class Planner:
    def __init__(self):
        self.routes = {}; self.idx = {}; self.herders = set(); self.day_built = -1

    def build_zones(self, tiles, units):
        K = max(1, len(units) - 1)
        animals = sorted(((x, y) for y, row in enumerate(tiles) for x, tl in enumerate(row) if isinstance(tl, dict) and "animal" in tl), key=lambda t: (t[0] // 3, t[1], t[0]))
        aset = set(animals)
        plants = [(x, y) for y, row in enumerate(tiles) for x, tl in enumerate(row) if tl != "LOCKED" and (x, y) not in aset
                  and not (isinstance(tl, dict) and tl.get("kind") in ("COOP", "PASTURE"))]
        M = min(K, max(1, math.ceil(len(animals) / P["herd"]))) if animals else 0
        zones = {0: []}; self.herders = set(range(1, 1 + M))
        # contiguous chunks along the spatial sort: each herder's animals sit together
        per = math.ceil(len(animals) / M) if M else 0
        for m in range(M): zones[1 + m] = animals[m * per:(m + 1) * per]
        Kp = K - M
        if Kp > 0 and plants:
            plants.sort(key=lambda t: (t[1] // 3, t[0] if (t[1] // 3) % 2 == 0 else -t[0]))
            cents = [plants[int(k * len(plants) / Kp)] for k in range(Kp)]
            groups = {}
            for _ in range(8):
                groups = {k: [] for k in range(Kp)}
                for t in plants: groups[min(range(Kp), key=lambda k: dist(t, cents[k]))].append(t)
                cents = [(sum(a for a, _ in g) / len(g), sum(b for _, b in g) / len(g)) if g else cents[k] for k, g in groups.items()]
            for k in range(Kp): zones[1 + M + k] = groups.get(k, [])
        for i in range(len(units)): zones.setdefault(i, [])
        # serpentine route order within each zone, starting nearest the shed
        self.routes = {}
        for i, z in zones.items():
            z = sorted(z, key=lambda t: (t[1], t[0] if t[1] % 2 == 0 else -t[0]))
            self.routes[i] = z
        self.idx = {i: 0 for i in zones}

    def act(self, obs):
        step, seat = int(obs["step"]), int(obs["player"]); day, hour = step // 24, step % 24
        farm = obs["farms"][seat]; private = obs["private"]; tiles = farm["tiles"]
        prices = obs["market"]["prices"]; shed = private["shed"]; seeds = private["seeds"]
        units = [tuple(farm["farmer"]), *[tuple(h) for h in farm["hands"]]]
        invs = private["inventories"]; inv = lambda i: invs[i] if i < len(invs) else {}
        money = farm["money"]; market = []
        n_animals = sum(1 for row in tiles for tl in row if isinstance(tl, dict) and "animal" in tl)
        empties = [(x, y) for y, row in enumerate(tiles) for x, tl in enumerate(row) if tl is None]
        plant_crop = "WHEAT"
        if self.day_built != day and hour >= 1:
            self.build_zones(tiles, units); self.day_built = day
        acts = [["PASS"] for _ in units]
        claimed = set()
        # dawn loading at the shed
        if hour == 1:
            left = shed.get("WHEAT", 0); fert = max(0, shed.get("FERTILIZER", 0))
            for i in range(1, len(units)):
                if units[i] not in SHED: continue
                if i in self.herders:
                    na = len(self.routes.get(i, []))
                    if na and left > 0: q = min(na + 1, left); acts[i] = ["PICKUP", "WHEAT", q]; left -= q
                else:
                    nf = sum(1 for (x, y) in self.routes.get(i, []) if isinstance(tiles[y][x], dict) and tiles[y][x].get("kind") == "PLANT"
                             and tiles[y][x].get("fertilized_until_day", -1) <= day and CROPS[tiles[y][x]["crop"]].get("ongoing"))
                    if nf and fert > 0: q = min(nf, 8, fert); acts[i] = ["PICKUP", "FERTILIZER", q]; fert -= q
        # patrol: hands
        for i in range(1, len(units)):
            if acts[i] != ["PASS"]: continue
            pos = units[i]; route = self.routes.get(i, [])
            held = sum(v for k, v in inv(i).items() if k not in ("WHEAT", "FERTILIZER") and v > 0)
            near = dist(pos, min(SHED, key=lambda s_: dist(pos, s_)))
            if (held >= 8 and near <= 2) or (held > 0 and hour >= 21 and near <= 2) or (day >= 29 and held > 0 and hour >= 24 - 2 - near) or held >= 14:
                mv = None if pos in SHED else step_toward(pos, min(SHED, key=lambda s_: dist(pos, s_)))
                acts[i] = mv or ["DROP"]; continue
            # current tile work first
            here = pending(tiles[pos[1]][pos[0]], day, hour, inv(i), seeds, plant_crop) if pos not in claimed else []
            if here:
                acts[i] = here[0]; claimed.add(pos); continue
            # next tile on the route with pending work (cyclic scan)
            n = len(route); tgt = None
            for k in range(n):
                j = (self.idx[i] + k) % n; t = route[j]
                if t in claimed: continue
                if pending(tiles[t[1]][t[0]], day, hour, inv(i), seeds, plant_crop):
                    tgt = t; self.idx[i] = j; break
            if tgt is None:
                # zone done: help anywhere (nearest pending tile), else pass
                best = None
                for y, row in enumerate(tiles):
                    for x, tl in enumerate(row):
                        if (x, y) in claimed or tl == "LOCKED": continue
                        ops = pending(tl, day, hour, inv(i), seeds, plant_crop)
                        if ops and (best is None or dist(pos, (x, y)) < best[0]): best = (dist(pos, (x, y)), (x, y))
                if best: tgt = best[1]
            if tgt is None: continue
            claimed.add(tgt)
            mv = step_toward(pos, tgt)
            acts[i] = mv if mv else pending(tiles[tgt[1]][tgt[0]], day, hour, inv(i), seeds, plant_crop)[0]
        # farmer: free agent
        pos = units[0]
        here = pending(tiles[pos[1]][pos[0]], day, hour, inv(0), seeds, plant_crop) if pos not in claimed else []
        if here: acts[0] = here[0]
        else:
            best = None
            for y, row in enumerate(tiles):
                for x, tl in enumerate(row):
                    if (x, y) in claimed or tl == "LOCKED": continue
                    ops = pending(tl, day, hour, inv(0), seeds, plant_crop)
                    if ops and (best is None or dist(pos, (x, y)) < best[0]): best = (dist(pos, (x, y)), (x, y))
            if best:
                mv = step_toward(pos, best[1]); acts[0] = mv if mv else pending(tiles[best[1][1]][best[1][0]], day, hour, inv(0), seeds, plant_crop)[0]
        # ---------------- market
        if hour == 0: market += [["HIRE"] for _ in range(min(P["hands"], 10))]
        carried_w = sum(inv(i).get("WHEAT", 0) for i in range(len(units)))
        need = n_animals + 4 - shed.get("WHEAT", 0) - carried_w
        if need > 0 and hour in (0, 23): market.append(["BUY_PRODUCT", "WHEAT", min(need, 30)])
        if day <= P["replant_until"] and hour in (0, 6, 12, 18):
            maturing = sum(1 for row in tiles for tl in row if isinstance(tl, dict) and tl.get("kind") == "PLANT" and not CROPS[tl["crop"]].get("ongoing")
                           and day - tl["planted_day"] >= CROPS[tl["crop"]].get("max_yield_day", 99) - 1)
            weeds = sum(1 for row in tiles for tl in row if isinstance(tl, dict) and tl.get("kind") == "WEED")
            want = len(empties) + weeds + maturing - sum(seeds.values())
            if want > 0 and money > 300: market.append(["BUY_SEED", "WHEAT", min(want, 30)])
        final = step >= 716
        for item in ("WOOL", "MILK", "STRAWBERRY", "MELON", "EGG", "TOMATO", "CARROT", "FERTILIZER", "WHEAT"):
            q = shed.get(item, 0)
            if item == "WHEAT" and not final: q -= n_animals + 4
            if item == "FERTILIZER" and not final: q -= P["fert_reserve"] if day <= 27 else 0
            if q <= 0 or prices.get(item, 0) < 2: continue
            market.append(["SELL", item, int(q)])
            if len(market) >= 10: break
        return {"farmer": acts[0], "hands": acts[1:], "market": market[:10]}
