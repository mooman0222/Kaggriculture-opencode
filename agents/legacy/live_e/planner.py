"""live_e planner: demand-aware macro targets at dawn, zone patrol micro-assignment each step, price-aware market.

Macro (hour 0): hands, land, animals, crop-tile targets from shops (demand/day), money, day and the rival's visible farm.
Micro: herders own animal clusters (feed/care/collect/harvest), planters own tiles (water when it pays or saves the
plant, fertilize ongoing crops, harvest, dig, plant the macro's crop). Farmer fills gaps and places bought animals.
Market: hire at dawn, buy land/animals/seeds per macro, keep a feed reserve, sell premium items only at decent prices
(or under shed pressure / rival's imminent dump), sell volume items freely, liquidate at the end.
"""
from __future__ import annotations
import math, os

CROPS = {"WHEAT": dict(seed=10, first=2, maxd=4, ongoing=False, maxy=6), "CARROT": dict(seed=20, first=2, maxd=3, ongoing=False, maxy=4),
         "TOMATO": dict(seed=50, first=8, maxd=8, ongoing=True), "STRAWBERRY": dict(seed=100, first=10, maxd=10, ongoing=True),
         "MELON": dict(seed=80, first=10, maxd=12, ongoing=False, maxy=6)}
ANIMALS = {"GOOSE": dict(cost=300, struct="COOP", product="EGG"), "COW": dict(cost=400, struct="PASTURE", product="MILK"), "SHEEP": dict(cost=500, struct="PASTURE", product="WOOL")}
SHOPS = {"BAKERY": ["EGG", "WHEAT"], "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"], "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"], "YARN_STORE": ["WOOL"],
         "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"], "PET_CAFE": ["CARROT"], "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"], "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"]}
BASE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}
PREMIUM = ("STRAWBERRY", "MILK", "WOOL")
SHED = [(4, 4), (5, 4), (4, 5), (5, 5)]
LAND_PRICES = [1000, 2000, 4000]

P = {  # tunables (env overrides for A/B)
    "hands": [int(x) for x in os.environ.get("LE_HANDS", "8,9,9,10,11").split(",")],  # day6, d7, d8, d9, d10+
    "hands_late": [int(x) for x in os.environ.get("LE_HANDS_LATE", "10,9").split(",")],  # d28, d29
    "herd": int(os.environ.get("LE_HERD", "4")), "care_hour": int(os.environ.get("LE_CARE_HOUR", "10")), "drop_held": int(os.environ.get("LE_DROP", "4")), "land_ne_day": 6, "land_sw_day": int(os.environ.get("LE_SW_DAY", "9")), "land_last_day": 14,
    "straw_base": int(os.environ.get("LE_STRAW", "6")), "straw_per_shop": int(os.environ.get("LE_STRAW_PER", "5")), "straw_cap": 28, "straw_last_day": 16,
    "carrot_per_shop": 4, "carrot_from_day": 8, "carrot_last_day": 26, "wheat_last_day": 26,
    "cow_base": 6, "cow_per_shop": 2, "cow_cap": 12, "sheep_per_yarn": 4, "sheep_cap": 12, "goose_per_shop": 1, "goose_cap": 4, "animal_last_day": int(os.environ.get("LE_ANIMAL_LAST", "13")),
    "tiles_per_planter": int(os.environ.get("LE_TPP", "10")), "fert_load": int(os.environ.get("LE_FLOAD", "12")), "water_all": os.environ.get("LE_WATER_ALL", "0") == "1", "wander": int(os.environ.get("LE_WANDER", "3")), "hold_frac": float(os.environ.get("LE_HOLD", "0")), "vol_frac": 0.35, "shed_hi": 70, "fert_reserve": int(os.environ.get("LE_FRES", "24")),
    "opp_share": float(os.environ.get("LE_OPP_SHARE", "0.0")), "match_plus": int(os.environ.get("LE_MATCH", "1")), "match_straw": int(os.environ.get("LE_MATCH_STRAW", "1")), "wheat_per_shop": int(os.environ.get("LE_WHEAT_PER", "4")),
}


def dist(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step_toward(pos, tgt):
    x, y = pos
    if tgt[0] != x: return ["EAST" if tgt[0] > x else "WEST"]
    if tgt[1] != y: return ["SOUTH" if tgt[1] > y else "NORTH"]
    return None


def demand_per_day(shops):
    d = {k: 1 for k in BASE if k != "FERTILIZER"}
    for s in shops:
        prods = SHOPS[s]; m = 2 if len(prods) == 1 else 1
        for p in prods: d[p] += 6 * m
    return d


class Planner:
    def __init__(self):
        self.routes = {}; self.idx = {}; self.herders = set(); self.day_built = -1; self.day_planned = -1
        self.plan = {}; self.plant_quota = {}; self.build_queue = []; self.reserved = set(); self.want_fert = True

    # ---------------- macro -----------------
    def plan_day(self, obs, day, farm, opp, shops, money, tiles):
        dem = demand_per_day(shops)
        cnt = lambda p: sum((2 if len(SHOPS[s]) == 1 else 1) for s in shops if p in SHOPS[s])
        quads = len(farm.get("unlocked_quadrants", []))
        own = self.count_tiles(tiles); oth = self.count_tiles(opp["tiles"])
        if day >= 29: hands = P["hands_late"][1]
        elif day == 28: hands = P["hands_late"][0]
        else: hands = P["hands"][min(max(day - 6, 0), len(P["hands"]) - 1)]
        land = None  # resolved per step in land_price()
        # crop targets (tiles alive), rival supply reduces the premium target
        s_shops = cnt("STRAWBERRY")
        straw = min(P["straw_cap"], P["straw_base"] + P["straw_per_shop"] * s_shops)
        straw = max(straw, oth.get("STRAWBERRY", 0) + 2) if P["match_straw"] else straw
        straw = min(P["straw_cap"], straw) if day <= P["straw_last_day"] else 0
        n_anim = sum(v for k, v in own.items() if k in ANIMALS)
        wheat = n_anim + 4 + P["wheat_per_shop"] * cnt("WHEAT") if day <= P["wheat_last_day"] else 0
        carrot = P["carrot_per_shop"] * cnt("CARROT") if P["carrot_from_day"] <= day <= P["carrot_last_day"] else 0
        # animals
        want = {}
        if day <= P["animal_last_day"]:
            base = {"COW": min(P["cow_cap"], P["cow_base"] + P["cow_per_shop"] * cnt("MILK")),
                    "SHEEP": min(P["sheep_cap"], P["sheep_per_yarn"] * cnt("WOOL") // 2 + (3 if cnt("WOOL") else 0)),
                    "GOOSE": min(P["goose_cap"], P["goose_per_shop"] * cnt("EGG"))}
            for a in base: want[a] = max(base[a], min(P["sheep_cap"], oth.get(a, 0) + P["match_plus"]))
            value = {"SHEEP": dem["WOOL"] * BASE["WOOL"], "COW": dem["MILK"] * BASE["MILK"], "GOOSE": dem["EGG"] * BASE["EGG"]}
            want = dict(sorted(want.items(), key=lambda kv: -value[kv[0]]))
        self.plan = {"hands": hands, "land": land, "straw": straw, "carrot": carrot, "wheat": wheat, "want": want, "dem": dem, "own": own}

    @staticmethod
    def count_tiles(tiles):
        c = {}
        for row in tiles:
            for tl in row:
                if isinstance(tl, dict):
                    if tl.get("kind") == "PLANT": c[tl["crop"]] = c.get(tl["crop"], 0) + 1
                    elif "animal" in tl: c[tl["animal"]] = c.get(tl["animal"], 0) + 1
        return c

    def plant_choice(self, day, hour):
        if hour > 21: return None
        q = self.plant_quota
        for crop in ("STRAWBERRY", "CARROT", "WHEAT"):
            if q.get(crop, 0) > 0: return crop
        return None

    # ---------------- tile work -----------------
    def pending(self, tile, day, hour, inv, seeds):
        ops = []
        if tile is None:
            crop = self.plant_choice(day, hour)
            if crop and seeds.get(crop, 0) > 0: ops.append(["PLANT", crop])
            return ops
        if not isinstance(tile, dict): return ops
        if "animal" in tile:
            morning = hour < P["care_hour"] and inv.get("WHEAT", 0) > 0  # feed-first: nothing else until the herd is fed
            if not tile["fed_today"] and inv.get("WHEAT", 0) > 0: ops.append(["FEED"])
            if tile.get("yield_units", 0) > 0 and (not morning or tile["yield_units"] >= 4): ops.append(["HARVEST"])
            if not morning:
                if not tile["cared_today"]: ops.append(["CARE"])
                if tile.get("fertilizer_available") and self.want_fert: ops.append(["COLLECT_FERTILIZER"])
            return ops
        kind = tile.get("kind")
        if kind == "WEED":
            if day <= P["wheat_last_day"]: ops.append(["DIG"])
            return ops
        if kind in ("COOP", "PASTURE"):
            a = "GOOSE" if kind == "COOP" else None
            for an in (["GOOSE"] if kind == "COOP" else ["COW", "SHEEP"]):
                if inv.get(an, 0) > 0: ops.append(["PLACE", an]); break
            return ops
        if kind == "PLANT":
            cd = CROPS[tile["crop"]]; age = day - tile["planted_day"]; yu = tile.get("yield_units", 0)
            ripe = yu > 0 and (cd["ongoing"] or age >= cd["maxd"] or day >= 29 or (age >= cd["first"] and yu >= cd.get("maxy", 99)))
            if ripe: ops.append(["HARVEST"])
            if not tile["watered_today"]:
                survive = tile.get("consecutive_unwatered", 0) >= 1
                pays = (tile.get("fertilized_until_day", -1) >= day) if cd["ongoing"] else ((cd["maxd"] + 1) // 2 <= age <= cd["maxd"])
                if survive or pays or P["water_all"]: ops.append(["WATER"])
            if inv.get("FERTILIZER", 0) > 0 and tile.get("fertilized_until_day", -1) < day and day <= 27:
                ws = (cd["maxd"] + 1) // 2
                in_window = (not cd["ongoing"]) and ws - 1 <= age <= cd["maxd"] - 1
                if (cd["ongoing"] and age >= cd["first"] - 2) or in_window: ops.append(["FERTILIZE"])
        return ops

    def build_zones(self, tiles, units):
        K = max(1, len(units) - 1)
        animals = sorted(((x, y) for y, row in enumerate(tiles) for x, tl in enumerate(row) if isinstance(tl, dict) and ("animal" in tl or tl.get("kind") in ("COOP", "PASTURE"))), key=lambda t: (t[0] // 3, t[1], t[0]))
        aset = set(animals)
        crops = [(x, y) for y, row in enumerate(tiles) for x, tl in enumerate(row) if isinstance(tl, dict) and tl.get("kind") in ("PLANT", "WEED") and (x, y) not in aset]
        empties = [(x, y) for y, row in enumerate(tiles) for x, tl in enumerate(row) if tl is None]
        need = sum(max(0, v) for v in self.plant_quota.values())
        empties.sort(key=lambda t: dist(t, (4, 4)))
        plants = crops + empties[:need]
        M = min(K, max(1, math.ceil(len(animals) / P["herd"]))) if animals else 0
        zones = {0: []}; self.herders = set(range(1, 1 + M))
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
        self.routes = {i: sorted(z, key=lambda t: (t[1], t[0] if t[1] % 2 == 0 else -t[0])) for i, z in zones.items()}
        self.idx = {i: 0 for i in zones}

    # ---------------- main -----------------
    def act(self, obs):
        step, seat = int(obs["step"]), int(obs["player"]); day, hour = step // 24, step % 24
        farm = obs["farms"][seat]; opp = obs["farms"][1 - seat]; private = obs["private"]; tiles = farm["tiles"]
        prices = obs["market"]["prices"]; minv = obs["market"].get("inventory", {}); shed = private["shed"]; seeds = private["seeds"]
        shops = obs["town"]["unlocked_shops"]
        units = [tuple(farm["farmer"]), *[tuple(h) for h in farm["hands"]]]
        invs = private["inventories"]; inv = lambda i: invs[i] if i < len(invs) else {}
        money = float(farm["money"]); market = []
        if self.day_planned != day:
            self.plan_day(obs, day, farm, opp, shops, money, tiles); self.day_planned = day
            own = self.plan["own"]
            planters = self.planters(tiles)
            cap = P["tiles_per_planter"] * planters
            alive = sum(v for k, v in own.items() if k in CROPS)
            room = max(0, cap - alive)
            q = {"STRAWBERRY": max(0, self.plan["straw"] - own.get("STRAWBERRY", 0)), "CARROT": max(0, self.plan["carrot"] - own.get("CARROT", 0))}
            q["WHEAT"] = max(0, min(self.plan["wheat"] - own.get("WHEAT", 0), room - q["STRAWBERRY"] - q["CARROT"]))
            self.plant_quota = q
        plan = self.plan
        self.want_fert = shed.get("FERTILIZER", 0) < P["fert_reserve"] or prices.get("FERTILIZER", 0) >= 25
        n_animals = sum(1 for row in tiles for tl in row if isinstance(tl, dict) and "animal" in tl)
        if self.day_built != day and hour >= 1:
            self.build_zones(tiles, units); self.day_built = day
        acts = [["PASS"] for _ in units]; claimed = set()
        # dawn loading at the shed
        if hour == 1:
            left = shed.get("WHEAT", 0); fert = max(0, shed.get("FERTILIZER", 0))
            for i in range(1, len(units)):
                if units[i] not in SHED: continue
                if i in self.herders:
                    na = sum(1 for (x, y) in self.routes.get(i, []) if isinstance(tiles[y][x], dict) and "animal" in tiles[y][x])
                    if na and left > 0: q = min(na + 1, left); acts[i] = ["PICKUP", "WHEAT", q]; left -= q
                else:
                    nf = 0
                    for (x, y) in self.routes.get(i, []):
                        tl = tiles[y][x]
                        if isinstance(tl, dict) and tl.get("kind") == "PLANT" and tl.get("fertilized_until_day", -1) <= day:
                            cd = CROPS[tl["crop"]]; age = day - tl["planted_day"]
                            if cd["ongoing"] or (cd["maxd"] + 1) // 2 - 1 <= age <= cd["maxd"] - 1: nf += 1
                    if nf and fert > 0: q = min(nf, P["fert_load"], fert); acts[i] = ["PICKUP", "FERTILIZER", q]; fert -= q
        # animals waiting in the shed: farmer (or first herder) picks them up and places them
        waiting = [(a, shed.get(a, 0)) for a in ANIMALS if shed.get(a, 0) > 0]
        if waiting and units[0] in SHED and not any(inv(0).get(a, 0) for a in ANIMALS):
            a, n = waiting[0]; acts[0] = ["PICKUP", a, 1]
        carried_animal = next((a for a in ANIMALS if inv(0).get(a, 0) > 0), None)
        if carried_animal and acts[0] == ["PASS"]:
            struct = ANIMALS[carried_animal]["struct"]
            free = [(x, y) for y, row in enumerate(tiles) for x, tl in enumerate(row) if isinstance(tl, dict) and tl.get("kind") == struct and "animal" not in tl]
            if free:
                tgt = min(free, key=lambda t: dist(units[0], t)); mv = step_toward(units[0], tgt)
                acts[0] = mv or ["PLACE", carried_animal]; claimed.add(tgt)
            else:
                empt = [(x, y) for y, row in enumerate(tiles) for x, tl in enumerate(row) if tl is None and (x, y) not in self.reserved]
                if empt:
                    anchor = [(x, y) for y, row in enumerate(tiles) for x, tl in enumerate(row) if isinstance(tl, dict) and ("animal" in tl or tl.get("kind") in ("COOP", "PASTURE"))]
                    tgt = min(empt, key=lambda t: (min([dist(t, a) for a in anchor] or [0]) + dist(units[0], t)))
                    mv = step_toward(units[0], tgt); acts[0] = mv or ["BUILD_" + struct]; claimed.add(tgt)
        # patrol: hands
        for i in range(1, len(units)):
            if acts[i] != ["PASS"]: continue
            pos = units[i]; route = self.routes.get(i, [])
            held = sum(v for k, v in inv(i).items() if k not in ("WHEAT", "FERTILIZER") and k not in ANIMALS and v > 0)
            nearest = min(SHED, key=lambda s_: dist(pos, s_)); near = dist(pos, nearest)
            if (held >= P["drop_held"] and near <= 2) or (held > 0 and hour >= 21 and near <= 2) or (day >= 29 and held > 0 and hour >= 24 - 2 - near) or held >= 12 or (step >= 716 and held > 0):
                acts[i] = (None if pos in SHED else step_toward(pos, nearest)) or ["DROP"]; continue
            here = self.pending(tiles[pos[1]][pos[0]], day, hour, inv(i), seeds) if pos not in claimed else []
            if here and i in self.herders and hour < P["care_hour"] and inv(i).get("WHEAT", 0) > 0 and "animal" not in (tiles[pos[1]][pos[0]] or {}): here = []
            if here: acts[i] = here[0]; claimed.add(pos); self.note_plant(here[0]); continue
            n = len(route); tgt = None
            if i not in self.herders:
                urgent = [t for t in route if t not in claimed and isinstance(tiles[t[1]][t[0]], dict) and tiles[t[1]][t[0]].get("kind") == "PLANT"
                          and not tiles[t[1]][t[0]]["watered_today"] and tiles[t[1]][t[0]].get("consecutive_unwatered", 0) >= 1]
                if urgent: tgt = min(urgent, key=lambda t: dist(pos, t))
            for k in range(n if tgt is None else 0):
                j = (self.idx[i] + k) % n; t = route[j]
                if t in claimed: continue
                if self.pending(tiles[t[1]][t[0]], day, hour, inv(i), seeds): tgt = t; self.idx[i] = j; break
            if tgt is None:
                tgt = self.nearest_pending(tiles, pos, claimed, day, hour, inv(i), seeds)
                if tgt is not None and dist(pos, tgt) > P["wander"]: tgt = None
            if tgt is None: continue
            claimed.add(tgt); mv = step_toward(pos, tgt)
            if mv: acts[i] = mv
            else:
                op = self.pending(tiles[tgt[1]][tgt[0]], day, hour, inv(i), seeds)[0]; acts[i] = op; self.note_plant(op)
        # farmer: free agent
        if acts[0] == ["PASS"]:
            pos = units[0]
            here = self.pending(tiles[pos[1]][pos[0]], day, hour, inv(0), seeds) if pos not in claimed else []
            if here: acts[0] = here[0]; self.note_plant(here[0])
            else:
                tgt = self.nearest_pending(tiles, pos, claimed, day, hour, inv(0), seeds)
                if tgt:
                    mv = step_toward(pos, tgt)
                    if mv: acts[0] = mv
                    else: op = self.pending(tiles[tgt[1]][tgt[0]], day, hour, inv(0), seeds)[0]; acts[0] = op; self.note_plant(op)
        # ---------------- market
        if hour == 0: market += [["HIRE"] for _ in range(min(plan["hands"], 8))]
        elif hour == 1 and plan["hands"] > 8 and len(units) - 1 < plan["hands"]: market += [["HIRE"] for _ in range(plan["hands"] - (len(units) - 1))]
        lp = self.land_price(day, len(farm.get("unlocked_quadrants", [])))
        if lp and money >= lp and hour >= 2: market.append(["BUY_LAND"]); money -= lp
        want = plan.get("want", {}); own = self.count_tiles(tiles)
        for a, n in want.items():
            have = own.get(a, 0) + shed.get(a, 0) + sum(inv(i).get(a, 0) for i in range(len(units)))
            if have < n and money >= ANIMALS[a]["cost"] + 60 and hour >= 2 and not waiting:
                market.append(["BUY_ANIMAL", a, 1]); money -= ANIMALS[a]["cost"]; break
        if hour in (0, 12):
            planters = self.planters(tiles); today_cap = 4 * planters
            for crop in ("STRAWBERRY", "CARROT", "WHEAT"):
                need = min(self.plant_quota.get(crop, 0), today_cap) - seeds.get(crop, 0)
                afford = int(max(0, money - 60) // CROPS[crop]["seed"])
                need = min(need, afford)
                if need > 0:
                    market.append(["BUY_SEED", crop, int(need)]); money -= CROPS[crop]["seed"] * need
        carried_w = sum(inv(i).get("WHEAT", 0) for i in range(len(units)))
        need_w = n_animals + 3 - shed.get("WHEAT", 0) - carried_w
        if need_w > 0 and hour in (0, 23) and day < 29: market.append(["BUY_PRODUCT", "WHEAT", int(min(need_w, 30))])
        final = step >= 716
        total = sum(max(0, v) for k, v in shed.items() if k not in ANIMALS)
        carried = sum(v for i in range(len(units)) for k, v in inv(i).items() if k not in ANIMALS and v > 0)
        if hour >= 20: total += carried
        opp_ready = self.rival_ready(opp["tiles"])
        sells = []
        for item in ("WOOL", "MILK", "STRAWBERRY", "MELON", "EGG", "TOMATO", "CARROT", "FERTILIZER", "WHEAT"):
            q = shed.get(item, 0)
            if item == "WHEAT" and not final: q -= n_animals + 3
            if item == "FERTILIZER" and not final: q -= P["fert_reserve"] if day <= 26 else 0
            if q <= 0: continue
            px = prices.get(item, 0)
            if px < 2 and not (final or total > P["shed_hi"]): continue  # floor price: only to make room
            if item in PREMIUM and P["hold_frac"] > 0 and px < P["hold_frac"] * BASE[item] and not (final or total > P["shed_hi"] or opp_ready.get(item, 0) >= 3):
                continue
            sells.append([item, q])
        sells.sort(key=lambda s: -prices[s[0]] * s[1])
        for item, q in sells:
            if len(market) >= 10: break
            market.append(["SELL", item, int(q)])
        return {"farmer": acts[0], "hands": acts[1:], "market": market[:10]}

    @staticmethod
    def land_price(day, quads):
        if quads == 1 and P["land_ne_day"] <= day <= P["land_last_day"]: return LAND_PRICES[0]
        if quads == 2 and P["land_sw_day"] <= day <= P["land_last_day"]: return LAND_PRICES[1]
        return None

    def planters(self, tiles):
        n_animals = sum(1 for row in tiles for tl in row if isinstance(tl, dict) and "animal" in tl)
        herders = math.ceil(n_animals / P["herd"]) if n_animals else 0
        return max(1, self.plan.get("hands", 8) - herders)

    def note_plant(self, op):
        if op and op[0] == "PLANT": self.plant_quota[op[1]] = self.plant_quota.get(op[1], 0) - 1

    def nearest_pending(self, tiles, pos, claimed, day, hour, inv, seeds):
        best = None
        for y, row in enumerate(tiles):
            for x, tl in enumerate(row):
                if (x, y) in claimed or tl == "LOCKED": continue
                if self.pending(tl, day, hour, inv, seeds) and (best is None or dist(pos, (x, y)) < best[0]): best = (dist(pos, (x, y)), (x, y))
        return best[1] if best else None

    @staticmethod
    def rival_ready(tiles):
        r = {}
        for row in tiles:
            for tl in row:
                if not isinstance(tl, dict): continue
                yu = int(tl.get("yield_units", 0) or 0)
                if yu <= 0: continue
                if "animal" in tl: p = ANIMALS[tl["animal"]]["product"]
                elif tl.get("kind") == "PLANT": p = tl["crop"]
                else: continue
                r[p] = r.get(p, 0) + yu
        return r
