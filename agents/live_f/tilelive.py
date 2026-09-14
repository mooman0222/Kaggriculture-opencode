"""Tile-live layer: keep the tape's unit movements (proven circuits), decide the work on each tile live.

For every unit whose tape action is not a move / logistics op, look at the tile it stands on and pick:
harvest if ripe (also one-time crops at full yield), water if it saves or pays, fertilize when it doubles
the next production, plant the macro's crop instead of the tape's when demand says so (seeds bought at
dawn for the planned substitutions), otherwise keep the tape's action.
"""
from __future__ import annotations
import os

CROPS = {"WHEAT": dict(seed=10, first=2, maxd=4, ongoing=False, maxy=6), "CARROT": dict(seed=20, first=2, maxd=3, ongoing=False, maxy=4),
         "TOMATO": dict(seed=50, first=8, maxd=8, ongoing=True), "STRAWBERRY": dict(seed=100, first=10, maxd=10, ongoing=True),
         "MELON": dict(seed=80, first=10, maxd=12, ongoing=False, maxy=6)}
SHOPS = {"BAKERY": ["EGG", "WHEAT"], "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"], "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"], "YARN_STORE": ["WOOL"],
         "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"], "PET_CAFE": ["CARROT"], "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"], "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"]}
MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}
KEEP = MOVES | {"PICKUP", "DROP", "PLACE", "BUILD_COOP", "BUILD_PASTURE", "FEED", "CARE", "COLLECT_FERTILIZER", "DIG"}
P = {"straw_per_shop": int(os.environ.get("LF_STRAW_PER", "5")), "straw_base": int(os.environ.get("LF_STRAW", "4")), "straw_cap": int(os.environ.get("LF_STRAW_CAP", "30")),
     "straw_last_day": int(os.environ.get("LF_STRAW_LAST", "15")), "carrot_per_shop": int(os.environ.get("LF_CARROT_PER", "5")), "carrot_from": 8, "carrot_last": 26,
     "wheat_min": int(os.environ.get("LF_WHEAT_MIN", "20")), "harvest_live": os.environ.get("LF_HARV", "1") == "1", "water_live": os.environ.get("LF_WATER", "1") == "1",
     "fert_live": os.environ.get("LF_FERT", "1") == "1", "plant_live": os.environ.get("LF_PLANT", "1") == "1", "match_straw": int(os.environ.get("LF_MATCH_STRAW", "1"))}


def cnt(shops, p): return sum((2 if len(SHOPS[s]) == 1 else 1) for s in shops if p in SHOPS[s])


def tiles_by_crop(tiles):
    c = {}
    for row in tiles:
        for tl in row:
            if isinstance(tl, dict) and tl.get("kind") == "PLANT": c[tl["crop"]] = c.get(tl["crop"], 0) + 1
    return c


class TileLive:
    def __init__(self, base):
        self.b = base; self.day = -1; self.quota = {}; self.subs_today = 0

    def tape_action(self, seat, step):
        ch = self.b._IMPL.chassis; st = ch.players.get(seat) or {}
        route = st.get("route"); route = route if route in ch.routes else 0
        tape = ch.routes[route]
        return tape[step] if 0 <= step < len(tape) and isinstance(tape[step], dict) else {}

    def plan_day(self, obs, seat, day):
        farm = obs["farms"][seat]; opp = obs["farms"][1 - seat]; shops = obs["town"]["unlocked_shops"]
        own = tiles_by_crop(farm["tiles"]); oth = tiles_by_crop(opp["tiles"])
        straw = min(P["straw_cap"], P["straw_base"] + P["straw_per_shop"] * cnt(shops, "STRAWBERRY"))
        if P["match_straw"]: straw = max(straw, oth.get("STRAWBERRY", 0) + 2)
        q = {"STRAWBERRY": max(0, straw - own.get("STRAWBERRY", 0)) if day <= P["straw_last_day"] else 0,
             "CARROT": max(0, P["carrot_per_shop"] * cnt(shops, "CARROT") - own.get("CARROT", 0)) if P["carrot_from"] <= day <= P["carrot_last"] else 0}
        self.quota = q; self.wheat_ok = own.get("WHEAT", 0) > P["wheat_min"]

    def act(self, act, obs, step):
        seat = int(obs["player"]); day, hour = step // 24, step % 24
        farm = obs["farms"][seat]; tiles = farm["tiles"]; private = obs["private"]; seeds = dict(private["seeds"])
        invs = private["inventories"]
        if day != self.day: self.plan_day(obs, seat, day); self.day = day
        units = [farm["farmer"], *farm["hands"]]
        acts = [act.get("farmer") or ["PASS"], *(act.get("hands") or [])]
        while len(acts) < len(units): acts.append(["PASS"])
        tape = self.tape_action(seat, step)
        tacts = [tape.get("farmer") or ["PASS"], *(tape.get("hands") or [])]
        planted = {}
        own = tiles_by_crop(tiles)
        for i, pos in enumerate(units):
            a = acts[i]; op = a[0] if a else "PASS"
            if op in KEEP: continue
            x, y = int(pos[0]), int(pos[1]); tile = tiles[y][x]; inv = invs[i] if i < len(invs) else {}
            if tile is None:
                if op == "PLANT" and P["plant_live"] and len(a) > 1 and a[1] == "WHEAT" and self.wheat_ok:
                    for crop in ("STRAWBERRY", "CARROT"):
                        if self.quota.get(crop, 0) > 0 and seeds.get(crop, 0) - planted.get(crop, 0) > 0:
                            acts[i] = ["PLANT", crop]; planted[crop] = planted.get(crop, 0) + 1; self.quota[crop] -= 1; own["WHEAT"] = own.get("WHEAT", 0); break
                continue
            if not isinstance(tile, dict) or tile.get("kind") != "PLANT": continue
            cd = CROPS[tile["crop"]]; age = day - tile["planted_day"]; yu = int(tile.get("yield_units", 0) or 0)
            harvestable = yu > 0 and age >= cd["first"]
            ripe = harvestable and (cd["ongoing"] or age >= cd["maxd"] or day >= 29 or yu >= cd.get("maxy", 99))
            if P["harvest_live"] and ripe and not cd["ongoing"] and op in ("PASS", "WATER") and yu >= cd.get("maxy", 99): acts[i] = ["HARVEST"]; continue
            if op == "HARVEST" and not harvestable: op = "PASS"  # tape expected a different crop here: do useful work instead
            if op == "PLANT": op = "PASS"  # occupied tile (crop substituted earlier)
            if op in ("PASS", "WATER", "FERTILIZE"):
                need_water = not tile["watered_today"] and (tile.get("consecutive_unwatered", 0) >= 1 or
                             ((tile.get("fertilized_until_day", -1) >= day) if cd["ongoing"] else ((cd["maxd"] + 1) // 2 <= age <= cd["maxd"])))
                ws = (cd["maxd"] + 1) // 2
                can_fert = P["fert_live"] and inv.get("FERTILIZER", 0) > 0 and tile.get("fertilized_until_day", -1) < day and day <= 27 and \
                    ((cd["ongoing"] and age >= cd["first"] - 2) or ((not cd["ongoing"]) and ws - 1 <= age <= cd["maxd"] - 1))
                if op == "WATER" and tile["watered_today"]: op = "PASS"
                if op == "PASS" or (op == "WATER" and not P["water_live"]):
                    if need_water: acts[i] = ["WATER"]
                    elif can_fert: acts[i] = ["FERTILIZE"]
                    elif op == "PASS": acts[i] = ["PASS"]
                elif op == "WATER":
                    pass  # keep the tape's watering as is
        act["farmer"], act["hands"] = acts[0], acts[1:]
        # dawn seeds for today's substitutions: count tape PLANT WHEAT for the day
        if hour == 0 and P["plant_live"]:
            ch = self.b._IMPL.chassis; st = ch.players.get(seat) or {}; route = st.get("route") if (st.get("route") in ch.routes) else 0
            n = 0
            for t in range(step, min(step + 24, 719)):
                ta = ch.routes[route][t] if isinstance(ch.routes[route][t], dict) else {}
                for u in [ta.get("farmer") or [], *(ta.get("hands") or [])]:
                    if u and u[0] == "PLANT" and len(u) > 1 and u[1] == "WHEAT": n += 1
            money = float(farm["money"]) - 300
            for crop in ("STRAWBERRY", "CARROT"):
                want = min(self.quota.get(crop, 0), n) - seeds.get(crop, 0)
                afford = int(max(0, money) // CROPS[crop]["seed"]); want = min(want, afford)
                if want > 0 and len(act["market"]) < 10:
                    act["market"] = list(act["market"]) + [["BUY_SEED", crop, int(want)]]; money -= CROPS[crop]["seed"] * want; n -= want
        return act
