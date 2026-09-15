"""Observation -> compact integer tensors (tiles / units / items / global) and per-unit legality masks.

Everything here is numpy only so the same code runs inside the Kaggle agent (no torch at inference time).
"""
from __future__ import annotations
import numpy as np

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
ITEMS = PRODUCTS + ANIMALS  # 12 inventory items
CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
SHOPS = ["BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE", "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET"]
SHOP_PRODUCTS = {"BAKERY": ["EGG", "WHEAT"], "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"], "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"], "YARN_STORE": ["WOOL"],
                 "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"], "PET_CAFE": ["CARROT"], "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"], "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"]}
BASE_PRICE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}
ANIMAL_PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
ANIMAL_STRUCT = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}
MAX_UNITS = 16
TILE_F, UNIT_F, ITEM_F, GLOB_F = 18, 4 + len(ITEMS) + 2, 9, 40
SHED_TILES = {(4, 4), (5, 4), (4, 5), (5, 5)}
LAND_PRICES = [1000, 2000, 4000]
FIRST_YIELD_DAY = {"WHEAT": 2, "CARROT": 2, "TOMATO": 8, "STRAWBERRY": 10, "MELON": 10}  # sim.hpp CROPS; HARVEST is a no-op before this age

# unit op classes
OPS = (["PASS", "NORTH", "SOUTH", "EAST", "WEST"] + ["PLANT_" + c for c in CROPS] + ["WATER", "HARVEST", "FERTILIZE", "DIG", "BUILD_COOP", "BUILD_PASTURE"]
       + ["PICKUP_" + i for i in ITEMS] + ["DROP"] + ["PLACE_" + i for i in ITEMS] + ["FEED", "CARE", "COLLECT_FERTILIZER"])
OP_INDEX = {o: i for i, o in enumerate(OPS)}
N_OPS = len(OPS)
QTY_BUCKETS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50]
MKT_BUCKETS = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]


def bucket(q, buckets):
    q = int(q)
    return int(np.argmin([abs(q - b) for b in buckets]))


def unbucket(k, buckets): return buckets[int(k)]


def _tile_row(tl, x, y, day, step, opp, units_here):
    f = np.zeros(TILE_F, dtype=np.int16); f[16] = x; f[17] = y; f[15] = opp; f[14] = units_here
    if tl == "LOCKED": f[0] = 0; return f
    if tl is None: f[0] = 1; return f
    if not isinstance(tl, dict): f[0] = 1; return f
    kind = tl.get("kind")
    if "animal" in tl:
        f[0] = 6; f[2] = ANIMALS.index(tl["animal"]) + 1 if tl["animal"] in ANIMALS else 0
        f[3] = min(30, max(0, day - int(tl.get("placed_day", day)))); f[4] = min(6, int(tl.get("yield_units", 0) or 0))
        f[8] = int(bool(tl.get("fed_today"))); f[9] = int(bool(tl.get("cared_today"))); f[10] = min(3, int(tl.get("consecutive_unfed", 0) or 0))
        f[11] = int(bool(tl.get("fertilizer_available"))); f[12] = min(3, int(tl.get("pending_care_bonus", 0) or 0)); return f
    if kind == "WEED": f[0] = 2; return f
    if kind == "COOP": f[0] = 4; return f
    if kind == "PASTURE": f[0] = 5; return f
    if kind == "PLANT":
        f[0] = 3; f[1] = CROPS.index(tl["crop"]) + 1 if tl.get("crop") in CROPS else 0
        f[3] = min(30, max(0, day - int(tl.get("planted_day", day)))); f[4] = min(6, int(tl.get("yield_units", 0) or 0))
        f[5] = int(bool(tl.get("watered_today"))); f[6] = min(3, int(tl.get("consecutive_unwatered", 0) or 0))
        f[7] = max(0, min(3, int(tl.get("fertilized_until_day", -1)) - day + 1))
        mls = int(tl.get("max_lifespan_step", -1)); f[13] = 8 if mls < 0 else max(0, min(8, (mls - step) // 24)); return f
    f[0] = 1; return f


def encode(obs, seat):
    """Returns dict of int16/float32 arrays. seat = which farm is 'own'."""
    step = int(obs["step"]); day, hour = step // 24, step % 24
    farms = obs["farms"]; own, opp = farms[seat], farms[1 - seat]; private = obs["private"]
    units = [own["farmer"], *own["hands"]][:MAX_UNITS]
    here = {}
    for u in units: here[(int(u[0]), int(u[1]))] = here.get((int(u[0]), int(u[1])), 0) + 1
    tiles = np.zeros((2, 10, 10, TILE_F), dtype=np.int16)
    for k, farm in ((0, own), (1, opp)):
        for y, row in enumerate(farm["tiles"]):
            for x, tl in enumerate(row):
                tiles[k, y, x] = _tile_row(tl, x, y, day, step, k, here.get((x, y), 0) if k == 0 else 0)
    invs = private["inventories"]
    U = np.zeros((MAX_UNITS, UNIT_F), dtype=np.int16)
    for i, u in enumerate(units):
        inv = invs[i] if i < len(invs) else {}
        U[i, 0] = 1; U[i, 1] = int(i == 0); U[i, 2] = int(u[0]); U[i, 3] = int(u[1])
        for j, it in enumerate(ITEMS): U[i, 4 + j] = min(60, int(inv.get(it, 0) or 0))
        U[i, 4 + len(ITEMS)] = int((int(u[0]), int(u[1])) in SHED_TILES); U[i, 5 + len(ITEMS)] = i
    prices = obs["market"]["prices"]; minv = obs["market"].get("inventory", {}); shed = private["shed"]; shops = obs["town"]["unlocked_shops"]
    carried = {it: sum(int(inv.get(it, 0) or 0) for inv in invs) for it in ITEMS}
    def producing(farm, prod):
        n = 0
        for row in farm["tiles"]:
            for tl in row:
                if isinstance(tl, dict):
                    if tl.get("kind") == "PLANT" and tl.get("crop") == prod: n += 1
                    elif "animal" in tl and ANIMAL_PRODUCT.get(tl["animal"]) == prod: n += 1
        return n
    I = np.zeros((len(PRODUCTS), ITEM_F), dtype=np.float32)
    for j, p in enumerate(PRODUCTS):
        I[j] = [prices.get(p, 0) / BASE_PRICE[p], (minv.get(p, 10000) - 10000) / 100.0, min(200, shed.get(p, 0)) / 50.0, min(200, carried[p]) / 50.0,
                sum((2 if len(SHOP_PRODUCTS[s]) == 1 else 1) for s in shops if p in SHOP_PRODUCTS[s]) / 4.0, producing(own, p) / 20.0, producing(opp, p) / 20.0,
                BASE_PRICE[p] / 250.0, j / 8.0]
    quads = len(own.get("unlocked_quadrants", [])); oq = len(opp.get("unlocked_quadrants", []))
    G = np.zeros(GLOB_F, dtype=np.float32)
    G[0:6] = [day / 29.0, hour / 23.0, np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24), step / 719.0, float(day >= 29)]
    G[6:12] = [np.log1p(max(0.0, float(own["money"]))) / 12.0, np.log1p(max(0.0, float(opp["money"]))) / 12.0, quads / 4.0, oq / 4.0, len(own["hands"]) / 12.0, len(opp["hands"]) / 12.0]
    G[12] = int(own.get("hires_today", 0)) / 12.0
    for s in shops:
        if s in SHOPS: G[13 + SHOPS.index(s)] += 1 / 3.0
    G[21] = len(shops) / 8.0
    seeds = private["seeds"]
    for j, c in enumerate(CROPS): G[22 + j] = min(60, int(seeds.get(c, 0) or 0)) / 30.0
    for j, a in enumerate(ANIMALS): G[27 + j] = min(6, int(shed.get(a, 0) or 0)) / 3.0
    G[30] = (LAND_PRICES[quads - 1] / 4000.0) if 1 <= quads <= 3 else 0.0
    G[31] = sum(max(0, v) for k, v in shed.items() if k in PRODUCTS) / 100.0
    G[32] = seat
    return {"tiles": tiles, "units": U, "items": I, "glob": G}


def legal_ops(obs, seat):
    """[MAX_UNITS, N_OPS] bool mask of engine-accepted ops per own unit (present units only)."""
    own = obs["farms"][seat]; tiles = own["tiles"]; private = obs["private"]; invs = private["inventories"]; seeds = private["seeds"]; shed = private["shed"]
    units = [own["farmer"], *own["hands"]][:MAX_UNITS]
    M = np.zeros((MAX_UNITS, N_OPS), dtype=bool)
    for i, u in enumerate(units):
        x, y = int(u[0]), int(u[1]); tl = tiles[y][x]; inv = invs[i] if i < len(invs) else {}
        m = M[i]; m[OP_INDEX["PASS"]] = True
        m[OP_INDEX["NORTH"]] = y > 0; m[OP_INDEX["SOUTH"]] = y < 9; m[OP_INDEX["WEST"]] = x > 0; m[OP_INDEX["EAST"]] = x < 9
        at_shed = (x, y) in SHED_TILES
        if at_shed:
            for it in ITEMS:
                if int(shed.get(it, 0) or 0) > 0: m[OP_INDEX["PICKUP_" + it]] = True
            if any(int(v or 0) > 0 for v in inv.values()): m[OP_INDEX["DROP"]] = True
            for it in PRODUCTS:
                if int(inv.get(it, 0) or 0) > 0: m[OP_INDEX["PLACE_" + it]] = True
        if tl is None:
            for c in CROPS:
                if int(seeds.get(c, 0) or 0) > 0: m[OP_INDEX["PLANT_" + c]] = True
            m[OP_INDEX["BUILD_COOP"]] = True; m[OP_INDEX["BUILD_PASTURE"]] = True; m[OP_INDEX["DIG"]] = True
        elif isinstance(tl, dict):
            if "animal" in tl:
                if int(inv.get("WHEAT", 0) or 0) > 0: m[OP_INDEX["FEED"]] = True
                m[OP_INDEX["CARE"]] = True
                if int(tl.get("yield_units", 0) or 0) > 0: m[OP_INDEX["HARVEST"]] = True
                if tl.get("fertilizer_available"): m[OP_INDEX["COLLECT_FERTILIZER"]] = True
            else:
                kind = tl.get("kind")
                if kind == "WEED": m[OP_INDEX["DIG"]] = True
                elif kind == "PLANT":
                    m[OP_INDEX["WATER"]] = True; m[OP_INDEX["DIG"]] = True
                    if int(tl.get("yield_units", 0) or 0) > 0: m[OP_INDEX["HARVEST"]] = True
                    if int(inv.get("FERTILIZER", 0) or 0) > 0: m[OP_INDEX["FERTILIZE"]] = True
                elif kind in ("COOP", "PASTURE"):
                    m[OP_INDEX["DIG"]] = True
                    for a in ANIMALS:
                        if ANIMAL_STRUCT[a] == kind and int(inv.get(a, 0) or 0) > 0: m[OP_INDEX["PLACE_" + a]] = True
    return M


def legal_ops_at(obs, seat, unit_i, x, y):
    """Legality of ops for own unit `unit_i` if it were standing on tile (x, y) with its current inventory."""
    own = obs["farms"][seat]; tiles = own["tiles"]; private = obs["private"]; invs = private["inventories"]; seeds = private["seeds"]; shed = private["shed"]
    tl = tiles[y][x]; inv = invs[unit_i] if unit_i < len(invs) else {}
    m = np.zeros(N_OPS, dtype=bool); m[OP_INDEX["PASS"]] = True
    if (x, y) in SHED_TILES:
        for it in ITEMS:
            if int(shed.get(it, 0) or 0) > 0: m[OP_INDEX["PICKUP_" + it]] = True
        if any(int(v or 0) > 0 for v in inv.values()): m[OP_INDEX["DROP"]] = True
        for it in PRODUCTS:
            if int(inv.get(it, 0) or 0) > 0: m[OP_INDEX["PLACE_" + it]] = True
    if tl is None:
        for c in CROPS:
            if int(seeds.get(c, 0) or 0) > 0: m[OP_INDEX["PLANT_" + c]] = True
        m[OP_INDEX["BUILD_COOP"]] = True; m[OP_INDEX["BUILD_PASTURE"]] = True; m[OP_INDEX["DIG"]] = True
    elif isinstance(tl, dict):
        if "animal" in tl:
            if int(inv.get("WHEAT", 0) or 0) > 0 and not tl.get("fed_today"): m[OP_INDEX["FEED"]] = True
            if not tl.get("cared_today"): m[OP_INDEX["CARE"]] = True
            if int(tl.get("yield_units", 0) or 0) > 0: m[OP_INDEX["HARVEST"]] = True
            if tl.get("fertilizer_available"): m[OP_INDEX["COLLECT_FERTILIZER"]] = True
        else:
            kind = tl.get("kind")
            if kind == "WEED": m[OP_INDEX["DIG"]] = True
            elif kind == "PLANT":
                if not tl.get("watered_today"): m[OP_INDEX["WATER"]] = True
                m[OP_INDEX["DIG"]] = True
                day = int(obs["step"]) // 24
                if int(tl.get("yield_units", 0) or 0) > 0 and day - int(tl.get("planted_day", day)) >= FIRST_YIELD_DAY.get(tl.get("crop"), 0): m[OP_INDEX["HARVEST"]] = True
                if int(inv.get("FERTILIZER", 0) or 0) > 0: m[OP_INDEX["FERTILIZE"]] = True
            elif kind in ("COOP", "PASTURE"):
                m[OP_INDEX["DIG"]] = True
                for a in ANIMALS:
                    if ANIMAL_STRUCT[a] == kind and int(inv.get(a, 0) or 0) > 0: m[OP_INDEX["PLACE_" + a]] = True
    return m


def dest_mask(obs, seat):
    """[100] bool: tiles a unit may target (everything but LOCKED)."""
    tiles = obs["farms"][seat]["tiles"]
    return np.array([tiles[y][x] != "LOCKED" for y in range(10) for x in range(10)], dtype=bool)
