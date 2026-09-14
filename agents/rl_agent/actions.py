"""Action dict <-> factorized targets (per-unit op + quantity, market quantity heads), and back."""
from __future__ import annotations
import numpy as np
from features import (OPS, OP_INDEX, N_OPS, ITEMS, PRODUCTS, CROPS, ANIMALS, MAX_UNITS, QTY_BUCKETS, MKT_BUCKETS, bucket, unbucket, BASE_PRICE)

BUY_PRODUCTS = ["WHEAT", "FERTILIZER"]
N_MKT = len(PRODUCTS) + len(BUY_PRODUCTS) + len(CROPS) + len(ANIMALS) + 2  # sell(9) buy_product(2) buy_seed(5) buy_animal(3) hire land
MKT_NAMES = ["SELL_" + p for p in PRODUCTS] + ["BUYP_" + p for p in BUY_PRODUCTS] + ["SEED_" + c for c in CROPS] + ["ANIMAL_" + a for a in ANIMALS] + ["HIRE", "LAND"]
MKT_CLASSES = [len(MKT_BUCKETS)] * (len(PRODUCTS) + len(BUY_PRODUCTS) + len(CROPS)) + [5] * len(ANIMALS) + [13, 2]


def unit_op(a):
    if not a: return "PASS", 1
    op = a[0]
    if op == "PLANT" and len(a) > 1 and a[1] in CROPS: return "PLANT_" + a[1], 1
    if op == "PICKUP" and len(a) > 1 and a[1] in ITEMS: return "PICKUP_" + a[1], (int(a[2]) if len(a) > 2 else 1)
    if op == "PLACE" and len(a) > 1 and a[1] in ITEMS: return "PLACE_" + a[1], (int(a[2]) if len(a) > 2 else 1)
    if op in OP_INDEX: return op, 1
    return "PASS", 1


def encode_action(action, n_units):
    """-> (op[MAX_UNITS] int8 with -1 for absent, qty[MAX_UNITS] int8, mkt[N_MKT] int8)."""
    action = action or {}
    acts = [action.get("farmer") or ["PASS"], *(action.get("hands") or [])]
    op = np.full(MAX_UNITS, -1, dtype=np.int8); qty = np.zeros(MAX_UNITS, dtype=np.int8)
    for i in range(min(n_units, MAX_UNITS)):
        a = acts[i] if i < len(acts) else ["PASS"]
        name, q = unit_op(a); op[i] = OP_INDEX[name]; qty[i] = bucket(q, QTY_BUCKETS)
    sums = {}
    hires = 0; land = 0
    for o in action.get("market") or []:
        if not o: continue
        k = o[0]
        if k == "SELL" and len(o) > 2 and o[1] in PRODUCTS: sums["SELL_" + o[1]] = sums.get("SELL_" + o[1], 0) + int(o[2])
        elif k == "BUY_PRODUCT" and len(o) > 2 and o[1] in BUY_PRODUCTS: sums["BUYP_" + o[1]] = sums.get("BUYP_" + o[1], 0) + int(o[2])
        elif k == "BUY_SEED" and len(o) > 2 and o[1] in CROPS: sums["SEED_" + o[1]] = sums.get("SEED_" + o[1], 0) + int(o[2])
        elif k == "BUY_ANIMAL" and len(o) > 2 and o[1] in ANIMALS: sums["ANIMAL_" + o[1]] = sums.get("ANIMAL_" + o[1], 0) + int(o[2])
        elif k == "HIRE": hires += 1
        elif k == "BUY_LAND": land = 1
    mkt = np.zeros(N_MKT, dtype=np.int8)
    for j, name in enumerate(MKT_NAMES):
        if name.startswith("ANIMAL_"): mkt[j] = min(4, sums.get(name, 0))
        elif name == "HIRE": mkt[j] = min(12, hires)
        elif name == "LAND": mkt[j] = land
        else: mkt[j] = bucket(min(sums.get(name, 0), MKT_BUCKETS[-1]), MKT_BUCKETS)
    return op, qty, mkt


def decode_action(op, qty, mkt, obs, seat):
    """Factorized predictions -> engine action dict. SELLs first (by value), then buys, hires, land; capped at 10 orders."""
    own = obs["farms"][seat]; n = min(MAX_UNITS, 1 + len(own["hands"])); prices = obs["market"]["prices"]
    units = []
    for i in range(n):
        name = OPS[int(op[i])]; q = unbucket(qty[i], QTY_BUCKETS)
        if name.startswith("PLANT_"): units.append(["PLANT", name[6:]])
        elif name.startswith("PICKUP_"): units.append(["PICKUP", name[7:], int(q)])
        elif name.startswith("PLACE_"): units.append(["PLACE", name[6:], int(q)] if name[6:] in PRODUCTS else ["PLACE", name[6:]])
        else: units.append([name])
    orders = []
    sells = []
    for j, p in enumerate(PRODUCTS):
        q = unbucket(mkt[j], MKT_BUCKETS)
        if q > 0: sells.append((prices.get(p, 0) * q, ["SELL", p, int(q)]))
    orders += [o for _, o in sorted(sells, key=lambda t: -t[0])]
    base = len(PRODUCTS)
    for j, p in enumerate(BUY_PRODUCTS):
        q = unbucket(mkt[base + j], MKT_BUCKETS)
        if q > 0: orders.append(["BUY_PRODUCT", p, int(q)])
    base += len(BUY_PRODUCTS)
    for j, c in enumerate(CROPS):
        q = unbucket(mkt[base + j], MKT_BUCKETS)
        if q > 0: orders.append(["BUY_SEED", c, int(q)])
    base += len(CROPS)
    for j, a in enumerate(ANIMALS):
        q = int(mkt[base + j])
        if q > 0: orders.append(["BUY_ANIMAL", a, q])
    base += len(ANIMALS)
    orders += [["HIRE"]] * int(mkt[base])
    if int(mkt[base + 1]): orders.append(["BUY_LAND"])
    return {"farmer": units[0] if units else ["PASS"], "hands": units[1:], "market": orders[:10]}
