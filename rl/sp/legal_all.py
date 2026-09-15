"""Vectorised arrival legality: for every own unit and every tile, which ops the engine would accept (mirrors features.legal_ops_at exactly).
Input: encoded features (features.encode output). Output: bool [MAX_UNITS, 100, N_OPS]. Pure numpy, ~0.1 ms."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from features import (OPS, OP_INDEX, N_OPS, MAX_UNITS, ITEMS, PRODUCTS, CROPS, ANIMALS, ANIMAL_STRUCT, SHED_TILES, FIRST_YIELD_DAY)

SHED_IDX = np.array([y * 10 + x for (x, y) in sorted(SHED_TILES)])
OP = OP_INDEX
PICKUP = np.array([OP["PICKUP_" + it] for it in ITEMS]); PLACE_P = np.array([OP["PLACE_" + p] for p in PRODUCTS]); PLACE_A = np.array([OP["PLACE_" + a] for a in ANIMALS])
PLANT = np.array([OP["PLANT_" + c] for c in CROPS]); FIRST_YIELD = np.array([0] + [FIRST_YIELD_DAY[c] for c in CROPS])  # crop code 0 = none
INV_WHEAT = 4 + ITEMS.index("WHEAT"); INV_FERT = 4 + ITEMS.index("FERTILIZER"); INV_ANIM = np.array([4 + ITEMS.index(a) for a in ANIMALS])
STRUCT_KIND = np.array([4 if ANIMAL_STRUCT[a] == "COOP" else 5 for a in ANIMALS])  # tile kind codes: 4 COOP, 5 PASTURE


def legal_all(f):
    """f = features.encode(obs, seat). Returns bool [MAX_UNITS, 100, N_OPS]; rows for absent units are PASS-only."""
    T = f["tiles"][0].reshape(100, -1); U = f["units"]; G = f["glob"]; I = f["items"]
    kind = T[:, 0]; crop = T[:, 1]; age = T[:, 3]; yld = T[:, 4]; watered = T[:, 5] > 0; fed = T[:, 8] > 0; cared = T[:, 9] > 0; fert_avail = T[:, 11] > 0
    present = U[:, 0] > 0; inv = U[:, 4:4 + len(ITEMS)]
    m = np.zeros((MAX_UNITS, 100, N_OPS), dtype=bool); m[:, :, OP["PASS"]] = True
    # shed tiles: pickups from shed stock, drop if carrying anything, place products carried
    shed_stock = np.concatenate([I[:, 2] > 0, G[27:30] > 0])  # 9 products then 3 animals (ITEMS order)
    for j in np.where(shed_stock)[0]: m[:, SHED_IDX, PICKUP[j]] = True
    carrying = (inv > 0).any(1)
    m[carrying, :, OP["DROP"]] &= False; m[np.ix_(np.where(carrying)[0], SHED_IDX, [OP["DROP"]])] = True
    for j, p in enumerate(PRODUCTS):
        has = inv[:, j] > 0
        if has.any(): m[np.ix_(np.where(has)[0], SHED_IDX, [PLACE_P[j]])] = True
    # empty tiles
    empty = kind == 1
    for c in range(len(CROPS)):
        if G[22 + c] > 0: m[:, empty, PLANT[c]] = True
    m[:, empty, OP["BUILD_COOP"]] = True; m[:, empty, OP["BUILD_PASTURE"]] = True; m[:, empty, OP["DIG"]] = True
    # animals
    an = kind == 6
    has_wheat = inv[:, INV_WHEAT - 4] > 0
    m[np.ix_(np.where(has_wheat)[0], np.where(an & ~fed)[0], [OP["FEED"]])] = True
    m[:, an & ~cared, OP["CARE"]] = True; m[:, an & (yld > 0), OP["HARVEST"]] = True; m[:, an & fert_avail, OP["COLLECT_FERTILIZER"]] = True
    # weeds / plants
    m[:, kind == 2, OP["DIG"]] = True
    pl = kind == 3
    m[:, pl & ~watered, OP["WATER"]] = True; m[:, pl, OP["DIG"]] = True
    m[:, pl & (yld > 0) & (age >= FIRST_YIELD[np.clip(crop, 0, 5)]), OP["HARVEST"]] = True
    has_fert = inv[:, INV_FERT - 4] > 0
    m[np.ix_(np.where(has_fert)[0], np.where(pl)[0], [OP["FERTILIZE"]])] = True
    # structures
    st = (kind == 4) | (kind == 5)
    m[:, st, OP["DIG"]] = True
    for j in range(len(ANIMALS)):
        has = inv[:, INV_ANIM[j] - 4] > 0
        if has.any(): m[np.ix_(np.where(has)[0], np.where(kind == STRUCT_KIND[j])[0], [PLACE_A[j]])] = True
    m[~present] = False; m[~present, :, OP["PASS"]] = True
    return m


if __name__ == "__main__":  # parity check against features.legal_ops_at on real states
    import kagsim, time
    from features import encode, legal_ops_at
    from play2 import load_py
    opp = load_py("third_party/public_agents/v41/main.py"); bad = 0; n = 0; t_sum = 0.0
    for seed in (1, 2):
        for k in ("_LIVE", "_POLICY"):
            if hasattr(opp, k): setattr(opp, k, None)
        g = kagsim.Game(seed); rng = np.random.default_rng(seed)
        while not g.done:
            o0, o1 = g.observe(0), g.observe(1)
            if rng.random() < 0.05:
                for seat, o in ((0, o0), (1, o1)):
                    f = encode(o, seat); t = time.perf_counter(); M = legal_all(f); t_sum += time.perf_counter() - t
                    units = [o["farms"][seat]["farmer"], *o["farms"][seat]["hands"]][:MAX_UNITS]
                    for i in range(len(units)):
                        for ti in range(100):
                            ref = legal_ops_at(o, seat, i, ti % 10, ti // 10); n += 1
                            if not np.array_equal(ref, M[i, ti]):
                                bad += 1
                                if bad <= 3: print("MISMATCH step", o["step"], "unit", i, "tile", ti, [OPS[j] for j in np.where(ref != M[i, ti])[0]], "ref", ref[ref != M[i, ti]])
            g.step(opp.agent(o0), opp.agent(o1))
    print(f"checked {n} (unit,tile) pairs, mismatches {bad}, legal_all {t_sum / max(1, n / 100 / 1) * 1e6 / 100:.0f} us per call")
