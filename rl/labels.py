"""Destination labels: for each (step, unit) the tile where this unit next performs a non-move action within the day, and that action."""
from __future__ import annotations
import numpy as np
from features import MAX_UNITS, OP_INDEX
from actions import unit_op, encode_action
MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}


def destination_labels(steps, seat, T=719):
    """Returns dest [T, U] int8 (tile index y*10+x, -1 absent/unknown), dop [T, U] int8 (op class at destination), dqty [T,U] int8."""
    from features import QTY_BUCKETS, bucket
    dest = np.full((T, MAX_UNITS), -1, dtype=np.int8); dop = np.full((T, MAX_UNITS), -1, dtype=np.int8); dqty = np.zeros((T, MAX_UNITS), dtype=np.int8)
    for t in range(T):
        obs = steps[t][seat]["observation"]; farm = obs["farms"][seat]; units = [farm["farmer"], *farm["hands"]][:MAX_UNITS]
        day_end = (t // 24 + 1) * 24
        for i, pos in enumerate(units):
            found = None; now = "PASS"
            for t2 in range(t, min(day_end, T)):
                a2 = steps[t2 + 1][seat]["action"] or {}; acts = [a2.get("farmer") or ["PASS"], *(a2.get("hands") or [])]
                if i >= len(acts): break
                op = acts[i][0] if acts[i] else "PASS"
                if t2 == t: now = op
                farm2 = steps[t2][seat]["observation"]["farms"][seat]; u2 = [farm2["farmer"], *farm2["hands"]]
                if i >= len(u2): break
                if op not in MOVES and op != "PASS":
                    name, q = unit_op(acts[i]); found = (int(u2[i][0]), int(u2[i][1]), name, q); break
            if found is None:
                # 当日中に作業が無い。実際に待機していたなら PASS、移動中なら「日跨ぎで意図が切れた」だけなので
                # -1 (= bc_loss3 の present から外れる) にする。ここを PASS にすると夕方の移動が全部 PASS 教師になる。
                if now not in MOVES:
                    dest[t, i] = int(pos[1]) * 10 + int(pos[0]); dop[t, i] = OP_INDEX["PASS"]
            else:
                x, y, name, q = found; dest[t, i] = y * 10 + x; dop[t, i] = OP_INDEX[name]; dqty[t, i] = bucket(q, QTY_BUCKETS)
    return dest, dop, dqty
