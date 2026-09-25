"""Torch-free shared inference helpers (also runs inside the Kaggle agent).

step_toward / trim_plants moved from act2, legal_option_mask + market sell
rules moved from act3 — all torch-free so numpy-only deployments can reuse
them bit-exactly. Torch callers import from here (act2/act3 re-export).
"""
from __future__ import annotations

import numpy as np

from features import (
    CROPS,
    MAX_UNITS,
    MKT_BUCKETS,
    N_OPS,
    OP_INDEX,
    PRODUCTS,
    SHED_TILES,
    legal_ops_at,
)


# 学習時 _mkt_ce が非ゼロクラスを pos_w 倍に重み付けした分 (model3.bc_loss3 の各ヘッド)
MKT_POS_W = {"sell": 6.0, "buyp": 6.0, "seed": 6.0, "anim": 20.0, "hire": 8.0, "land": 30.0}


def mkt_argmax(logits, head, debias=()):
    """市場ヘッドの argmax。head が debias に入っていれば非ゼロクラスから log(pos_w) を引いて事前確率を戻す。"""
    if head not in debias:
        return logits.argmax(-1)
    adjusted = np.asarray(logits, dtype=np.float64).copy()
    adjusted[..., 1:] -= np.log(MKT_POS_W[head])
    return adjusted.argmax(-1)


MKT_VALUES = {"sell": MKT_BUCKETS, "buyp": MKT_BUCKETS, "seed": MKT_BUCKETS, "anim": list(range(5)), "hire": list(range(13)), "land": [0, 1]}


def mkt_decode(logits, head, mode="argmax", debias=(), state=None):
    """市場ヘッドの復号。argmax は 1 手ごとの最頻値で、局の総量を +20〜90% 歪める (教師の「たまに 1 個買う」を
    毎手の 0 か pos_w で膨らんだ非ゼロに潰すため)。sample / dither は pos_w を戻した較正確率を使い、局の総量の期待値を保つ:
    sample はクラスを抽選、dither は期待数量を手をまたいで繰り越し、整数部を出す (誤差拡散、決定的)。"""
    if mode == "argmax":
        return mkt_argmax(logits, head, debias)
    lg = np.asarray(logits, dtype=np.float64)
    shape = lg.shape[:-1]
    lg = lg.reshape(-1, lg.shape[-1]).copy()
    lg[:, 1:] -= np.log(MKT_POS_W[head])
    p = np.exp(lg - lg.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    state = {} if state is None else state
    if mode == "sample":
        rng = state.setdefault("mkt_rng", np.random.default_rng(0))
        idx = np.minimum((p.cumsum(-1) < rng.random((len(p), 1))).sum(-1), p.shape[-1] - 1)
    else:
        values = np.asarray(MKT_VALUES[head], dtype=np.float64)
        acc = state.setdefault("mkt_acc_" + head, np.zeros(len(p)))
        acc += p @ values
        idx = (values[None, :] <= acc[:, None] + 1e-9).sum(-1) - 1
        acc -= values[idx]
    return idx.reshape(shape)


def step_toward(pos, tgt):
    x, y = pos
    if tgt[0] != x:
        return ["EAST" if tgt[0] > x else "WEST"]
    if tgt[1] != y:
        return ["SOUTH" if tgt[1] > y else "NORTH"]
    return None


def trim_plants(unit_actions, seeds):
    """Engine drops EVERY PLANT of a crop when requests exceed seeds held: keep the first seeds[c] requests (farmer first), PASS the rest. Returns trimmed unit indices."""
    left = {c: int(seeds.get(c, 0) or 0) for c in CROPS}
    out = []
    for i, ua in enumerate(unit_actions):
        if ua and ua[0] == "PLANT":
            if left[ua[1]] > 0:
                left[ua[1]] -= 1
            else:
                unit_actions[i] = ["PASS"]
                out.append(i)
    return out


def legal_option_mask(obs, seat, unit_index, position):
    """Return current legal options plus structurally valid options at remote tiles."""
    mask = np.zeros((100, N_OPS), dtype=bool)
    tiles = obs["farms"][seat]["tiles"]
    for destination in range(100):
        x, y = destination % 10, destination // 10
        if (x, y) == position:
            mask[destination] = legal_ops_at(obs, seat, unit_index, x, y)
            mask[destination, 1:5] = False
            continue
        tile = tiles[y][x]
        names = []
        if tile is None:
            names = ["PLANT_" + crop for crop in CROPS] + ["DIG", "BUILD_COOP", "BUILD_PASTURE"]
        elif isinstance(tile, dict):
            if "animal" in tile:
                names = ["FEED", "CARE", "HARVEST", "COLLECT_FERTILIZER"]
            elif tile.get("kind") == "WEED":
                names = ["DIG"]
            elif tile.get("kind") == "PLANT":
                names = ["WATER", "HARVEST", "FERTILIZE", "DIG"]
            elif tile.get("kind") == "COOP":
                names = ["DIG", "PLACE_GOOSE"]
            elif tile.get("kind") == "PASTURE":
                names = ["DIG", "PLACE_COW", "PLACE_SHEEP"]
        if (x, y) in SHED_TILES:
            names += ["PICKUP_" + item for item in (*PRODUCTS, "GOOSE", "COW", "SHEEP")]
            names += ["DROP"] + ["PLACE_" + item for item in PRODUCTS]
        for name in names:
            mask[destination, OP_INDEX[name]] = True
    return mask


# Sell rule C (bc15 vs E065 diagnosis, +3k/32 games vs v41): evening bulk-sell
# overlay. SELLs stay in front slots (E041 doctrine).
SELL_RULE_C_HOUR = 20
SELL_RULE_C_KEEP = ("WHEAT", "FERTILIZER")


def apply_sell_rule_c(action, obs, seat):
    """Evening bulk-sell overlay (default options)."""
    return apply_sell_rules(action, obs, seat)


def apply_sell_rules(action, obs, seat, dump_fert=True, intra_thresh=0, melon_hold_until=0):
    """Market-layer sell overlay (pure function, portable to numpy export).

    evening: at hour >= 20 dump all shed except WHEAT (+FERTILIZER unless
      dump_fert) as front-slot SELLs; model's own sells for those products
      are replaced, everything else kept with farm purchases protected.
    intra_thresh: if > 0, also dump any such product whose shed reaches the
      threshold — but only in the post-tick phase (h%4 in {1,2}, E039).
    melon_hold_until: if > 0, MELON is excluded from evening dumps while
      day < melon_hold_until (burst-recovery hold; rejected vs E065, keep 0).
    """
    step = int(obs["step"])
    day, hour = step // 24, step % 24
    prices = obs["market"]["prices"]
    shed = (obs.get("private") or {}).get("shed") or {}
    skip = ("WHEAT",) if dump_fert else ("WHEAT", "FERTILIZER")
    if 0 < melon_hold_until and day < melon_hold_until:
        skip = skip + ("MELON",)

    def full_dump():
        out = []
        for product in PRODUCTS:
            if product in skip:
                continue
            quantity = int(shed.get(product, 0) or 0)
            if quantity > 0:
                out.append((prices.get(product, 0) * quantity, ["SELL", product, quantity]))
        return out

    forced = []
    if hour >= SELL_RULE_C_HOUR:
        forced = full_dump()
    elif intra_thresh > 0 and hour % 4 in (1, 2):
        for product in PRODUCTS:
            if product in skip:
                continue
            quantity = int(shed.get(product, 0) or 0)
            if quantity >= intra_thresh:
                forced.append((prices.get(product, 0) * quantity, ["SELL", product, quantity]))
    if not forced:
        return action
    forced = [order for _, order in sorted(forced, key=lambda pair: -pair[0])]
    kept = []
    for order in action.get("market") or []:
        if not order:
            continue
        if order[0] == "SELL" and len(order) > 1 and order[1] not in skip:
            continue
        kept.append(order)
    # Bulk sells stay in front (E041 slot priority) but never crowd out the
    # model's own orders (decode_action already caps those at 10): on
    # overflow the lowest-value bulk dumps are trimmed first.
    action["market"] = (forced[: max(0, 10 - len(kept))] + kept)[:10]
    return action


def apply_cow_cap(action, state, cap=9):
    """Redirect BUY_ANIMAL COW orders beyond lifetime cap to SHEEP (pure-ish).

    state["cows_bought"] tracks ordered (not executed) counts. Rejected by
    A/B (unplaced sheep, E046-like); kept as a flag for experiments.
    """
    if state is None:
        return action
    bought = int(state.get("cows_bought", 0))
    market = action.get("market") or []
    for order in market:
        if not order or order[0] != "BUY_ANIMAL" or len(order) < 3:
            continue
        if order[1] == "COW":
            room = max(0, cap - bought)
            take = min(int(order[2]), room)
            extra = int(order[2]) - take
            bought += int(order[2])
            if extra > 0:
                order[2] = take
                market.append(["BUY_ANIMAL", "SHEEP", extra])
    action["market"] = [o for o in market if not (o and o[0] == "BUY_ANIMAL" and len(o) > 2 and int(o[2]) <= 0)][:10]
    state["cows_bought"] = bought
    return action
