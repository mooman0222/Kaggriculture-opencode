"""Inference for Policy2: sequential per-unit destination choice with tile claiming, shortest-path moves, op at arrival."""
from __future__ import annotations
import numpy as np, torch
from features import encode, legal_ops_at, OPS, MAX_UNITS, SHED_TILES, QTY_BUCKETS, unbucket, PRODUCTS
from actions import decode_action


def step_toward(pos, tgt):
    x, y = pos
    if tgt[0] != x: return ["EAST" if tgt[0] > x else "WEST"]
    if tgt[1] != y: return ["SOUTH" if tgt[1] > y else "NORTH"]
    return None


def act_policy2(model, obs, seat, dev, temperature=0.0, rng=None, state=None):
    """Returns (action dict, info) using greedy (or sampled) destination + op. Units are resolved farmer-first; a non-shed tile is claimed once.
    state: dict carried across steps ({"prev": np.int16[U,2]}) for the previous-decision features; reset at hour 0."""
    f = encode(obs, seat); farm = obs["farms"][seat]; units = [farm["farmer"], *farm["hands"]][:MAX_UNITS]; n = len(units)
    prev = np.zeros((MAX_UNITS, 2), dtype=np.int16); prev[:, 0] = 100; prev[:, 1] = 44
    if state is not None and int(obs["step"]) % 24 != 0 and "prev" in state: prev = state["prev"]
    with torch.no_grad():
        H = model.encode(*(torch.from_numpy(f[k]).unsqueeze(0).to(dev) for k in ("tiles", "units", "items", "glob")), prev=torch.from_numpy(prev).unsqueeze(0).to(dev))
        dl = model.dest_logits(H)[0].cpu().numpy()  # [U,100]
        dl[:, H["locked"][0].cpu().numpy()] = -1e9
        mk = model.market(H)
    claimed = set(); dest_idx = np.zeros(MAX_UNITS, dtype=np.int64); acts = []
    for i in range(n):
        lg = dl[i].copy()
        for c in claimed: lg[c] = -1e9
        if temperature > 0:
            p = np.exp((lg - lg.max()) / temperature); p /= p.sum(); d = int((rng or np.random).choice(100, p=p))
        else: d = int(lg.argmax())
        dest_idx[i] = d; tx, ty = d % 10, d // 10
        if (tx, ty) not in SHED_TILES: claimed.add(d)
        acts.append((tx, ty))
    with torch.no_grad():
        opl, qtl = model.op_logits(H, torch.from_numpy(dest_idx).unsqueeze(0).to(dev)); opl = opl[0].cpu().numpy(); qtl = qtl[0].cpu().numpy()
    unit_actions = []; newprev = np.zeros((MAX_UNITS, 2), dtype=np.int16); newprev[:, 0] = 100; newprev[:, 1] = 44
    for i, (tx, ty) in enumerate(acts):
        newprev[i, 0] = dest_idx[i]
        pos = (int(units[i][0]), int(units[i][1]))
        if pos != (tx, ty):
            unit_actions.append(step_toward(pos, (tx, ty))); continue
        m = legal_ops_at(obs, seat, i, tx, ty); lg = opl[i].copy(); lg[~m] = -1e9
        if temperature > 0:
            p = np.exp((lg - lg.max()) / temperature); p /= p.sum(); k = int((rng or np.random).choice(len(OPS), p=p))
        else: k = int(lg.argmax())
        name = OPS[k]; q = unbucket(int(qtl[i].argmax()), QTY_BUCKETS); newprev[i, 1] = k
        if name.startswith("PLANT_"): unit_actions.append(["PLANT", name[6:]])
        elif name.startswith("PICKUP_"): unit_actions.append(["PICKUP", name[7:], int(q)])
        elif name.startswith("PLACE_"): unit_actions.append(["PLACE", name[6:], int(q)] if name[6:] in PRODUCTS else ["PLACE", name[6:]])
        else: unit_actions.append([name])
    mkt = np.concatenate([mk["sell"][0].argmax(-1).cpu().numpy(), mk["buyp"][0].argmax(-1).cpu().numpy(), mk["seed"][0].argmax(-1).cpu().numpy(),
                          mk["anim"][0].argmax(-1).cpu().numpy(), [int(mk["hire"][0].argmax())], [int(mk["land"][0].argmax())]])
    a = decode_action(np.zeros(MAX_UNITS, dtype=int), np.zeros(MAX_UNITS, dtype=int), mkt, obs, seat)
    a["farmer"] = unit_actions[0] if unit_actions else ["PASS"]; a["hands"] = unit_actions[1:]
    if state is not None: state["prev"] = newprev
    return a, {"dest": dest_idx[:n]}
