"""Inference for Policy2: sequential per-unit destination choice with tile claiming, shortest-path moves, op at arrival."""
from __future__ import annotations
import numpy as np, torch
import os
from features import encode, legal_ops_at, OPS, OP_INDEX, MAX_UNITS, SHED_TILES, QTY_BUCKETS, unbucket, PRODUCTS
RESELECT = os.environ.get("RL_RESELECT", "1") != "0"; RESELECT_TRIES = 4
COMMIT_DEST = os.environ.get("RL_COMMIT_DEST", "1") != "0"

from actions import decode_action
from act_common import step_toward, trim_plants  # noqa: F401 (re-export for rollout2/sp)


def act_policy2(model, obs, seat, dev, temperature=0.0, rng=None, state=None, eps=0.0, eps_op=None):
    """Returns (action dict, info) using greedy (or sampled) destination + op. Units are resolved farmer-first; a non-shed tile is claimed once.
    state: dict carried across steps ({"prev": np.int16[U,2]}) for the previous-decision features; reset at hour 0.
    eps: 目的地の決定のこの割合を「合法な選択肢から一様乱択」に差し替える (4d の用量反応測定用。0 で従来どおり)。
    eps_op: 到着時の作業に対する同じ割合。None なら eps と同じ (目的地側と作業側を分けて効きを見るため)。"""
    f = encode(obs, seat); farm = obs["farms"][seat]; units = [farm["farmer"], *farm["hands"]][:MAX_UNITS]; n = len(units)
    prev = np.zeros((MAX_UNITS, 2), dtype=np.int16); prev[:, 0] = 100; prev[:, 1] = 44
    if state is not None and int(obs["step"]) % 24 != 0 and "prev" in state: prev = state["prev"]
    with torch.no_grad():
        H = model.encode(*(torch.from_numpy(f[k]).unsqueeze(0).to(dev) for k in ("tiles", "units", "items", "glob")), prev=torch.from_numpy(prev).unsqueeze(0).to(dev))
        dl = model.dest_logits(H)[0].cpu().numpy()  # [U,100]
        dl[:, H["locked"][0].cpu().numpy()] = -1e9
        mk = model.market(H)
    claimed = set(); dest_idx = np.zeros(MAX_UNITS, dtype=np.int64); acts = []
    _rng = rng or np.random
    if eps_op is None: eps_op = eps
    def noisy(lg, e):
        """e の確率で合法な選択肢 (lg > -1e8) から一様乱択、それ以外は argmax。"""
        if e > 0 and _rng.random() < e:
            ok = np.flatnonzero(lg > -1e8)
            if len(ok): return int(_rng.choice(ok))
        return int(lg.argmax())
    def arrival_op(i):
        """Op the op-head would pick if unit i stood on its chosen destination now (engine-exact legality)."""
        with torch.no_grad(): opl, _ = model.op_logits(H, torch.from_numpy(dest_idx).unsqueeze(0).to(dev))
        tx, ty = int(dest_idx[i]) % 10, int(dest_idx[i]) // 10
        lg = opl[0, i].cpu().numpy().copy(); lg[~legal_ops_at(obs, seat, i, tx, ty)] = -1e9
        return int(lg.argmax())
    for i in range(n):
        lg = dl[i].copy()
        for c in claimed: lg[c] = -1e9
        prior_dest = int(prev[i, 0])
        pos = (int(units[i][0]), int(units[i][1]))
        if COMMIT_DEST and prior_dest < 100 and pos != (prior_dest % 10, prior_dest // 10) and lg[prior_dest] > -1e8:
            d = prior_dest
        elif temperature > 0:
            p = np.exp((lg - lg.max()) / temperature); p /= p.sum(); d = int(_rng.choice(100, p=p))
        else: d = noisy(lg, eps)
        dest_idx[i] = d
        if RESELECT:  # a destination where the unit would only PASS is wasted: fall back to the next-best tiles (BUILD fallbacks matter: structures come almost only from here)
            for _ in range(RESELECT_TRIES):
                if arrival_op(i) != OP_INDEX["PASS"]: break
                lg[dest_idx[i]] = -1e9; dest_idx[i] = int(lg.argmax())
        d = int(dest_idx[i]); tx, ty = d % 10, d // 10
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
            p = np.exp((lg - lg.max()) / temperature); p /= p.sum(); k = int(_rng.choice(len(OPS), p=p))
        else: k = noisy(lg, eps_op)
        name = OPS[k]; q = unbucket(int(qtl[i].argmax()), QTY_BUCKETS); newprev[i, 1] = k
        if name.startswith("PLANT_"): unit_actions.append(["PLANT", name[6:]])
        elif name.startswith("PICKUP_"): unit_actions.append(["PICKUP", name[7:], int(q)])
        elif name.startswith("PLACE_"): unit_actions.append(["PLACE", name[6:], int(q)] if name[6:] in PRODUCTS else ["PLACE", name[6:]])
        else: unit_actions.append([name])
    for i in trim_plants(unit_actions, obs["private"]["seeds"]): newprev[i, 1] = OP_INDEX["PASS"]
    mkt = np.concatenate([mk["sell"][0].argmax(-1).cpu().numpy(), mk["buyp"][0].argmax(-1).cpu().numpy(), mk["seed"][0].argmax(-1).cpu().numpy(),
                          mk["anim"][0].argmax(-1).cpu().numpy(), [int(mk["hire"][0].argmax())], [int(mk["land"][0].argmax())]])
    a = decode_action(np.zeros(MAX_UNITS, dtype=int), np.zeros(MAX_UNITS, dtype=int), mkt, obs, seat)
    a["farmer"] = unit_actions[0] if unit_actions else ["PASS"]; a["hands"] = unit_actions[1:]
    if state is not None: state["prev"] = newprev
    return a, {"dest": dest_idx[:n]}
