"""Vectorised rollouts for Policy2 (destination + op). Stores the exact masks used at sampling so the PPO update can recompute log-probs.
Reward per step = (Δ own money − Δ rival money) / 1000."""
from __future__ import annotations
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F, kagsim
from features import encode, legal_ops_at, OPS, MAX_UNITS, SHED_TILES, QTY_BUCKETS, unbucket, PRODUCTS
from actions import decode_action
from rollout import ScriptedOpp, load_py
from act2 import step_toward

N_OPS = len(OPS)


def _cat(logits, greedy):
    d = torch.distributions.Categorical(logits=logits); a = logits.argmax(-1) if greedy else d.sample(); return a, d.log_prob(a), d.entropy()


TEMP = 1.0  # sampling temperature for dest/op heads (set by the trainer; policy := softmax(logits / TEMP))


def logp2(model_out_fn, batch, act, per_head=False):
    """Recompute joint log-prob and entropy of stored actions. model_out_fn(batch, dest) -> (dest_logits, op_logits, qty_logits, market dict).
    per_head=True returns a [B, K] tensor of sub-action log-probs (units' dest, units' op, market heads) instead of the sum."""
    dest_lg, op_lg, qty_lg, mk = model_out_fn(batch, act["dest"])
    dest_lg = (dest_lg / TEMP).masked_fill(~batch["dest_mask"], -1e9); op_lg = (op_lg / TEMP).masked_fill(~batch["op_mask"], -1e9)
    present = batch["present"]; at_dest = batch["at_dest"]
    ld = torch.distributions.Categorical(logits=dest_lg); lo = torch.distributions.Categorical(logits=op_lg); lq = torch.distributions.Categorical(logits=qty_lg)
    op = act["op"]; is_q = ((op >= 16) & (op <= 27)) | ((op >= 29) & (op <= 40))
    parts = [ld.log_prob(act["dest"]) * present, lo.log_prob(op) * present * at_dest, lq.log_prob(act["qty"]) * present * at_dest * is_q]
    ent = (ld.entropy() * present).sum(-1) / present.sum(-1).clamp(min=1)
    for k in ("sell", "buyp", "seed", "anim"):
        d = torch.distributions.Categorical(logits=mk[k] / TEMP); parts.append(d.log_prob(act[k]))
        if k == "sell": ent = ent + d.entropy().mean(-1) * 0.2
    for k in ("hire", "land"):
        d = torch.distributions.Categorical(logits=mk[k] / TEMP); parts.append(d.log_prob(act[k]).unsqueeze(-1))
        if k == "hire": ent = ent + d.entropy() * 0.2
    if per_head: return torch.cat(parts, -1), ent
    return sum(p.sum(-1) for p in parts), ent


def model_out_fn(model):
    def fn(batch, dest):
        H = model.encode(batch["tiles"], batch["units"], batch["items"], batch["glob"], batch.get("prev")); dl = model.dest_logits(H); op_lg, q_lg = model.op_logits(H, dest.clamp(min=0)); mk = model.market(H)
        return dl, op_lg, q_lg, mk
    return fn


def rollout2(model, dev, opponents, n_games=32, seed0=0, greedy=False, opening=None, opening_steps=0):
    games = []
    for i in range(n_games):
        opp = random.choice(opponents); opp.reset(); seat = i % 2
        if opening: opening.reset()
        games.append({"g": kagsim.Game(seed0 + i), "opp": opp, "seat": seat, "prev": (3000.0, 3000.0)})
    T = {k: [] for k in ("tiles", "units", "items", "glob", "prev", "dest_mask", "op_mask", "present", "at_dest", "dest", "op", "qty", "sell", "buyp", "seed", "anim", "hire", "land", "logp", "value", "reward")}
    model.eval(); fn = model_out_fn(model)
    prev = np.zeros((n_games, MAX_UNITS, 2), dtype=np.int16); prev[..., 0] = 100; prev[..., 1] = 44
    for step in range(719):
        obs = [gm["g"].observe(gm["seat"]) for gm in games]; feats = [encode(o, gm["seat"]) for o, gm in zip(obs, games)]
        batch = {k: torch.from_numpy(np.stack([f[k] for f in feats])).to(dev) for k in ("tiles", "units", "items", "glob")}
        if step % 24 == 0: prev[:] = 0; prev[..., 0] = 100; prev[..., 1] = 44
        batch["prev"] = torch.from_numpy(prev.copy()).to(dev); prev_used = prev.copy()
        B = n_games
        with torch.no_grad():
            H = model.encode(batch["tiles"], batch["units"], batch["items"], batch["glob"], batch["prev"]); dl = model.dest_logits(H) / TEMP; mk = model.market(H)
        dl_np = dl.cpu().numpy(); present = np.stack([f["units"][:, 0] > 0 for f in feats])
        dest = np.zeros((B, MAX_UNITS), dtype=np.int64); dmask = np.zeros((B, MAX_UNITS, 100), dtype=bool)
        for b in range(B):
            claimed = set(); locked = feats[b]["tiles"][0, :, :, 0].reshape(100) == 0
            for i in range(MAX_UNITS):
                if not present[b, i]: dmask[b, i, 0] = True; continue
                m = ~locked.copy()
                for c in claimed: m[c] = False
                if not m.any(): m[:] = ~locked
                dmask[b, i] = m; lg = dl_np[b, i].copy(); lg[~m] = -1e9
                if greedy: d = int(lg.argmax())
                else:
                    p = np.exp(lg - lg.max()); p /= p.sum(); d = int(np.random.choice(100, p=p))
                dest[b, i] = d
                if (d % 10, d // 10) not in SHED_TILES: claimed.add(d)
        dest_t = torch.from_numpy(dest).to(dev)
        with torch.no_grad(): op_lg, q_lg = model.op_logits(H, dest_t); op_lg = op_lg / TEMP
        op_mask = np.zeros((B, MAX_UNITS, N_OPS), dtype=bool); at_dest = np.zeros((B, MAX_UNITS), dtype=bool)
        for b in range(B):
            gm = games[b]; o = obs[b]; farm = o["farms"][gm["seat"]]; units = [farm["farmer"], *farm["hands"]][:MAX_UNITS]
            for i, u in enumerate(units):
                d = dest[b, i]; tx, ty = d % 10, d // 10
                if (int(u[0]), int(u[1])) == (tx, ty): at_dest[b, i] = True; op_mask[b, i] = legal_ops_at(o, gm["seat"], i, tx, ty)
                else: op_mask[b, i, 0] = True
            for i in range(len(units), MAX_UNITS): op_mask[b, i, 0] = True
        op_mask_t = torch.from_numpy(op_mask).to(dev); present_t = torch.from_numpy(present).to(dev); at_dest_t = torch.from_numpy(at_dest).to(dev)
        with torch.no_grad():
            dlm = dl.masked_fill(~torch.from_numpy(dmask).to(dev), -1e9); olm = op_lg.masked_fill(~op_mask_t, -1e9)
            op_a, _, _ = _cat(olm, greedy); q_a, _, _ = _cat(q_lg, greedy)
            mk_a = {k: _cat(mk[k] / TEMP, greedy)[0] for k in ("sell", "buyp", "seed", "anim", "hire", "land")}
            act = {"dest": dest_t, "op": op_a, "qty": q_a, **mk_a}
            bt = {"tiles": batch["tiles"], "units": batch["units"], "items": batch["items"], "glob": batch["glob"], "prev": batch["prev"], "dest_mask": torch.from_numpy(dmask).to(dev), "op_mask": op_mask_t, "present": present_t, "at_dest": at_dest_t}
            lp, _ = logp2(fn, bt, act, per_head=True)
        act_np = {k: v.cpu().numpy() for k, v in act.items()}
        prev[..., 0] = np.where(present, dest, 100); prev[..., 1] = np.where(present & at_dest, act_np["op"], 44)
        for b, gm in enumerate(games):
            o = obs[b]; g = gm["g"]; s = gm["seat"]
            if opening and step < opening_steps: a = opening.act(o, s)
            else:
                farm = o["farms"][s]; units = [farm["farmer"], *farm["hands"]][:MAX_UNITS]; ua = []
                for i, u in enumerate(units):
                    d = dest[b, i]; tx, ty = d % 10, d // 10; pos = (int(u[0]), int(u[1]))
                    if pos != (tx, ty): ua.append(step_toward(pos, (tx, ty))); continue
                    name = OPS[int(act_np["op"][b, i])]; q = unbucket(int(act_np["qty"][b, i]), QTY_BUCKETS)
                    if name.startswith("PLANT_"): ua.append(["PLANT", name[6:]])
                    elif name.startswith("PICKUP_"): ua.append(["PICKUP", name[7:], int(q)])
                    elif name.startswith("PLACE_"): ua.append(["PLACE", name[6:], int(q)] if name[6:] in PRODUCTS else ["PLACE", name[6:]])
                    else: ua.append([name])
                mkt = np.concatenate([act_np["sell"][b], act_np["buyp"][b], act_np["seed"][b], act_np["anim"][b], [act_np["hire"][b]], [act_np["land"][b]]])
                a = decode_action(np.zeros(MAX_UNITS, dtype=int), np.zeros(MAX_UNITS, dtype=int), mkt, o, s); a["farmer"] = ua[0] if ua else ["PASS"]; a["hands"] = ua[1:]
            bopp = gm["opp"].act(g.observe(1 - s), 1 - s); g.step(*((a, bopp) if s == 0 else (bopp, a)))
            o2 = g.observe(0); m_own = float(o2["farms"][s]["money"]); m_opp = float(o2["farms"][1 - s]["money"])
            T["reward"].append(((m_own - gm["prev"][0]) - (m_opp - gm["prev"][1])) / 1000.0); gm["prev"] = (m_own, m_opp)
        for k in ("tiles", "units", "items", "glob"): T[k].append(np.stack([f[k] for f in feats]))
        T["prev"].append(prev_used)
        T["dest_mask"].append(dmask); T["op_mask"].append(op_mask); T["present"].append(present); T["at_dest"].append(at_dest)
        for k in act_np: T[k].append(act_np[k])
        T["logp"].append(lp.cpu().numpy()); T["value"].append(mk["value"].cpu().numpy())
    res = [(gm["g"].reward(gm["seat"]), gm["g"].reward(1 - gm["seat"])) for gm in games]
    traj = {k: np.stack(v) for k, v in T.items() if k != "reward"}; traj["reward"] = np.array(T["reward"], dtype=np.float32).reshape(719, n_games)
    return traj, res


class PolicyOpp:
    """Frozen learned policy as an opponent (greedy Policy2 via act2) — for self-play in PPO."""
    def __init__(self, ckpt, dev="cpu"):
        from model2 import Policy2; from act2 import act_policy2
        self.m = Policy2().to(dev); self.m.load_state_dict(torch.load(ckpt, map_location=dev)); self.m.eval(); self.dev = dev; self.act2 = act_policy2; self.state = {}
    def reset(self): self.state = {}
    def act(self, obs, seat):
        if int(obs["step"]) == 0: self.state = {}
        return self.act2(self.m, obs, seat, self.dev, 0.0, state=self.state)[0]
