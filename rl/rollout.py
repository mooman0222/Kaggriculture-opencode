"""Vectorised self-play rollouts in kagsim: G games in lockstep, batched policy inference, opponents from a pool.
Reward per step = (Δ own money − Δ rival money) / 1000 so the episode return equals the final margin in k$."""
from __future__ import annotations
import os, sys, importlib.util, hashlib, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, kagsim
from features import encode, legal_ops, MAX_UNITS, N_OPS
from actions import decode_action, MKT_CLASSES
from model import Policy, N_MB


def load_py(p):
    d = os.path.dirname(os.path.abspath(p)); sys.path.insert(0, d)
    s = importlib.util.spec_from_file_location("opp_" + hashlib.md5(p.encode()).hexdigest()[:8], p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


class ScriptedOpp:
    def __init__(self, path): self.m = load_py(path)
    def reset(self):
        for k in ("_LIVE", "_POLICY"):
            if hasattr(self.m, k): setattr(self.m, k, None)
    def act(self, obs, seat): return self.m.agent(obs)


def sample_heads(out, mask, greedy=False):
    """Sample factorized action from logits (batched torch). Returns dict of index tensors and summed log-prob."""
    logits = out["op"].masked_fill(~mask, -1e9)
    def pick(lg):
        dist = torch.distributions.Categorical(logits=lg)
        a = lg.argmax(-1) if greedy else dist.sample(); return a, dist.log_prob(a), dist.entropy()
    op, lp_op, ent_op = pick(logits); qty, lp_q, _ = pick(out["qty"])
    is_q = ((op >= 16) & (op <= 27)) | ((op >= 29) & (op <= 40))
    sell, lp_s, ent_s = pick(out["sell"]); buyp, lp_b, _ = pick(out["buyp"]); seed, lp_sd, _ = pick(out["seed"]); anim, lp_a, _ = pick(out["anim"])
    hire, lp_h, ent_h = pick(out["hire"]); land, lp_l, _ = pick(out["land"])
    present = mask.any(-1)
    logp = (lp_op * present).sum(-1) + (lp_q * is_q * present).sum(-1) + lp_s.sum(-1) + lp_b.sum(-1) + lp_sd.sum(-1) + lp_a.sum(-1) + lp_h + lp_l
    ent = (ent_op * present).sum(-1) / present.sum(-1).clamp(min=1) + ent_s.mean(-1) + ent_h
    return {"op": op, "qty": qty, "sell": sell, "buyp": buyp, "seed": seed, "anim": anim, "hire": hire, "land": land}, logp, ent


def logp_of(out, mask, act):
    """Log-prob (and entropy) of stored actions under current logits — for the PPO update."""
    logits = out["op"].masked_fill(~mask, -1e9)
    def lp(lg, a):
        d = torch.distributions.Categorical(logits=lg); return d.log_prob(a), d.entropy()
    present = mask.any(-1); op = act["op"]
    is_q = ((op >= 16) & (op <= 27)) | ((op >= 29) & (op <= 40))
    l_op, e_op = lp(logits, op); l_q, _ = lp(out["qty"], act["qty"]); l_s, e_s = lp(out["sell"], act["sell"]); l_b, _ = lp(out["buyp"], act["buyp"])
    l_sd, _ = lp(out["seed"], act["seed"]); l_a, _ = lp(out["anim"], act["anim"]); l_h, e_h = lp(out["hire"], act["hire"]); l_l, _ = lp(out["land"], act["land"])
    logp = (l_op * present).sum(-1) + (l_q * is_q * present).sum(-1) + l_s.sum(-1) + l_b.sum(-1) + l_sd.sum(-1) + l_a.sum(-1) + l_h + l_l
    ent = (e_op * present).sum(-1) / present.sum(-1).clamp(min=1) + e_s.mean(-1) + e_h
    return logp, ent


def to_action(acts, i, obs, seat):
    mkt = np.concatenate([acts["sell"][i], acts["buyp"][i], acts["seed"][i], acts["anim"][i], [acts["hire"][i]], [acts["land"][i]]])
    return decode_action(acts["op"][i], acts["qty"][i], mkt, obs, seat)


def rollout(model, dev, opponents, n_games=32, seed0=0, greedy=False, opening=None, opening_steps=0):
    """Play n_games (learner seat alternates) against opponents sampled from the list. Returns trajectory dict (numpy) and results."""
    games = []
    for i in range(n_games):
        opp = random.choice(opponents); opp.reset(); seat = i % 2
        if opening: opening.reset()
        games.append({"g": kagsim.Game(seed0 + i), "opp": opp, "seat": seat, "prev": (3000.0, 3000.0)})
    T = {k: [] for k in ("tiles", "units", "items", "glob", "mask", "op", "qty", "sell", "buyp", "seed", "anim", "hire", "land", "logp", "value", "reward")}
    model.eval()
    for step in range(719):
        obs = [gm["g"].observe(gm["seat"]) for gm in games]
        feats = [encode(o, gm["seat"]) for o, gm in zip(obs, games)]; masks = np.stack([legal_ops(o, gm["seat"]) for o, gm in zip(obs, games)])
        batch = {k: torch.from_numpy(np.stack([f[k] for f in feats])).to(dev) for k in ("tiles", "units", "items", "glob")}
        mask_t = torch.from_numpy(masks).to(dev)
        with torch.no_grad():
            out = model(batch["tiles"], batch["units"], batch["items"], batch["glob"]); acts, logp, _ = sample_heads(out, mask_t, greedy)
        acts_np = {k: v.cpu().numpy() for k, v in acts.items()}
        for i, gm in enumerate(games):
            o = obs[i]; g = gm["g"]; s = gm["seat"]
            a = opening.act(o, s) if (opening and step < opening_steps) else to_action(acts_np, i, o, s)
            b = gm["opp"].act(g.observe(1 - s), 1 - s)
            g.step(*((a, b) if s == 0 else (b, a)))
            o2 = g.observe(0); m_own = float(o2["farms"][s]["money"]); m_opp = float(o2["farms"][1 - s]["money"])
            r = ((m_own - gm["prev"][0]) - (m_opp - gm["prev"][1])) / 1000.0; gm["prev"] = (m_own, m_opp)
            T["reward"].append(r)
        for k in ("tiles", "units", "items", "glob"): T[k].append(np.stack([f[k] for f in feats]))
        T["mask"].append(masks)
        for k in acts_np: T[k].append(acts_np[k])
        T["logp"].append(logp.cpu().numpy()); T["value"].append(out["value"].cpu().numpy())
    res = [(gm["g"].reward(gm["seat"]), gm["g"].reward(1 - gm["seat"])) for gm in games]
    traj = {k: np.stack(v) for k, v in T.items() if k != "reward"}  # [T, G, ...]
    traj["reward"] = np.array(T["reward"], dtype=np.float32).reshape(719, n_games)
    return traj, res
