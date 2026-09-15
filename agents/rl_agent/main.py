"""Learned policy agent (numpy inference). Weights in weights.npz next to this file; features/actions/np_policy2 copied from rl/."""
from __future__ import annotations
import os, sys, importlib.util
import numpy as np
from pathlib import Path



def _load(folder, name):
    spec = importlib.util.spec_from_file_location(name, str(folder / (name + ".py"))); mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


class RLAgent:
    def __init__(self, folder):
        folder = Path(folder); sys.path.insert(0, str(folder))
        self.F = _load(folder, "features"); self.A = _load(folder, "actions"); _load(folder, "np_policy"); self.NP = _load(folder, "np_policy2")
        W = dict(np.load(folder / "weights.npz")); meta = W.pop("__meta__", None)
        d, layers, heads = (int(meta[0]), int(meta[1]), int(meta[2])) if meta is not None else (128, 3, 4)
        self.pol = self.NP.NpPolicy2(W, d, layers, heads)

    def act(self, obs):
        F, A = self.F, self.A; seat = int(obs["player"]); f = F.encode(obs, seat); step = int(obs["step"])
        farm = obs["farms"][seat]; units = [farm["farmer"], *farm["hands"]][:F.MAX_UNITS]; n = len(units)
        if step % 24 == 0 or getattr(self, "prev", None) is None:
            self.prev = np.zeros((F.MAX_UNITS, 2), dtype=np.int16); self.prev[:, 0] = 100; self.prev[:, 1] = 44
        H = self.pol.encode(f, self.prev); dl = self.pol.dest_logits(H); claimed = set(); dest = np.zeros(F.MAX_UNITS, dtype=np.int64); tg = []
        newprev = np.zeros((F.MAX_UNITS, 2), dtype=np.int16); newprev[:, 0] = 100; newprev[:, 1] = 44
        for i in range(n):
            lg = dl[i].copy()
            for c in claimed: lg[c] = -1e9
            d = int(lg.argmax()); dest[i] = d
            for _ in range(4):  # a destination where the unit would only PASS is wasted: fall back to the next-best tiles (mirrors rl/act2.py)
                olg = self.pol.op_logits(H, dest)[0][i].copy(); olg[~F.legal_ops_at(obs, seat, i, d % 10, d // 10)] = -1e9
                if int(olg.argmax()) != F.OP_INDEX["PASS"]: break
                lg[d] = -1e9; d = int(lg.argmax()); dest[i] = d
            tx, ty = d % 10, d // 10
            if (tx, ty) not in F.SHED_TILES: claimed.add(d)
            tg.append((tx, ty))
        opl, qtl = self.pol.op_logits(H, dest); ua = []
        for i, (tx, ty) in enumerate(tg):
            pos = (int(units[i][0]), int(units[i][1])); newprev[i, 0] = dest[i]
            if pos != (tx, ty):
                ua.append(["EAST" if tx > pos[0] else "WEST"] if tx != pos[0] else ["SOUTH" if ty > pos[1] else "NORTH"]); continue
            m = F.legal_ops_at(obs, seat, i, tx, ty); lg = opl[i].copy(); lg[~m] = -1e9; k = int(lg.argmax()); name = F.OPS[k]; q = F.unbucket(int(qtl[i].argmax()), F.QTY_BUCKETS); newprev[i, 1] = k
            if name.startswith("PLANT_"): ua.append(["PLANT", name[6:]])
            elif name.startswith("PICKUP_"): ua.append(["PICKUP", name[7:], int(q)])
            elif name.startswith("PLACE_"): ua.append(["PLACE", name[6:], int(q)] if name[6:] in F.PRODUCTS else ["PLACE", name[6:]])
            else: ua.append([name])
        left = {c: int(obs["private"]["seeds"].get(c, 0) or 0) for c in F.CROPS}  # engine drops every PLANT of a crop when requests exceed seeds held
        for i, u in enumerate(ua):
            if u and u[0] == "PLANT":
                if left[u[1]] > 0: left[u[1]] -= 1
                else: ua[i] = ["PASS"]; newprev[i, 1] = F.OP_INDEX["PASS"]
        mk = self.pol.market(H)
        mkt = np.concatenate([mk["sell"].argmax(-1), mk["buyp"].argmax(-1), mk["seed"].argmax(-1), mk["anim"].argmax(-1), [int(mk["hire"].argmax())], [int(mk["land"].argmax())]])
        a = A.decode_action(np.zeros(F.MAX_UNITS, dtype=int), np.zeros(F.MAX_UNITS, dtype=int), mkt, obs, seat)
        a["farmer"] = ua[0] if ua else ["PASS"]; a["hands"] = ua[1:]; a["market"] = a["market"][:10]; self.prev = newprev
        return a


_AGENT = None


def agent(observation, configuration=None):
    global _AGENT
    try:
        if _AGENT is None: _AGENT = RLAgent(Path(agent.__code__.co_filename).resolve().parent)
        return _AGENT.act(observation)
    except Exception:
        try:
            farm = observation["farms"][int(observation["player"])]
            return {"farmer": ["PASS"], "hands": [["PASS"] for _ in farm["hands"]], "market": []}
        except Exception:
            return {"farmer": ["PASS"], "hands": [], "market": []}
