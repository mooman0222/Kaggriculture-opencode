"""BC v2 方策 (目的タイル + 作業) を kagsim で走らせて評価 (対 v41 既定、両席)。使い方: .venv/bin/python rl/bc_play.py tmp/rl/bc.pt [--games 8] [--vs third_party/public_agents/v41/main.py] [--seed0 5000] [--debug]"""
import argparse, os, sys, importlib.util, hashlib, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, kagsim
from features import encode, legal_ops, OPS
from actions import decode_action
from model2 import Policy2 as Policy
from act2 import act_policy2


class BCAgent:
    def __init__(self, path, d=128, layers=3, dev="cpu", temperature=0.0, eps=0.0, eps_op=None, seed=0):
        self.dev = torch.device(dev); self.model = Policy(d=d, layers=layers).to(self.dev); self.model.load_state_dict(torch.load(path, map_location=self.dev), strict=False); self.model.eval(); self.temp = temperature; self.eps = eps; self.eps_op = eps_op; self.rng = np.random.RandomState(seed)

    def act(self, obs, seat):
        if int(obs["step"]) == 0: self.state = {}
        return act_policy2(self.model, obs, seat, self.dev, self.temp, rng=self.rng, state=self.__dict__.setdefault("state", {}), eps=self.eps, eps_op=self.eps_op)[0]

    def _act_v1(self, obs, seat):
        f = encode(obs, seat); m = legal_ops(obs, seat)
        with torch.no_grad():
            out = self.model(*(torch.from_numpy(f[k]).unsqueeze(0).to(self.dev) for k in ("tiles", "units", "items", "glob")))
        logits = out["op"][0].cpu().numpy(); logits[~m] = -1e9
        n = 1 + len(obs["farms"][seat]["hands"])
        if self.temp > 0:
            p = np.exp((logits - logits.max(-1, keepdims=True)) / self.temp); p /= p.sum(-1, keepdims=True)
            op = np.array([np.random.choice(len(OPS), p=p[i]) for i in range(len(p))])
        else: op = logits.argmax(-1)
        qty = out["qty"][0].argmax(-1).cpu().numpy()
        mkt = np.concatenate([out["sell"][0].argmax(-1).cpu().numpy(), out["buyp"][0].argmax(-1).cpu().numpy(), out["seed"][0].argmax(-1).cpu().numpy(),
                              out["anim"][0].argmax(-1).cpu().numpy(), [int(out["hire"][0].argmax())], [int(out["land"][0].argmax())]])
        return decode_action(op, qty, mkt, obs, seat)


def load_py(p):
    d = os.path.dirname(os.path.abspath(p)); sys.path.insert(0, d)
    s = importlib.util.spec_from_file_location("opp_" + hashlib.md5(p.encode()).hexdigest()[:8], p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--games", type=int, default=8); ap.add_argument("--vs", default="third_party/public_agents/v41/main.py")
    ap.add_argument("--seed0", type=int, default=5000); ap.add_argument("--debug", action="store_true"); ap.add_argument("--opening", type=int, default=0, help="この step までは agents/e058 を使う"); ap.add_argument("--d", type=int, default=128); ap.add_argument("--layers", type=int, default=3); ap.add_argument("--temp", type=float, default=0.0); ap.add_argument("--eps", type=float, default=0.0, help="決定のこの割合を合法手の一様乱択に差し替える (4d)"); ap.add_argument("--eps-op", type=float, default=None, help="到着時の作業だけの eps (既定は --eps と同じ)"); ap.add_argument("--noise-seed", type=int, default=0)
    a = ap.parse_args(); agent = BCAgent(a.ckpt, a.d, a.layers, temperature=a.temp, eps=a.eps, eps_op=a.eps_op, seed=a.noise_seed); opp = load_py(a.vs); res = []; t0 = time.time(); tmax = 0
    e058 = load_py("agents/e058/main.py") if a.opening else None
    for g_i in range(a.games):
        seed = a.seed0 + g_i // 2; seat = g_i % 2
        for k in ("_LIVE", "_POLICY"):
            if hasattr(opp, k): setattr(opp, k, None)
        g = kagsim.Game(seed); daily = []
        if e058: e058._LIVE = None
        while not g.done:
            o = g.observe(seat); t1 = time.perf_counter(); x = e058.agent(o) if (e058 and g.step_count < a.opening) else agent.act(o, seat); tmax = max(tmax, time.perf_counter() - t1); y = opp.agent(g.observe(1 - seat))
            if a.debug and g.step_count % 72 == 12: daily.append((g.step_count // 24, round(o["farms"][seat]["money"]), len(o["farms"][seat]["hands"]), sum(1 for r in o["farms"][seat]["tiles"] for t in r if isinstance(t, dict) and "animal" in t), sum(1 for r in o["farms"][seat]["tiles"] for t in r if isinstance(t, dict) and t.get("kind") == "PLANT")))
            g.step(*((x, y) if seat == 0 else (y, x)))
        tel = g.telemetry(seat); res.append((g.reward(seat), g.reward(1 - seat)))
        print(f"seed {seed} seat {seat}: own {g.reward(seat):8.0f} opp {g.reward(1-seat):8.0f} margin {g.reward(seat)-g.reward(1-seat):+8.0f} sold {tel.get('sold_units')} refused {sum(v for k,v in tel.items() if k.startswith('refused'))} {daily if a.debug else ''}", flush=True)
    print(f"mean own {np.mean([r[0] for r in res]):.0f} opp {np.mean([r[1] for r in res]):.0f} margin {np.mean([r[0]-r[1] for r in res]):+.0f} wins {sum(r[0]>r[1] for r in res)}/{len(res)} max step {tmax*1000:.0f} ms [{time.time()-t0:.0f}s]")
