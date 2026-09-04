"""2エージェントを同シード両席で直接対決させ、ペア margin と初ショップを出す。
使い方: .venv/bin/python tests/h2h.py A.py B.py --games 16 [--seed0 0]
"""
import argparse, importlib.util, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kaggle_environments import make

def load(p):
    s = importlib.util.spec_from_file_location(Path(p).stem + str(abs(hash(p))), p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return getattr(m, "agent_entry", m.agent)

def play(a, b, seed):
    env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=False)
    env.info["seed"] = seed
    env.run([a, b])
    shop = ",".join(s[:3] for s in (env.steps[-1][0].observation.get("town") or {}).get("unlocked_shops", []))
    return env.steps[-1][0].reward, env.steps[-1][1].reward, shop

ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--games", type=int, default=16); ap.add_argument("--seed0", type=int, default=0)
o = ap.parse_args(); A, B = load(o.a), load(o.b)
tot = 0; w = 0; n = 0
for seed in range(o.seed0, o.seed0 + o.games):
    a0, b0, shop = play(A, B, seed); b1, a1, _ = play(B, A, seed)
    for m in (a0 - b0, a1 - b1):
        tot += m; n += 1; w += m > 0
    print(f"seed {seed:>3} {shop:<32} A-as-p0 {a0-b0:>+9,.0f}  A-as-p1 {a1-b1:>+9,.0f}", flush=True)
print(f"A={o.a} vs B={o.b}: {w}W{n-w}L, avg margin {tot/n:+,.0f}")
