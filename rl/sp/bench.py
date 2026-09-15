"""Throughput benchmark: VecEnv + batched Policy2 (BC init) self-play for N steps. Usage: .venv/bin/python rl/sp/bench.py [--workers 10] [--games 24] [--steps 60] [--ckpt tmp/rl/bc5_ep3.pt]"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from model2 import Policy2
from vec_env import VecEnv
from policy_batch import act_batch
from features import MAX_UNITS
ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=10); ap.add_argument("--games", type=int, default=24); ap.add_argument("--steps", type=int, default=60); ap.add_argument("--ckpt", default="tmp/rl/bc5_ep3.pt"); ap.add_argument("--dev", default="mps")
a = ap.parse_args(); dev = torch.device(a.dev)
m = Policy2().to(dev); m.load_state_dict(torch.load(a.ckpt, map_location=dev), strict=False); m.eval()
env = VecEnv(a.workers, a.games); A = env.A; slots = np.arange(env.n); prev = np.zeros((env.n, MAX_UNITS, 2), dtype=np.int16); prev[..., 0] = 100; prev[..., 1] = 44
t_inf = t_env = 0.0; t0 = time.time(); done = 0
for it in range(a.steps):
    t = time.perf_counter(); out = act_batch(m, dev, A, slots, prev, temp=0.3); torch.mps.synchronize() if a.dev == "mps" else None; t_inf += time.perf_counter() - t
    prev[..., 0] = np.where(out["present"], out["dest"], 100); prev[..., 1] = np.where(out["at_dest"], out["op"], 44)
    t = time.perf_counter(); fin = env.step(); t_env += time.perf_counter() - t; done += len(fin)
    hour = A["step"][0] % 24
    if hour == 0: prev[:] = 0; prev[..., 0] = 100; prev[..., 1] = 44
el = time.time() - t0; n = a.steps * env.G
print(f"games {env.G} slots {env.n}: {a.steps} steps in {el:.1f}s -> {n / el:.0f} game-steps/s ({2 * n / el:.0f} decisions/s) | inference {t_inf / a.steps * 1000:.0f} ms/step, env {t_env / a.steps * 1000:.0f} ms/step | money@step{int(A['step'][0])}: {A['money'][:, 0].mean():.0f}")
env.close()
