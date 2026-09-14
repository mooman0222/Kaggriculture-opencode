"""checkpoint (.pt) -> agents/rl_agent/weights.npz (+ copies the current rl/ inference modules). 使い方: .venv/bin/python rl/export_agent.py tmp/rl/bc4.pt [--d 128 --layers 3]"""
import argparse, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from np_policy import export_weights
ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--d", type=int, default=128); ap.add_argument("--layers", type=int, default=3); ap.add_argument("--heads", type=int, default=4); ap.add_argument("--out", default="agents/rl_agent")
a = ap.parse_args(); W = export_weights(torch.load(a.ckpt, map_location="cpu")); W["__meta__"] = np.array([a.d, a.layers, a.heads], dtype=np.int64)
np.savez(os.path.join(a.out, "weights.npz"), **W)
for f in ("features.py", "actions.py", "np_policy.py", "np_policy2.py"): shutil.copy(os.path.join("rl", f), os.path.join(a.out, f))
print("exported", a.ckpt, "->", a.out, "size MB", round(os.path.getsize(os.path.join(a.out, "weights.npz")) / 1e6, 2))
