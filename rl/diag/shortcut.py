"""Open-loop shortcut probe on held-out Majkel games: how much do unit options depend on the clock vs the board?
For each ckpt: option accuracy, and the fraction of units whose argmax option flips when only the time features
(glob[0:6] = day, hour, sin, cos, step, last-day flag) are shifted by +1 day, or when the opponent farm is blanked.
usage: shortcut.py CKPT..."""
import sys, os, glob, random
import numpy as np, torch
sys.path.insert(0, os.path.join(os.getcwd(), "rl"))
from model3 import Policy3, option_mask_from_features, N_OWN
from features import N_OPS

files = sorted(glob.glob("tmp/rl/majkel_all/*.npz")); random.Random(0).shuffle(files); files = files[:40]
batches = []
for f in files:
    d = np.load(f); idx = np.arange(0, len(d["dest"]), 3)
    batches.append({k: d[k][idx] for k in ("tiles", "units", "items", "glob", "dest", "dop")})


def options(m, b, glob_override=None, tiles_override=None):
    t = torch.from_numpy((tiles_override if tiles_override is not None else b["tiles"]).astype(np.int16))
    u = torch.from_numpy(b["units"].astype(np.int16))
    g = torch.from_numpy(glob_override if glob_override is not None else b["glob"])
    h = m.encode(t, u, torch.from_numpy(b["items"]), g)
    lg = m.option_logits(h).masked_fill(~option_mask_from_features(t, u), -1e9).flatten(2)
    return lg.argmax(-1).numpy()


for ck in sys.argv[1:]:
    m = Policy3(); m.load_state_dict(torch.load(ck, map_location="cpu")); m.eval()
    acc = n = flip_day = flip_hour = flip_opp = present_n = 0
    with torch.no_grad():
        for b in batches:
            present = b["units"][..., 0] > 0; lab = b["dest"] >= 0
            base = options(m, b)
            target = b["dest"].astype(int) * N_OPS + b["dop"].astype(int)
            acc += (base == target)[lab].sum(); n += lab.sum()
            g = b["glob"].copy(); g[:, 0] = np.minimum(1.0, g[:, 0] + 1 / 29.0); g[:, 4] = np.minimum(1.0, g[:, 4] + 24 / 719.0)
            flip_day += (options(m, b, glob_override=g) != base)[present].sum()
            g = b["glob"].copy(); hour = np.round(g[:, 1] * 23); hour2 = (hour + 1) % 24
            g[:, 1] = hour2 / 23.0; g[:, 2] = np.sin(2 * np.pi * hour2 / 24); g[:, 3] = np.cos(2 * np.pi * hour2 / 24); g[:, 4] = np.minimum(1.0, g[:, 4] + 1 / 719.0)
            flip_hour += (options(m, b, glob_override=g) != base)[present].sum()
            t = b["tiles"].copy(); t[:, 1] = 0; t[:, 1, ..., 16] = b["tiles"][:, 1, ..., 16]; t[:, 1, ..., 17] = b["tiles"][:, 1, ..., 17]; t[:, 1, ..., 0] = 1; t[:, 1, ..., 15] = 1
            flip_opp += (options(m, b, tiles_override=t) != base)[present].sum()
            present_n += present.sum()
    print(f"{os.path.basename(ck):16s} option acc {acc / n:.3f} | flip: +1 day {flip_day / present_n:.3f}  +1 hour {flip_hour / present_n:.3f}  opp farm blanked {flip_opp / present_n:.3f}", flush=True)
