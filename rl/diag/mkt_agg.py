"""Open-loop: does per-step market decoding reproduce the teacher's per-game totals?
label (bucketed) vs argmax (deployed, pos_w-biased) vs argmax(debiased) vs E[q] under debiased probs. usage: mkt_agg.py CKPT [n_val]"""
import sys, os, glob, random
import numpy as np, torch
sys.path.insert(0, os.path.join(os.getcwd(), "rl"))
from model3 import Policy3
from model import market_targets_split
from features import MKT_BUCKETS
from act_common import MKT_POS_W
from actions import MKT_NAMES

ck = sys.argv[1]; nval = int(sys.argv[2]) if len(sys.argv) > 2 else 85
files = sorted(glob.glob("tmp/rl/majkel_all/*.npz")); random.Random(0).shuffle(files); files = files[:nval]
m = Policy3(); m.load_state_dict(torch.load(ck, map_location="cpu")); m.eval()
HEADS = [("sell", 0, 9), ("buyp", 9, 11), ("seed", 11, 16), ("anim", 16, 19), ("hire", 19, 20), ("land", 20, 21)]
def values(head):
    return np.array(MKT_BUCKETS, float) if head in ("sell", "buyp", "seed") else np.arange({"anim": 5, "hire": 13, "land": 2}[head], dtype=float)
tot = {k: np.zeros(21) for k in ("label", "argmax", "argmax_deb", "E_deb", "E_raw", "steps_label", "steps_argmax", "steps_deb", "P_deb")}
for f in files:
    d = np.load(f)
    T = len(d["mkt"])
    with torch.no_grad():
        for s in range(0, T, 256):
            sl = slice(s, min(T, s + 256))
            h = m.encode(*(torch.from_numpy(d[k][sl].astype(np.int16) if k in ("tiles", "units") else d[k][sl]) for k in ("tiles", "units", "items", "glob")))
            mk = m.market(h)
            lab = d["mkt"][sl].astype(int)
            for head, a, b in HEADS:
                lg = mk[head].numpy().reshape(lab.shape[0], b - a, -1)
                v = values(head)
                deb = lg.copy(); deb[..., 1:] -= np.log(MKT_POS_W[head])
                p_raw = np.exp(lg - lg.max(-1, keepdims=True)); p_raw /= p_raw.sum(-1, keepdims=True)
                p_deb = np.exp(deb - deb.max(-1, keepdims=True)); p_deb /= p_deb.sum(-1, keepdims=True)
                L = lab[:, a:b]
                tot["label"][a:b] += v[L].sum(0)
                tot["argmax"][a:b] += v[lg.argmax(-1)].sum(0)
                tot["argmax_deb"][a:b] += v[deb.argmax(-1)].sum(0)
                tot["E_deb"][a:b] += (p_deb * v).sum(-1).sum(0)
                tot["E_raw"][a:b] += (p_raw * v).sum(-1).sum(0)
                tot["steps_label"][a:b] += (L > 0).sum(0)
                tot["steps_argmax"][a:b] += (lg.argmax(-1) > 0).sum(0)
                tot["steps_deb"][a:b] += (deb.argmax(-1) > 0).sum(0)
                tot["P_deb"][a:b] += (1 - p_deb[..., 0]).sum(0)
n = len(files)
print(f"{ck}  n={n} games (per-game totals of quantity; [steps with an order])")
print(f"{'item':16s} {'label':>8s} {'argmax':>8s} {'arg_deb':>8s} {'E_deb':>8s} {'E_raw':>8s} | {'st_lab':>6s} {'st_arg':>6s} {'st_deb':>6s} {'ΣP_deb':>6s}")
for j, name in enumerate(MKT_NAMES):
    print(f"{name:16s} {tot['label'][j]/n:8.1f} {tot['argmax'][j]/n:8.1f} {tot['argmax_deb'][j]/n:8.1f} {tot['E_deb'][j]/n:8.1f} {tot['E_raw'][j]/n:8.1f} | "
          f"{tot['steps_label'][j]/n:6.1f} {tot['steps_argmax'][j]/n:6.1f} {tot['steps_deb'][j]/n:6.1f} {tot['P_deb'][j]/n:6.1f}")
