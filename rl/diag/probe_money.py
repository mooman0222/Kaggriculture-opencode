"""Minority-branch gate for raw-BC ckpts (no GPU/game needed, open-loop on shards).
t168 money>=1400 (SELL/HIRE minority) + t144 money<1000 (hiring-morning wheat sale):
agreement vs teacher labels, overall and per band. Usage:
  PYTHONPATH=.venv/lib/python3.14/site-packages /usr/bin/python3 rl/diag/probe_money.py CKPT 'glob1' ['glob2' ...]
"""
import sys, glob as g, os
sys.path.insert(0, os.path.join(os.getcwd(), "rl"))
import numpy as np, torch
from raw import Policy4, KINDS, KIND_INDEX
ckpt = sys.argv[1]
files = [f for pat in sys.argv[2:] for f in sorted(g.glob(pat)) if os.path.getsize(f) > 0]
model = Policy4(); model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True); model.eval()
SF, HIRE = KIND_INDEX["SELL_FERTILIZER"], KIND_INDEX["HIRE"]
res = []
with torch.no_grad():
    for f in files:
        a = np.load(f)
        for t in (144, 168):
            ins = (torch.from_numpy(a["tiles"][t:t+1]).float(), torch.from_numpy(a["units"][t:t+1]).float(),
                   torch.from_numpy(a["items"][t:t+1]), torch.from_numpy(a["glob"][t:t+1]))
            p = model(*ins)["mkind"][0, 0].softmax(-1)
            money = float(np.expm1(float(a["glob"][t, 6]) * 12.0))
            res.append((t, money, int(p.argmax(-1)), int(a["mk"][t, 0])))
res = np.array(res)
print(f"{ckpt}: n={len(res)} shards={len(files)}")
for t in (144, 168):
    m = res[:, 0] == t
    print(f" t={t}: agree={(res[m,2]==res[m,3]).mean():.3f}")
    for lo, hi in ((0, 1000), (1000, 1400), (1400, 1e9)):
        mm = m & (res[:, 1] >= lo) & (res[:, 1] < hi)
        if mm.sum() > 5:
            print(f"   money[{lo},{hi}): n={mm.sum():.0f} agree={(res[mm,2]==res[mm,3]).mean():.3f}")
