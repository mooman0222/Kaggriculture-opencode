"""チーム別リプレイ群の系統 (開幕24/72手ハッシュ) と試合間プラン一致率。使い方: .venv/bin/python tests/xray.py tmp/top0911/replays"""
import json, sys, hashlib
from collections import Counter, defaultdict
from pathlib import Path
PASS = {"farmer": ["PASS"], "hands": [], "market": []}
KNOWN = {'267f5f1e': 'SR0908', 'b3105bcc': 'SR0909', '0b496ee6': 'ThomasOpen', '87b9b0d1': 'KaitoOpen'}
def plan(a):
    a = a or PASS
    return (tuple(a.get("farmer") or ["PASS"]), tuple(tuple(h) for h in (a.get("hands") or [])))
root = Path(sys.argv[1])
for d in sorted(root.iterdir()):
    if not d.is_dir(): continue
    tname = d.name.split("_", 1)[1].replace("_", " ")
    eps = []
    for f in sorted(d.glob("episode-*.json")):
        r = json.load(open(f)); names = r["info"]["TeamNames"]; steps = r["steps"]
        if len(steps) < 720: continue
        me = next((i for i, n in enumerate(names) if n.replace(" ", "_") == d.name.split("_", 1)[1] or n == tname), None)
        if me is None: continue
        st = [steps[t + 1][me].get("action") for t in range(719)]
        h = lambda n: hashlib.sha256(json.dumps([plan(x) for x in st[:n]]).encode()).hexdigest()[:8]
        eps.append(dict(st=st, h24=h(24), h72=h(72), rew=r["rewards"], me=me, opp=names[1 - me]))
    if not eps: continue
    def agree(lo, hi):
        tot = n = 0
        for i in range(len(eps)):
            for j in range(i + 1, len(eps)):
                for t in range(lo * 24, min(hi * 24, 719)):
                    n += 1; tot += plan(eps[i]["st"][t]) == plan(eps[j]["st"][t])
        return tot / n if n else float("nan")
    wins = sum(1 for e in eps if e["rew"][e["me"]] > e["rew"][1 - e["me"]])
    h72s = Counter(e["h72"] for e in eps)
    lab = ",".join(f"{KNOWN.get(k, k)}x{v}" for k, v in h72s.items())
    print(f"{tname:<22} n={len(eps)} W{wins}L{len(eps)-wins} h72=[{lab}] plan-agree " + " ".join(f"d{a}-{b}:{agree(a,b):.2f}" for a, b in [(0,3),(3,6),(6,12),(12,18),(18,24),(24,30)]) + f" money={sum(e['rew'][e['me']] for e in eps)/len(eps):,.0f}")
