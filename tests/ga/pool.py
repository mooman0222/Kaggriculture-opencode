"""相手プール構築: リプレイから (相手の行動列, seed, world, 系統) を抽出して tmp/ga/pool.json に保存。

使い方: .venv/bin/python tests/ga/pool.py --glob 'tmp/top0911/**/episode-*.json' --exclude-team MMN0222 --out tmp/ga/pool.json
"""
import argparse, glob, hashlib, json
PASS = {"farmer": ["PASS"], "hands": [], "market": []}
KNOWN = {'267f5f1e': 'SR0908', 'b3105bcc': 'SR0909', '0b496ee6': 'ThomasOpen', '87b9b0d1': 'KaitoOpen', 'a17674d6': 'Griefer'}

def plan(a):
    a = a or PASS
    return (tuple(a.get("farmer") or ["PASS"]), tuple(tuple(h) for h in (a.get("hands") or [])))

ap = argparse.ArgumentParser(); ap.add_argument("--glob", required=True); ap.add_argument("--exclude-team", default="MMN0222"); ap.add_argument("--out", required=True)
a = ap.parse_args()
pool = []; seen = set()
for f in sorted(glob.glob(a.glob, recursive=True)):
    r = json.load(open(f)); eid = r["info"].get("EpisodeId", f)
    if eid in seen or len(r["steps"]) < 720: continue
    seen.add(eid); steps = r["steps"]; names = r["info"]["TeamNames"]
    shops = steps[-1][0]["observation"]["town"]["unlocked_shops"]
    for s in (0, 1):
        if names[s] == a.exclude_team: continue
        stream = [steps[t + 1][s].get("action") or PASS for t in range(719)]
        h = hashlib.sha256(json.dumps([plan(x) for x in stream[:72]]).encode()).hexdigest()[:8]
        pool.append({"episode": eid, "seat": s, "team": names[s], "lineage": KNOWN.get(h, "other"), "h72": h,
                     "seed": r["info"]["seed"], "shops": shops, "bank": r["rewards"][s], "stream": stream})
json.dump(pool, open(a.out, "w"))
from collections import Counter
print(len(pool), "opponent streams", Counter(p["lineage"] for p in pool))
