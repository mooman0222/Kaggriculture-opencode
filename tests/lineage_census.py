"""自提出のリプレイ群について、相手の系統 (開幕72手ハッシュ) 別に勝敗と margin を集計する。

使い方: .venv/bin/python tests/lineage_census.py --dir tmp/top0911/mine_e054/0_MMN0222 --team MMN0222 [--label HASH=NAME ...]
"""
import argparse, glob, hashlib, json
from collections import defaultdict
from pathlib import Path
PASS = {"farmer": ["PASS"], "hands": [], "market": []}
KNOWN = {'267f5f1e': 'SR0908', 'b3105bcc': 'SR0908v', '0b496ee6': 'ThomasOpen', '87b9b0d1': 'KaitoOpen', '149b67ca': '7bf8'}

def plan(a):
    a = a or PASS
    return (tuple(a.get("farmer") or ["PASS"]), tuple(tuple(h) for h in (a.get("hands") or [])))

def h(steps, seat, n=72):
    return hashlib.sha256(json.dumps([plan(steps[t + 1][seat].get("action")) for t in range(n)]).encode()).hexdigest()[:8]

ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--team", default="MMN0222"); ap.add_argument("--label", nargs="*", default=[])
a = ap.parse_args(); KNOWN.update(dict(x.split("=") for x in a.label))
agg = defaultdict(list); rows = []
for f in sorted(glob.glob(f"{a.dir}/episode-*.json")):
    r = json.load(open(f)); names = r["info"]["TeamNames"]; steps = r["steps"]
    if a.team not in names or len(steps) < 720: continue
    me = names.index(a.team); op = 1 - me
    hh = h(steps, op); lin = KNOWN.get(hh, hh)
    same_pos = sum(1 for t in range(1, 73) if [steps[t][0]["observation"]["farms"][me]["farmer"], *steps[t][0]["observation"]["farms"][me]["hands"]] == [steps[t][0]["observation"]["farms"][op]["farmer"], *steps[t][0]["observation"]["farms"][op]["hands"]])
    m = r["rewards"][me] - r["rewards"][op]
    agg[lin].append((m, names[op], same_pos, r["info"]["EpisodeId"]))
    rows.append((r["info"]["EpisodeId"], lin, names[op], m, same_pos))
for eid, lin, opp, m, sp in rows:
    print(f"{eid} {lin:<12} {opp[:22]:<22} {m:+8.0f} clone_pos {sp}/72")
print("== by opponent lineage")
for lin, v in sorted(agg.items(), key=lambda x: -len(x[1])):
    w = sum(1 for m, *_ in v if m > 0)
    print(f"  {lin:<12} n={len(v):2d} W{w} L{len(v)-w} mean {sum(m for m,*_ in v)/len(v):+7.0f}  clone_pos avg {sum(s for _,_,s,_ in v)/len(v):.0f}  opps {sorted(set(o[:14] for _,o,_,_ in v))[:6]}")
