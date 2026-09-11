"""自提出の敗戦を解剖: 相手系統・世界・margin・家畜数・kagsim テレメトリ (拒否購入/廃棄)・日別差の崩れ点。
使い方: .venv/bin/python tests/loss_probe.py --dir tmp/top0911/mine_e056/0_MMN0222 [--team MMN0222] [--all]"""
import argparse, glob, hashlib, json
from collections import Counter, defaultdict
import kagsim
PASS = {"farmer": ["PASS"], "hands": [], "market": []}
KNOWN = {'267f5f1e': 'SR0908', 'b3105bcc': 'SR0909', '0b496ee6': 'ThomasOpen', '87b9b0d1': 'KaitoOpen', 'a17674d6': 'Griefer'}
def plan(a):
    a = a or PASS; return (tuple(a.get("farmer") or ["PASS"]), tuple(tuple(h) for h in (a.get("hands") or [])))
def h72(steps, s): return hashlib.sha256(json.dumps([plan(steps[t + 1][s].get("action")) for t in range(72)]).encode()).hexdigest()[:8]
def animals(farm): return sum(1 for r in farm["tiles"] for t in r if isinstance(t, dict) and "animal" in t)
ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--team", default="MMN0222"); ap.add_argument("--all", action="store_true")
a = ap.parse_args()
rows = []; agg = defaultdict(list)
for f in sorted(glob.glob(f"{a.dir}/episode-*.json")):
    r = json.load(open(f)); names = r["info"]["TeamNames"]; steps = r["steps"]
    if a.team not in names or len(steps) < 720: continue
    me = names.index(a.team); op = 1 - me; m = r["rewards"][me] - r["rewards"][op]
    lin = KNOWN.get(h72(steps, op), "other"); agg[lin].append(m)
    if m >= 0 and not a.all: continue
    A = [steps[t + 1][0].get("action") or PASS for t in range(719)]; B = [steps[t + 1][1].get("action") or PASS for t in range(719)]
    g = kagsim.Game(r["info"]["seed"])
    while not g.done: g.step(A[g.step_count], B[g.step_count])
    tel = {k: v for k, v in g.telemetry(me).items() if v and (k.startswith("refused") or k == "shed_discarded_units")}
    otel = {k: v for k, v in g.telemetry(op).items() if v and (k.startswith("refused") or k == "shed_discarded_units")}
    diff = [steps[d * 24][me]["observation"]["farms"][me]["money"] - steps[d * 24][op]["observation"]["farms"][op]["money"] for d in range(30)] + [m]
    drops = sorted(((diff[d + 1] - diff[d], d) for d in range(30)))[:2]
    same_pos = sum(1 for t in range(1, 73) if [steps[t][0]["observation"]["farms"][me]["farmer"], *steps[t][0]["observation"]["farms"][me]["hands"]] == [steps[t][0]["observation"]["farms"][op]["farmer"], *steps[t][0]["observation"]["farms"][op]["hands"]])
    world = "/".join(s[:5] for s in steps[-1][0]["observation"]["town"]["unlocked_shops"][:2])
    print(f"{r['info']['EpisodeId']} {'L' if m<0 else 'W'} {m:+7.0f} vs {names[op][:16]:<16} {lin:<10} clone{same_pos:>2}/72 world {world:<12} animals {animals(steps[-1][me]['observation']['farms'][me])}/{animals(steps[-1][op]['observation']['farms'][op])} "
          f"worst-day-drops {[(int(x), d) for x, d in drops]} me_tel {tel} opp_tel {otel}")
print("== by lineage: n W-L mean")
for k, v in sorted(agg.items(), key=lambda x: -len(x[1])):
    print(f"  {k:<10} n={len(v):2d} W{sum(1 for x in v if x>0)} L{sum(1 for x in v if x<0)} mean {sum(v)/len(v):+7.0f}")
