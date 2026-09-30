"""テープ木エージェントの leave-one-out 評価 (相手は生きたエージェント)。各記録局の世界 (seed+ショップ列) で対戦。
  TR_LIB=lib.json python tests/tr/loo2.py agents/t001/main.py --opp agents/e090/main.py [--eids a,b] [--out f]
"""
import argparse, json, os, sys, time, statistics as S
sys.path.insert(0, 'tests')
import kagsim
from kag_eval import load, fresh, PASS
ap = argparse.ArgumentParser(); ap.add_argument('agent'); ap.add_argument('--opp', required=True); ap.add_argument('--eids'); ap.add_argument('--out')
ap.add_argument('--keep', action='store_true', help='自局を除外しない (上限の確認)')
a = ap.parse_args()
cand, cmod = load(a.agent); opp, omod = load(a.opp)
lib = json.load(open(os.environ['TR_LIB']))
want = set(int(x) for x in a.eids.split(',')) if a.eids else None
rows = []
for tp in lib['tapes']:
    if want and tp['eid'] not in want: continue
    s = tp['s']
    cmod._EXCLUDE = set() if a.keep else {tp['eid']}; cmod._STATE.clear(); fresh(omod)
    g = kagsim.Game(tp['seed'], 720, tp['shops'])
    while not g.done:
        acts = [None, None]; acts[s] = cand(g.observe(s)); acts[1 - s] = opp(g.observe(1 - s))
        g.step(acts[0], acts[1])
    used = sorted(set(x['eid'] for x in [cmod._STATE[s]['tape']] if x))
    row = dict(eid=tp['eid'], s=s, shops=tp['shops'][:3], m=g.reward(s) - g.reward(1 - s), own=g.reward(s), opp=g.reward(1 - s), final_tape=used)
    rows.append(row); print(json.dumps(row), flush=True); time.sleep(0.3)
v = [r['m'] for r in rows]
print(f"n={len(v)} mean {S.mean(v):+.0f} se {S.stdev(v)/len(v)**.5 if len(v)>1 else 0:.0f} win {sum(x>0 for x in v)}/{len(v)} own {S.mean(r['own'] for r in rows):.0f}")
if a.out: json.dump(rows, open(a.out, 'w'))
