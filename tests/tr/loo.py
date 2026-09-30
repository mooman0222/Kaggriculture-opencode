"""テープ切替エージェントの leave-one-out 評価: 各記録局の世界 (seed+ショップ列) で、記録相手テープと対戦。
  TR_LIB=tmp/tr/lib_fq.json python tests/tr/loo.py agents/t001/main.py tmp/fq_slim [--base agents/e090/main.py] [--n 20]
"""
import argparse, json, os, sys, time, statistics as S
sys.path.insert(0, 'tests')
import kagsim
from kag_eval import load, fresh, PASS
ap = argparse.ArgumentParser(); ap.add_argument('agent'); ap.add_argument('dir'); ap.add_argument('--base'); ap.add_argument('--n', type=int, default=999)
ap.add_argument('--out')
a = ap.parse_args()
cand, cmod = load(a.agent)
base, bmod = load(a.base) if a.base else (None, None)
lib = json.load(open(os.environ['TR_LIB']))
rows = []
def play(seed, shops, s, me, mod, opp_tape):
    if mod is not None: fresh(mod)
    g = kagsim.Game(seed, 720, shops)
    while not g.done:
        t = g.step_count
        acts = [None, None]; acts[s] = me(g.observe(s)); acts[1 - s] = opp_tape[t]
        g.step(acts[0], acts[1])
    return g.reward(s) - g.reward(1 - s), g.reward(s)
for tp in lib['tapes'][:a.n]:
    r = json.load(open(f"{a.dir}/episode-{tp['eid']}-replay.json")); st = r['steps']; s = tp['s']
    opp = [st[t + 1][1 - s].get('action') or PASS for t in range(719)]
    cmod._EXCLUDE = {tp['eid']}; cmod._STATE.clear()
    m, own = play(tp['seed'], tp['shops'], s, cand, cmod, opp)
    row = dict(eid=tp['eid'], s=s, shops=tp['shops'][:2], rec=tp['own'] - tp['opp'], cand=m, cand_own=own)
    if base:
        row['base'], row['base_own'] = play(tp['seed'], tp['shops'], s, base, bmod, opp)
    rows.append(row)
    print(json.dumps(row), flush=True)
    time.sleep(0.5)
def summ(k):
    v = [r[k] for r in rows]; return f"{k}: mean {S.mean(v):+8.0f} se {S.stdev(v)/len(v)**.5 if len(v)>1 else 0:6.0f} win {sum(x>0 for x in v)}/{len(v)}"
print(summ('rec')); print(summ('cand'))
if base:
    print(summ('base')); d = [r['cand'] - r['base'] for r in rows]
    print(f"cand-base: {S.mean(d):+8.0f} se {S.stdev(d)/len(d)**.5:6.0f}")
if a.out: json.dump(rows, open(a.out, 'w'))
