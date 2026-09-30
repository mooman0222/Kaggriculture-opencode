"""Diagnostic only: exact per-item market model of inventory-neutral churn.

Uses the opponent's recorded current action as an oracle, never a live feature.
No changes to game actions. The market model ignores opponent cash and capacity;
closed-loop simulator evaluation is required before interpreting it as profit.
"""
import argparse
import glob
import json
import random
import statistics
import time
from multiprocessing import Pool
from pathlib import Path

import kagsim
from swap_eval import episode, load

BASE = None


def init():
    global BASE
    BASE = load(str(Path('agents/e090/main.py').resolve()))


def run(ep):
    time.sleep(2)
    g = kagsim.Game(ep['seed'], 720, ep['shops'])
    me = ep['me']
    signed = upper = 0.0
    eligible = 0
    by_hour = {}
    while not g.done:
        t = g.step_count
        obs = g.observe(me)
        a = BASE.agent(obs)
        opp = ep['tape'][t]
        orders = a.get('market') or []
        stock = BASE.projected_shed(a, BASE.FarmView(obs))
        q = min(60, max(0, 100-sum(stock.values())))
        inv = obs['market']['inventory']['WHEAT']
        safe_cash = sum(BASE._r37_market_price('WHEAT', inv-2*j-1) for j in range(q))
        valid = (t>=144 and t<696 and t%24!=23 and len(orders)<=8
                 and obs['farms'][me]['money']>safe_cash+5000 and q>0
                 and not any(len(o)>1 and (o[1]=='WHEAT' or o[0] in
                             ('BUY_PRODUCT','BUY_ANIMAL')) for o in orders))
        if valid:
            other = BASE.projected_shed(opp, BASE.FarmView(g.observe(1-me)))
            theirs = [o if len(o)>1 and o[1]=='WHEAT' else []
                      for o in (opp.get('market') or [])]
            mine = [[] for o in orders]
            params = {'WHEAT': BASE._v44y_params(obs)['WHEAT']}
            args = ({'WHEAT':inv}, {'WHEAT':stock.get('WHEAT',0)},
                    {'WHEAT':other.get('WHEAT',0)}, params)
            b = BASE._v44y_lockstep(mine, theirs, *args)
            c = BASE._v44y_lockstep([['BUY_PRODUCT','WHEAT',q]]+mine+
                                  [['SELL','WHEAT',q]], theirs, *args)
            gain = c[0]-c[1] - b[0]+b[1]
            signed += gain
            upper += max(0,gain)
            by_hour[t%24] = by_hour.get(t%24,0)+gain
            eligible += 1
        g.step(*((a,opp) if me==0 else (opp,a)))
    return dict(eid=ep['eid'],seat=me,signed_estimate=signed,
                oracle_estimate=upper,eligible=eligible,by_hour=by_hour)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--n',type=int,default=12)
    ap.add_argument('--out',required=True)
    a=ap.parse_args()
    files=sorted(glob.glob('tmp/band0929_slim/episode-*.json'))
    random.Random(933).shuffle(files)
    eps=[e for f in files[:a.n] for e in episode(f,None)
         if e['eid'] not in (114986715,114998678,115004454)]
    with Pool(1,initializer=init) as p: rows=list(p.imap_unordered(run,eps))
    Path(a.out).write_text(json.dumps(rows))
    for k in ('signed_estimate','oracle_estimate','eligible'):
        vals=[r[k] for r in rows]
        print(k,statistics.mean(vals), 'SE',statistics.stdev(vals)/len(vals)**.5)
    print('hour means', {h:sum(r['by_hour'].get(h,0) for r in rows)/len(rows)
                         for h in range(24)})
