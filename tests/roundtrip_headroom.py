"""Read-only estimate of inventory-neutral fertilizer roundtrip opportunity."""
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
    while not g.done:
        t = g.step_count
        obs = g.observe(me)
        a = BASE.agent(obs)
        opp = ep['tape'][t]
        orders = a.get('market') or []
        stock = BASE.projected_shed(a, BASE.FarmView(obs))
        q = max(0, stock.get('FERTILIZER', 0)-sum(int(o[2]) for o in orders
                 if len(o)>2 and o[:2]==['SELL','FERTILIZER']))
        q = min(q, 40)
        valid = (t>=144 and t<696 and t%24!=23 and len(orders)<=8
                 and obs['farms'][me]['money']>3000
                 and obs['market']['prices']['FERTILIZER']>10 and q>0)
        if valid:
            other = BASE.projected_shed(opp, BASE.FarmView(g.observe(1-me)))
        g.step(*((a,opp) if me==0 else (opp,a)))
        if valid:
            net = other.get('FERTILIZER',0)-g.observe(1-me)['private']['shed'].get('FERTILIZER',0)
            gain = .2*q*net
            signed += gain
            upper += max(0,gain)
            eligible += 1
    return dict(eid=ep['eid'],seat=me,signed_estimate=signed,oracle_estimate=upper,eligible=eligible)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--n',type=int,default=12)
    ap.add_argument('--out',required=True)
    a=ap.parse_args()
    files=sorted(glob.glob('tmp/band0929_slim/episode-*.json'))
    random.Random(932).shuffle(files)
    eps=[e for f in files[:a.n] for e in episode(f,None)]
    with Pool(1,initializer=init) as p: rows=list(p.imap_unordered(run,eps))
    Path(a.out).write_text(json.dumps(rows))
    for k in ('signed_estimate','oracle_estimate','eligible'):
        vals=[r[k] for r in rows]
        print(k,statistics.mean(vals), 'SE',statistics.stdev(vals)/len(vals)**.5)
