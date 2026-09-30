"""Low-CPU live-opponent paired-seat check with pinned shop worlds."""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

import kagsim
from swap_eval import load

SHOPS=('BAKERY','BRUNCH_SPOT','FARMERS_MARKET','ICE_CREAM_SHOP',
       'PET_CAFE','PIZZA_SHOP','SMOOTHIE_SHOP','YARN_STORE')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--candidate',required=True);ap.add_argument('--opponent',required=True)
    ap.add_argument('--set',default='{}');ap.add_argument('--n',type=int,default=4)
    ap.add_argument('--seed0',type=int,default=937000);ap.add_argument('--out',required=True)
    args=ap.parse_args()
    a=load(str(Path(args.candidate).resolve()));b=load(str(Path(args.opponent).resolve()))
    for key,value in json.loads(args.set).items():
        assert hasattr(a,key),key
        setattr(a,key,value)
    rows=[]
    for seed in range(args.seed0,args.seed0+args.n):
        rng=random.Random(seed);shops=[rng.choice(SHOPS) for _ in range(8)]
        for seat in (0,1):
            time.sleep(2)
            g=kagsim.Game(seed,720,shops)
            cpu0=time.process_time();wall0=time.monotonic();worst=0
            while not g.done:
                if g.step_count%24==0:
                    delay=(time.process_time()-cpu0)/.30-(time.monotonic()-wall0)
                    if delay>0:time.sleep(delay)
                start=time.perf_counter();mine=a.agent(g.observe(seat));worst=max(worst,time.perf_counter()-start)
                other=b.agent(g.observe(1-seat))
                g.step(*((mine,other) if seat==0 else (other,mine)))
            row=dict(seed=seed,seat=seat,shops=shops,margin=g.reward(seat)-g.reward(1-seat),
                     max_act=worst,report=getattr(a,'_SH_REPORT',{}).copy())
            rows.append(row);Path(args.out).write_text(json.dumps(rows))
            print(seed,seat,row['margin'],row['report'],flush=True)
    pairs=[statistics.mean(r['margin'] for r in rows if r['seed']==seed)
           for seed in range(args.seed0,args.seed0+args.n)]
    se=statistics.stdev(pairs)/len(pairs)**.5 if len(pairs)>1 else 0
    mean=statistics.mean(pairs)
    print('W/L/T',sum(r['margin']>0 for r in rows),sum(r['margin']<0 for r in rows),sum(r['margin']==0 for r in rows),
          'pair mean',mean,'SE',se,'t',mean/se if se else 0)


if __name__=='__main__':main()
