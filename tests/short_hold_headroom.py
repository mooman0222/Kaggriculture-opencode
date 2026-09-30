"""Diagnostic one-turn sale deferral across a known town-consumption tick.

Replays each baseline game once. The independent per-item SELL model reproduces
actual per-item revenues before scoring a one-turn, inventory-neutral change.
Opponent orders are an oracle used only to estimate headroom, not a live policy.
Field inflows and all other baseline orders are frozen. Cash and cross-item
effects are not modeled; any candidate still needs closed-loop testing.
"""
import argparse
import glob
import json
import random
import statistics
import time
from pathlib import Path

import kagsim
from swap_eval import episode, load

ITEMS=('MILK','WOOL','STRAWBERRY')


def simulate(base,item,inv,stocks,queues,params):
    stocks=list(stocks)
    revenue=[0,0]
    for i in range(max(map(len,queues))):
        rem=[0,0]
        for p in (0,1):
            if i<len(queues[p]):
                o=queues[p][i]
                if len(o)>=3 and o[:2]==['SELL',item]:
                    rem[p]=min(stocks[p],max(0,int(o[2])))
        while any(rem):
            price=base._r37_market_price(item,inv,params)
            for p in (0,1):
                if rem[p]:
                    rem[p]-=1;stocks[p]-=1;revenue[p]+=price
                    if price>1:inv+=1
    return inv,stocks,revenue


def cut(orders,item,q,available):
    result=[list(o) for o in orders]
    left=min(available,sum(max(0,int(o[2])) for o in orders
             if len(o)>=3 and o[:2]==['SELL',item]))-q
    assert left>=0
    for o in result:
        if len(o)>=3 and o[:2]==['SELL',item]:
            o[2]=min(left,max(0,int(o[2])))
            left-=o[2]
    return result


def add(orders,item,q):
    result=[list(o) for o in orders]
    for o in result:
        if len(o)>=3 and o[:2]==['SELL',item]:
            o[2]+=q
            return result
    return [['SELL',item,q]]+result if len(result)<10 else None


def play(ep,base):
    time.sleep(2)
    g=kagsim.Game(ep['seed'],720,ep['shops'])
    me=ep['me']; frames=[]
    cpu_start=time.process_time();wall_start=time.monotonic()
    def pace():
        # Keep sustained use near 30% of one core, in addition to per-game rest.
        delay=(time.process_time()-cpu_start)/.30-(time.monotonic()-wall_start)
        if delay>0:time.sleep(delay)
    while not g.done:
        t=g.step_count
        if t%24==0:pace()
        obs=g.observe(me);opp_obs=g.observe(1-me)
        a=base.agent(obs);b=ep['tape'][t]
        before=[base.projected_shed(a,base.FarmView(obs)),
                base.projected_shed(b,base.FarmView(opp_obs))]
        rev0=[g.telemetry(p)['sell_revenue_items'] for p in (me,1-me)]
        g.step(*((a,b) if me==0 else (b,a)))
        rev1=[g.telemetry(p)['sell_revenue_items'] for p in (me,1-me)]
        frames.append(dict(step=t,inv=obs['market']['inventory'],before=before,
                           after=[g.observe(p)['private']['shed'] for p in (me,1-me)],
                           orders=[a.get('market') or [],b.get('market') or []],
                           money=obs['farms'][me]['money'],
                           draw=base._race_town(t,obs['town']['unlocked_shops']),
                           revenue=[{it:rev1[p][it]-rev0[p][it] for it in ITEMS} for p in (0,1)],
                           params=base._v44y_params(obs)))
    events=[]; mismatches=0; checks=0
    for t in range(144,695,4):
        pace()
        now,nxt=frames[t:t+2]
        if now['money']<5000:continue
        for item in ITEMS:
            original=min(now['before'][0].get(item,0),sum(int(o[2]) for o in now['orders'][0]
                         if len(o)>=3 and o[:2]==['SELL',item]))
            if original<=0 or now['draw'].get(item,0)<=0:continue
            start=[s.get(item,0) for s in now['before']]
            inv,end,rev=simulate(base,item,now['inv'][item],start,now['orders'],now['params'])
            ni,ne,nr=simulate(base,item,nxt['inv'][item],[s.get(item,0) for s in nxt['before']],nxt['orders'],nxt['params'])
            checks+=1
            if any(rev[p]!=now['revenue'][p][item] or nr[p]!=nxt['revenue'][p][item] for p in (0,1)):
                mismatches+=1;continue
            best=0;best_q=0;small_gain=None
            for q in sorted({1,min(original,now['draw'].get(item,0)),max(1,original//2),original}):
                if sum(now['after'][0].values())+q>85 or sum(nxt['before'][0].values())+q>95:continue
                delayed=add(nxt['orders'][0],item,q)
                if delayed is None:continue
                ci,ce,cr=simulate(base,item,now['inv'][item],start,
                                  [cut(now['orders'][0],item,q,start[0]),now['orders'][1]],now['params'])
                inflow=[nxt['before'][p].get(item,0)-end[p] for p in (0,1)]
                fi,fe,fr=simulate(base,item,ci-now['draw'].get(item,0),
                                  [ce[p]+inflow[p] for p in (0,1)],
                                  [delayed,nxt['orders'][1]],nxt['params'])
                if fe!=ne or fi!=ni:continue
                gain=cr[0]-cr[1]+fr[0]-fr[1]-(rev[0]-rev[1]+nr[0]-nr[1])
                if q==min(original,now['draw'].get(item,0)):small_gain=gain
                if gain>best:best,best_q=gain,q
            events.append(dict(step=t,item=item,original=original,draw=now['draw'].get(item,0),
                               gain=best,q=best_q,small_gain=small_gain))
    return dict(eid=ep['eid'],seat=me,events=events,checks=checks,mismatches=mismatches,
                oracle_gain=sum(e['gain'] for e in events),
                small_gain=sum(e['small_gain'] or 0 for e in events))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=8);ap.add_argument('--out',required=True)
    args=ap.parse_args();base=load(str(Path('agents/e090/main.py').resolve()))
    files=sorted(glob.glob('tmp/band0929_slim/episode-*.json'));random.Random(936).shuffle(files)
    rows=[]
    for file in files[:args.n]:
        for ep in episode(file,None):
            if ep['eid'] in (114986715,114998678,115004454):continue
            row=play(ep,base);rows.append(row);Path(args.out).write_text(json.dumps(rows))
            print(row['eid'],row['seat'],{k:row[k] for k in ('oracle_gain','small_gain','checks','mismatches')},flush=True)
    for key in ('oracle_gain','small_gain'):
        x=[r[key] for r in rows];se=statistics.stdev(x)/len(x)**.5
        print(key,'n',len(x),'mean',statistics.mean(x),'SE',se,'t',statistics.mean(x)/se if se else 0)


if __name__=='__main__':main()
