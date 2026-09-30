"""Read-only bound for extra GOOSE care/feed on inherited PASS commands.

No action is changed. Day-end unmet needs are intersected with same-day idle
visits. Prices are diagnostic quotes, not executable revenues. The estimate
assumes every extra egg can later be harvested and sold; hence optimistic.
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


def run(ep, base):
    time.sleep(2)
    g = kagsim.Game(ep['seed'], 720, ep['shops'])
    me = ep['me']
    opportunities = {}
    events = []
    goose_days = 0
    while not g.done:
        step = g.step_count
        obs = g.observe(me)
        action = base.agent(obs)
        farm = obs['farms'][me]
        positions = [farm['farmer'], *farm['hands']]
        commands = [action.get('farmer') or ['PASS'], *(action.get('hands') or [])]
        for i, (pos, cmd) in enumerate(zip(positions, commands)):
            x, y = pos
            tile = farm['tiles'][y][x]
            if not (isinstance(tile, dict) and tile.get('animal') == 'GOOSE'
                    and (cmd or ['PASS'])[0] == 'PASS'):
                continue
            item = opportunities.setdefault((x,y), dict(care=[],feed=[]))
            if not tile.get('cared_today'):
                item['care'].append(step)
            if not tile.get('fed_today') and obs['private']['inventories'][i].get('WHEAT',0)>0:
                item['feed'].append(step)
        if step % 24 == 23:
            projected, _ = base._r127_fields(obs, action)
            day = step // 24
            for y, row in enumerate(projected['tiles']):
                for x, tile in enumerate(row):
                    if not (isinstance(tile, dict) and tile.get('animal')=='GOOSE'):
                        continue
                    goose_days += 1
                    opportunity = opportunities.get((x,y), {})
                    feed = not tile.get('fed_today') and bool(opportunity.get('feed'))
                    care = not tile.get('cared_today') and bool(opportunity.get('care'))
                    if not (feed or care):
                        continue
                    # CARE accrues after today's production; it cannot pay until
                    # the following dawn. There is no final refresh at step 719.
                    mature = day+1 >= int(tile['placed_day'])+4
                    bonus = min(int(tile.get('pending_care_bonus',0)),
                                max(0, 4-int(tile.get('yield_units',0))-1)) if feed and mature else 0
                    care_bonus = int(day<=27 and (feed or tile.get('fed_today'))
                                     and (care or (feed and tile.get('cared_today'))))
                    eggs = bonus + care_bonus
                    price = obs['market']['prices']['EGG']
                    cost = obs['market']['prices']['WHEAT'] if feed else 0
                    events.append(dict(day=day,site=[x,y],feed=feed,care=care,
                                       eggs=eggs,gross=eggs*price,cost=cost,
                                       quoted_net=eggs*price-cost,
                                       pending=tile.get('pending_care_bonus',0),
                                       fed=tile.get('fed_today'),cared=tile.get('cared_today'),
                                       opportunities=opportunity))
            opportunities = {}
        other = ep['tape'][step]
        g.step(*((action,other) if me==0 else (other,action)))
    return dict(eid=ep['eid'],seat=me,goose_days=goose_days,events=events,
                quoted_net=sum(e['quoted_net'] for e in events),
                positive_net=sum(max(0,e['quoted_net']) for e in events),
                eggs=sum(e['eggs'] for e in events),
                feeds=sum(e['feed'] for e in events),cares=sum(e['care'] for e in events))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--n',type=int,default=8)
    ap.add_argument('--out',required=True)
    ap.add_argument('--goose-only',action='store_true',help='select previously measured E090 seats selling >=100 eggs')
    args=ap.parse_args()
    base=load(str(Path('agents/e090/main.py').resolve()))
    files=sorted(glob.glob('tmp/band0929_slim/episode-*.json'))
    random.Random(935).shuffle(files)
    eligible=None
    if args.goose_only:
        cached=json.loads(Path('tmp/route_candidate0929_result.json').read_text())
        eligible={(r['eid'],r['seat']) for r in cached['rows'] if r['ci']==0
                  and r['telemetry']['sold_units_items'].get('EGG',0)>=100}
    rows=[]
    used=0
    for file in files:
        if used>=args.n:break
        selected=[ep for ep in episode(file,None)
                  if eligible is None or (ep['eid'],ep['me']) in eligible]
        if not selected:continue
        used+=1
        for ep in selected:
            if ep['eid'] in (114986715,114998678,115004454):continue
            row=run(ep,base)
            rows.append(row)
            Path(args.out).write_text(json.dumps(rows))
            print(row['eid'],row['seat'], {k:row[k] for k in
                  ('goose_days','eggs','feeds','cares','quoted_net')},flush=True)
    for key in ('eggs','feeds','cares','quoted_net','positive_net'):
        vals=[r[key] for r in rows]
        mean=statistics.mean(vals)
        se=statistics.stdev(vals)/len(vals)**.5 if len(vals)>1 else 0
        print(key, 'n',len(vals),'mean',round(mean,2),'SE',round(se,2),
              't',round(mean/se,2) if se else 0)


if __name__=='__main__':
    main()
