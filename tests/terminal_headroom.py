"""Measure final-day unsold physical value before implementing a rescue policy."""
import argparse
import glob
import json
import random
import time
from multiprocessing import Pool
from pathlib import Path

import kagsim
from swap_eval import episode, load

BASE = None
PRODUCT = {'COW': 'MILK', 'SHEEP': 'WOOL', 'GOOSE': 'EGG'}
MATURE = {'WHEAT': 2, 'CARROT': 2, 'MELON': 10, 'TOMATO': 8, 'STRAWBERRY': 10}


def init(path):
    global BASE
    BASE = load(str(Path(path).resolve()))


def value(obs):
    farm = obs['farms'][obs['player']]
    prices = obs['market']['prices']
    goods = {}
    for bag in [obs['private']['shed'], *obs['private']['inventories']]:
        for item, q in bag.items():
            if item in prices:
                goods[item] = goods.get(item, 0) + q
    field = {}
    for row in farm['tiles']:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            item = PRODUCT.get(tile.get('animal')) or tile.get('crop')
            q = tile.get('yield_units', 0)
            if tile.get('crop') and 29-tile['planted_day'] < MATURE[item]:
                q = 0
            if q and item in prices:
                field[item] = field.get(item, 0) + q
            if tile.get('animal') and tile.get('fertilizer_available'):
                field['FERTILIZER'] = field.get('FERTILIZER', 0) + 1
    return {'goods': goods, 'field': field,
            'quoted_value': sum(prices.get(k, 0)*v for d in (goods, field) for k, v in d.items())}


def run(ep):
    time.sleep(2)
    g = kagsim.Game(ep['seed'], 720, ep['shops'])
    me = ep['me']
    snapshot = None
    before = 0
    while not g.done:
        t = g.step_count
        obs = g.observe(me)
        if t == 696:
            snapshot = obs
            before = g.telemetry(me)['sell_revenue']
        a = BASE.agent(obs)
        g.step(*((a, ep['tape'][t]) if me == 0 else (ep['tape'][t], a)))
    return dict(eid=ep['eid'], seat=me, start=snapshot, remaining=value(g.observe(me)),
                day29_sales=g.telemetry(me)['sell_revenue']-before)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--agent', default='agents/e090/main.py')
    ap.add_argument('--replays', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--jobs', type=int, default=1)
    a = ap.parse_args()
    files = sorted(glob.glob(a.replays))
    random.Random(931).shuffle(files)
    eps = [ep for f in files[:a.n] for ep in episode(f, None)]
    with Pool(a.jobs, initializer=init, initargs=(a.agent,)) as pool:
        rows = list(pool.imap_unordered(run, eps))
    Path(a.out).write_text(json.dumps(rows))
    for r in rows:
        print(r['eid'], r['seat'], 'remaining', r['remaining']['quoted_value'],
              'day29_sales', r['day29_sales'], r['remaining']['field'])
