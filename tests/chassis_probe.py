"""Paired chassis ablations on pinned replay worlds, with execution telemetry.

Specs are JSON objects with name, path, optional entry and constant overrides.
Each variant is imported independently; no shared configuration mutations.
"""
import argparse
import glob
import hashlib
import json
import math
import random
import statistics
import time
from multiprocessing import Pool
from pathlib import Path

import kagsim
from swap_eval import episode, load

MODULES = []
SPECS = []
PAUSE_PER_GAME = 2.0


def init(specs, pause_per_game=2.0):
    global MODULES, SPECS, PAUSE_PER_GAME
    PAUSE_PER_GAME = pause_per_game
    SPECS = specs
    MODULES = []
    for spec in specs:
        m = load(str(Path(spec['path']).resolve()))
        for name, value in spec.get('set', {}).items():
            assert hasattr(m, name), name
            setattr(m, name, value)
        for name, source in spec.get('alias', {}).items():
            assert hasattr(m, name), name
            setattr(m, name, getattr(m, source))
        if 'route' in spec:
            parent = m._IMPL.chassis.router
            route = spec['route']
            def router(obs, step, state, parent=parent, route=route):
                original = parent(obs, step, state)
                return route if 144 <= step < 648 else original
            m._IMPL.chassis.router = router
        MODULES.append(m)


def run(job):
    time.sleep(PAUSE_PER_GAME)
    cpu_start=time.process_time()
    wall_start=time.monotonic()
    ep, ci = job
    m = MODULES[ci]
    ag = getattr(m, SPECS[ci].get('entry', 'agent'))
    g = kagsim.Game(ep['seed'], 720, ep['shops'])
    me = ep['me']
    worst = 0
    features = None
    while not g.done:
        t = g.step_count
        if t%24==0:
            delay=(time.process_time()-cpu_start)/.30-(time.monotonic()-wall_start)
            if delay>0:time.sleep(delay)
        obs = g.observe(me)
        if t == 144:
            features = {'shops': obs['town']['unlocked_shops'],
                        'market': obs['market'], 'farms': []}
            for farm in (obs['farms'][me], obs['farms'][1-me]):
                counts = {}
                for row in farm['tiles']:
                    for tile in row:
                        if isinstance(tile, dict):
                            key = tile.get('animal') or tile.get('crop') or tile.get('kind')
                            counts[key] = counts.get(key, 0) + 1
                features['farms'].append({'counts': counts, 'money': farm['money'],
                                          'hands': len(farm.get('hands', []))})
        start = time.perf_counter()
        a = ag(obs)
        worst = max(worst, time.perf_counter() - start)
        g.step(*((a, ep['tape'][t]) if me == 0 else (ep['tape'][t], a)))
    tel = g.telemetry(me)
    errors = {k: v for k, v in vars(m).items()
              if k.endswith('_REPORT') and isinstance(v, dict)
              and any(val for key, val in v.items() if 'error' in key and isinstance(val, (int, float)))}
    return dict(eid=ep['eid'], seat=me, ci=ci, opp=ep['opp'],
                margin=g.reward(me)-g.reward(1-me), own=g.reward(me),
                max_act=worst, telemetry=tel, errors=errors, features=features)


def report(rows, specs):
    base = {(r['eid'], r['seat']): r for r in rows if r['ci'] == 0}
    for ci, spec in enumerate(specs):
        group = [r for r in rows if r['ci'] == ci]
        d = [r['margin'] - base[r['eid'], r['seat']]['margin'] for r in group]
        n = len(group)
        se = statistics.stdev(d) / math.sqrt(n) if n > 1 else 0
        mu = statistics.mean(d)
        print(f"{spec['name']:20s} n={n} W={sum(r['margin']>0 for r in group)} "
              f"tie={sum(r['margin']==0 for r in group)} margin={statistics.mean(r['margin'] for r in group):+.0f} "
              f"delta={mu:+.0f} SE={se:.0f} t={mu/se if se else 0:+.2f} "
              f"max_act={max(r['max_act'] for r in group):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--specs', required=True)
    ap.add_argument('--replays', required=True)
    ap.add_argument('--n', type=int, default=24, help='number of episodes, both seats')
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--sample-seed', type=int, default=929)
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--pause-per-game', type=float, default=2.0,
                    help='idle seconds per game to reduce sustained CPU load')
    ap.add_argument('--resume', action='store_true', help='reuse completed jobs from --out')
    ap.add_argument('--baseline-cache', help='reuse identical baseline from an earlier paired run')
    ap.add_argument('--out', required=True)
    ap.add_argument('--exclude', type=int, nargs='*', default=[])
    a = ap.parse_args()
    assert 1 <= a.jobs <= 4, 'use between 1 and 4 workers'
    assert a.pause_per_game >= 0, 'pause must be nonnegative'
    specs = json.loads(Path(a.specs).read_text())
    hashes = [hashlib.sha256(Path(s['path']).read_bytes()).hexdigest() for s in specs]
    files = sorted(glob.glob(a.replays, recursive=True))
    random.Random(a.sample_seed).shuffle(files)
    eps = [e for f in files[a.offset:a.offset+a.n] for e in episode(f, None)
           if e['eid'] not in a.exclude]
    assert eps
    start = time.monotonic()
    rows = []
    wanted = {(e['eid'], e['me']) for e in eps}
    if a.baseline_cache:
        cached = json.loads(Path(a.baseline_cache).read_text())
        assert cached['specs'][0] == specs[0], 'cached baseline spec mismatch'
        if 'source_hashes' in cached:
            assert cached['source_hashes'][0] == hashes[0], 'cached baseline source changed'
        rows = [r for r in cached['rows'] if r['ci'] == 0
                and (r['eid'], r['seat']) in wanted]
    if a.resume and Path(a.out).exists():
        prior = json.loads(Path(a.out).read_text())
        assert prior['specs'] == specs, 'resume specs mismatch'
        if 'source_hashes' in prior:
            assert prior['source_hashes'] == hashes, 'resume source changed'
        combined = {(r['eid'], r['seat'], r['ci']):r for r in rows}
        combined.update({(r['eid'], r['seat'], r['ci']):r for r in prior['rows']
                         if (r['eid'], r['seat']) in wanted})
        rows = list(combined.values())
    complete = {(r['eid'], r['seat'], r['ci']) for r in rows}
    jobs = [(e, ci) for e in eps for ci in range(len(specs))
            if (e['eid'], e['me'], ci) not in complete]
    print(f'resumed {len(rows)} games; remaining {len(jobs)}; workers={a.jobs}; pause={a.pause_per_game}s', flush=True)
    try:
        with Pool(a.jobs, initializer=init, initargs=(specs, a.pause_per_game)) as pool:
            for r in pool.imap_unordered(run, jobs):
                rows.append(r)
                if len(rows) % 32 == 0:
                    print(f"{len(rows)}/{len(eps)*len(specs)} games {time.monotonic()-start:.0f}s", flush=True)
                    Path(a.out).write_text(json.dumps(dict(specs=specs, rows=rows, args=vars(a), source_hashes=hashes)))
    finally:
        Path(a.out).write_text(json.dumps(dict(specs=specs, rows=rows, args=vars(a), source_hashes=hashes)))
    report(rows, specs)


if __name__ == '__main__':
    main()
