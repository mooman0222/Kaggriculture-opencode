"""Paired chassis ablations on pinned replay worlds, with execution telemetry.

Specs are JSON objects with name, path, optional entry and constant overrides.
Each variant is imported independently; no shared configuration mutations.
"""
import argparse
import glob
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


def init(specs):
    global MODULES, SPECS
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
        MODULES.append(m)


def run(job):
    ep, ci = job
    m = MODULES[ci]
    ag = getattr(m, SPECS[ci].get('entry', 'agent'))
    g = kagsim.Game(ep['seed'], 720, ep['shops'])
    me = ep['me']
    worst = 0
    while not g.done:
        t = g.step_count
        start = time.perf_counter()
        a = ag(g.observe(me))
        worst = max(worst, time.perf_counter() - start)
        g.step(*((a, ep['tape'][t]) if me == 0 else (ep['tape'][t], a)))
    tel = g.telemetry(me)
    errors = {k: v for k, v in vars(m).items()
              if k.endswith('_REPORT') and isinstance(v, dict)
              and any(val for key, val in v.items() if 'error' in key and isinstance(val, (int, float)))}
    return dict(eid=ep['eid'], seat=me, ci=ci, opp=ep['opp'],
                margin=g.reward(me)-g.reward(1-me), own=g.reward(me),
                max_act=worst, telemetry=tel, errors=errors)


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
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    specs = json.loads(Path(a.specs).read_text())
    files = sorted(glob.glob(a.replays, recursive=True))
    random.Random(a.sample_seed).shuffle(files)
    eps = [e for f in files[a.offset:a.offset+a.n] for e in episode(f, None)]
    assert eps
    start = time.monotonic()
    rows = []
    with Pool(a.jobs, initializer=init, initargs=(specs,)) as pool:
        for r in pool.imap_unordered(run, [(e, ci) for e in eps for ci in range(len(specs))]):
            rows.append(r)
            if len(rows) % 32 == 0:
                print(f"{len(rows)}/{len(eps)*len(specs)} games {time.monotonic()-start:.0f}s", flush=True)
    Path(a.out).write_text(json.dumps(dict(specs=specs, rows=rows)))
    report(rows, specs)


if __name__ == '__main__':
    main()
