"""Single-process, paced exact-action differential check for reorder memoization."""
import argparse
import glob
import json
import random
import time
from pathlib import Path

import kagsim
from swap_eval import episode, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    slow = load(str(Path('agents/e090/main.py').resolve()))
    fast = load(str(Path('agents/x090_fast/main.py').resolve()))
    files = sorted(glob.glob('tmp/band0929_slim/episode-*.json'))
    random.Random(934).shuffle(files)
    rows = []
    for path in files[:args.n]:
        for ep in episode(path, None):
            time.sleep(4)
            g = kagsim.Game(ep['seed'], 720, ep['shops'])
            seconds = [0.0, 0.0]
            worst = [0.0, 0.0]
            while not g.done:
                t = g.step_count
                obs = g.observe(ep['me'])
                actions = []
                for i, mod in enumerate((slow, fast)):
                    start = time.process_time()
                    actions.append(mod.agent(obs))
                    dt = time.process_time()-start
                    seconds[i] += dt
                    worst[i] = max(worst[i], dt)
                assert actions[0] == actions[1], (ep['eid'], ep['me'], t, actions)
                a, b = actions[0], ep['tape'][t]
                g.step(*((a,b) if ep['me']==0 else (b,a)))
            rows.append(dict(eid=ep['eid'], seat=ep['me'], steps=g.step_count,
                             cpu_seconds=seconds, max_act_cpu=worst,
                             cache=dict(fast._FAST_REPORT)))
            Path(args.out).write_text(json.dumps(rows))
            print(ep['eid'], ep['me'], 'identical', g.step_count,
                  'CPU seconds', [round(x,3) for x in seconds], flush=True)
    totals = [sum(r['cpu_seconds'][i] for r in rows) for i in (0,1)]
    print('total CPU', totals, 'speedup', totals[0]/totals[1])


if __name__ == '__main__':
    main()
