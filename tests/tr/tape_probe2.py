"""テープを別 seed (ショップ列は固定) で再生し、崩れ方を測る。"""
import json, glob, sys
sys.path.insert(0, 'tests')
import kagsim
from kag_eval import load, fresh, PASS
team, opp_path, f = sys.argv[1], sys.argv[2], sys.argv[3]
seeds = [int(x) for x in sys.argv[4].split(',')]
alt_shops = sys.argv[5].split(',') if len(sys.argv) > 5 else None
opp, omod = load(opp_path)
r = json.load(open(f)); st = r['steps']; names = r['info']['TeamNames']; s = names.index(team)
tape = [st[t + 1][s].get('action') or PASS for t in range(719)]
shops = alt_shops or st[-1][0]['observation']['town']['unlocked_shops']
for seed in seeds:
    fresh(omod)
    g = kagsim.Game(seed, 720, shops)
    while not g.done:
        t = g.step_count
        acts = [None, None]; acts[s] = tape[t]; acts[1 - s] = opp(g.observe(1 - s))
        g.step(acts[0], acts[1])
    print(f"seed {seed}: tape={g.reward(s):7.0f} opp={g.reward(1-s):7.0f} m={g.reward(s)-g.reward(1-s):+7.0f}", flush=True)
