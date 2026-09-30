"""記録席の農場行動 (farmer+hands) を、候補エージェントを同じ世界・記録相手テープで走らせた農場行動と比べ、最初の不一致手を出す。"""
import json, glob, sys
sys.path.insert(0, 'tests')
import kagsim
from kag_eval import load, fresh, PASS
team = sys.argv[1]; pattern = sys.argv[2]; cands = sys.argv[3:]
mods = [(c, *load(c)) for c in cands]
def plan(a):
    a = a or PASS; return json.dumps([a.get('farmer') or ['PASS'], a.get('hands') or []])
for f in sorted(glob.glob(pattern)):
    r = json.load(open(f)); st = r['steps']; names = r['info']['TeamNames']
    if team not in names: continue
    s = names.index(team)
    rec = [plan(st[t + 1][s].get('action')) for t in range(719)]
    opp = [st[t + 1][1 - s].get('action') or PASS for t in range(719)]
    shops = st[-1][0]['observation']['town']['unlocked_shops']
    out = {}
    for c, ag, m in mods:
        fresh(m); g = kagsim.Game(r['info']['seed'], 720, shops); first = 719
        while not g.done:
            t = g.step_count; a = ag(g.observe(s))
            if plan(a) != rec[t] and first == 719: first = t; break
            acts = [None, None]; acts[s] = a; acts[1 - s] = opp[t]; g.step(*acts)
        out[c.split('/')[1]] = first
    print(r['info']['EpisodeId'], 'seat', s, shops[:2], out, flush=True)
