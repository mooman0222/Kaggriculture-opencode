"""記録席の市場注文と、E090 を同じ世界・記録相手テープ・記録農場で走らせた市場注文を手番ごとに並べる。"""
import json, sys
sys.path.insert(0, 'tests')
import kagsim
from kag_eval import load, fresh, PASS
team, f = sys.argv[1], sys.argv[2]; lo, hi = int(sys.argv[3]), int(sys.argv[4])
ag, m = load('agents/e090/main.py'); fresh(m)
r = json.load(open(f)); st = r['steps']; names = r['info']['TeamNames']; s = names.index(team)
rec = [st[t + 1][s].get('action') or PASS for t in range(719)]
opp = [st[t + 1][1 - s].get('action') or PASS for t in range(719)]
shops = st[-1][0]['observation']['town']['unlocked_shops']
g = kagsim.Game(r['info']['seed'], 720, shops)
while not g.done:
    t = g.step_count
    a = ag(g.observe(s))
    if lo <= t < hi and a.get('market') != rec[t].get('market'):
        print(t, 'E090', a.get('market')); print(t, team, rec[t].get('market'))
    acts = [None, None]; acts[s] = rec[t]; acts[1 - s] = opp[t]; g.step(*acts)  # 記録どおり進める
print('rewards', r['rewards'], names)
