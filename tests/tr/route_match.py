"""記録席の農場行動を、E090 のルートテープ (_ROUTES) と step 144〜647 で照合し、最も一致するルートを出す。E090 自身の選択とも比較。"""
import json, glob, sys, collections
sys.path.insert(0, 'tests')
from kag_eval import load, PASS
team, pattern, out = sys.argv[1], sys.argv[2], sys.argv[3]
ag, m = load('agents/e090/main.py')
routes = m._IMPL.chassis.routes
def plan(a):
    a = a or PASS; return json.dumps([a.get('farmer') or ['PASS'], a.get('hands') or []])
rplans = {rid: [plan(a) for a in tape[:719]] for rid, tape in routes.items()}
res = []
for f in sorted(glob.glob(pattern)):
    r = json.load(open(f)); st = r['steps']; names = r['info']['TeamNames']
    if team not in names: continue
    s = names.index(team)
    rec = [plan(st[t + 1][s].get('action')) for t in range(719)]
    shops = tuple(st[-1][0]['observation']['town']['unlocked_shops'][:2])
    score = {rid: sum(rec[t] == p[t] for t in range(144, 648)) for rid, p in rplans.items()}
    best = max(score, key=score.get)
    # E090 の選択
    stt = {}
    obs = {'town': {'unlocked_shops': list(shops)}, 'farms': [{'money': 0}, {'money': 0}], 'player': s, 'market': {'inventory': {'WHEAT': 10000}}}
    e090_route = m._e074_router(obs, 200, stt) if hasattr(m, '_e074_router') else None
    res.append(dict(eid=r['info']['EpisodeId'], s=s, shops=shops, best=best, match=score[best], e090=e090_route, e090_match=score.get(e090_route), m=r['rewards'][s] - r['rewards'][1 - s]))
    print(res[-1], flush=True)
json.dump(res, open(out, 'w'))
