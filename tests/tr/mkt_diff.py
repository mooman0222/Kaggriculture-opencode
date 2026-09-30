"""記録席と E090 (同じ世界・記録どおり進行) の市場注文の差を、品目別の SELL 量の日別差でまとめる。"""
import json, sys, collections, glob
sys.path.insert(0, 'tests')
import kagsim
from kag_eval import load, fresh, PASS
team, pattern = sys.argv[1], sys.argv[2]; base = sys.argv[3] if len(sys.argv) > 3 else 'agents/e090/main.py'
ag, m = load(base)
def sells(a):
    c = collections.Counter()
    for o in (a or {}).get('market') or []:
        if o and o[0] == 'SELL' and len(o) >= 3: c[o[1]] += int(o[2])
        elif o and o[0] == 'BUY_PRODUCT' and len(o) >= 3: c['buy_' + o[1]] += int(o[2])
    return c
for f in sorted(glob.glob(pattern)):
    fresh(m)
    r = json.load(open(f)); st = r['steps']; names = r['info']['TeamNames']
    if team not in names: continue
    s = names.index(team)
    rec = [st[t + 1][s].get('action') or PASS for t in range(719)]
    opp = [st[t + 1][1 - s].get('action') or PASS for t in range(719)]
    shops = st[-1][0]['observation']['town']['unlocked_shops']
    g = kagsim.Game(r['info']['seed'], 720, shops)
    ndiff = 0; slot0 = collections.Counter(); hours = collections.Counter(); first = None
    while not g.done:
        t = g.step_count
        a = ag(g.observe(s))
        if a.get('market') != rec[t].get('market'):
            ndiff += 1; first = first if first is not None else t; hours[t % 24] += 1
        acts = [None, None]; acts[s] = rec[t]; acts[1 - s] = opp[t]; g.step(*acts)
    print(r['info']['EpisodeId'], 'seat', s, 'm', r['rewards'][s] - r['rewards'][1 - s], 'market-diff steps', ndiff, 'first', first, 'by hour', dict(sorted(hours.items())), flush=True)
