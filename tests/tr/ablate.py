"""記録席 (team) の市場注文から特定の往復を抜いて記録相手テープと再生し、margin の変化を測る (農場は記録どおり)。"""
import json, glob, sys, statistics as S
sys.path.insert(0, 'tests')
import kagsim
from kag_eval import PASS
team, pattern = sys.argv[1], sys.argv[2]
def strip(tape, kind):
    out = []
    pend = {}
    for t, a in enumerate(tape):
        mk = [list(o) for o in (a.get('market') or [])]
        if kind in ('fert', 'both'):
            buys = [o for o in mk if o and o[0] == 'BUY_PRODUCT' and o[1] == 'FERTILIZER' and int(o[2]) >= 20]
            for b in buys:
                q = int(b[2]); s = next((o for o in mk if o and o[0] == 'SELL' and o[1] == 'FERTILIZER' and int(o[2]) == q), None)
                if s is not None: mk.remove(b); mk.remove(s)
        if kind in ('wheat', 'both'):
            if t in pend:
                q = pend.pop(t); s = next((o for o in mk if o and o[0] == 'SELL' and o[1] == 'WHEAT' and int(o[2]) == q), None)
                if s is not None: mk.remove(s)
                mk = [o for o in mk if not (o and o[0] == 'SELL' and o[1] == 'WHEAT' and int(o[2]) == 0)]
            buys = [o for o in mk if o and o[0] == 'BUY_PRODUCT' and o[1] == 'WHEAT' and int(o[2]) >= 20]
            for b in buys:
                q = int(b[2]); s = next((o for o in mk if o and o[0] == 'SELL' and o[1] == 'WHEAT' and int(o[2]) == q), None)
                mk.remove(b)
                if s is not None: mk.remove(s)
                else: pend[t + 1] = q
        out.append(dict(a, market=mk))
    return out
rows = []
for f in sorted(glob.glob(pattern)):
    r = json.load(open(f)); st = r['steps']; names = r['info']['TeamNames']
    if team not in names: continue
    s = names.index(team)
    rec = [st[t + 1][s].get('action') or PASS for t in range(719)]
    opp = [st[t + 1][1 - s].get('action') or PASS for t in range(719)]
    shops = st[-1][0]['observation']['town']['unlocked_shops']
    res = {}
    for kind in ('none', 'fert', 'wheat', 'both'):
        tape = rec if kind == 'none' else strip(rec, kind)
        g = kagsim.Game(r['info']['seed'], 720, shops)
        while not g.done:
            t = g.step_count; acts = [None, None]; acts[s] = tape[t]; acts[1 - s] = opp[t]; g.step(*acts)
        res[kind] = g.reward(s) - g.reward(1 - s)
    rows.append(res)
    print(r['info']['EpisodeId'], {k: round(v) for k, v in res.items()}, flush=True)
for k in ('fert', 'wheat', 'both'):
    d = [x['none'] - x[k] for x in rows]
    print(f"value of {k} trips (none - stripped): {S.mean(d):+.0f} se {S.stdev(d)/len(d)**.5:.0f} n={len(d)}")
