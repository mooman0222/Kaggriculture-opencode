"""記録席 (team) の市場注文を、同じ状態で E090 が出す注文と比べ、差を種類別に数える (記録どおり進行)。"""
import json, glob, sys, collections
sys.path.insert(0, 'tests')
import kagsim
from kag_eval import load, fresh, PASS
team, pattern = sys.argv[1], sys.argv[2]
ag, m = load('agents/e090/main.py')
PREM = {'MILK', 'WOOL', 'STRAWBERRY', 'MELON', 'TOMATO', 'EGG', 'CARROT'}
cat = collections.Counter(); ex = collections.defaultdict(list)
def agg(mk):
    c = collections.Counter()
    for o in mk or []:
        if not o: continue
        if o[0] == 'HIRE': c[('HIRE', '')] += 1
        elif o[0] == 'BUY_LAND': c[('BUY_LAND', '')] += 1
        elif len(o) >= 3: c[(o[0], o[1])] += max(0, int(o[2]))
    return c
for f in sorted(glob.glob(pattern)):
    fresh(m)
    r = json.load(open(f)); st = r['steps']; names = r['info']['TeamNames']
    if team not in names: continue
    s = names.index(team)
    rec = [st[t + 1][s].get('action') or PASS for t in range(719)]
    opp = [st[t + 1][1 - s].get('action') or PASS for t in range(719)]
    g = kagsim.Game(r['info']['seed'], 720, st[-1][0]['observation']['town']['unlocked_shops'])
    while not g.done:
        t = g.step_count
        a = ag(g.observe(s))
        me, tr = a.get('market') or [], rec[t].get('market') or []
        if me != tr:
            A, B = agg(me), agg(tr)
            keys = set(A) | set(B)
            if A == B:
                k = 'reorder'
                pa = [o[1] for o in me if o and o[0] == 'SELL' and len(o) > 1]; pb = [o[1] for o in tr if o and o[0] == 'SELL' and len(o) > 1]
                cat[k] += 1; ex[k].append((t, me, tr)) if len(ex[k]) < 4 else None
            for key in keys:
                d = B[key] - A[key]
                if d == 0: continue
                op, item = key
                if op in ('BUY_PRODUCT', 'SELL') and item in ('WHEAT', 'FERTILIZER'): k = f'{op}_{item}_{"+" if d > 0 else "-"}'
                elif op == 'SELL': k = f'SELL_prem_{"+" if d > 0 else "-"}'
                else: k = f'{op}_{"+" if d > 0 else "-"}'
                cat[k] += 1
                if len(ex[k]) < 3: ex[k].append((t, item, d))
        acts = [None, None]; acts[s] = rec[t]; acts[1 - s] = opp[t]; g.step(*acts)
for k, v in cat.most_common(): print(f"{k:28s} {v:5d}  e.g. {ex[k][:3]}")
