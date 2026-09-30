import json, sys
sys.path.insert(0, 'tests')
import kagsim
from kag_eval import load, fresh, PASS
f = sys.argv[1]
r = json.load(open(f)); st = r['steps']; s = r['info']['TeamNames'].index('Fourth Quadrant')
tape = [st[t + 1][s].get('action') or PASS for t in range(719)]
shops = st[-1][0]['observation']['town']['unlocked_shops']; seed = r['info']['seed']
e1, m1 = load('agents/e090/main.py')
ITEMS = ['STRAWBERRY', 'MILK', 'WOOL', 'MELON']
def run(a0, a1, mods):
    for m in mods: fresh(m)
    g = kagsim.Game(seed, 720, shops)
    daily = [[{k: 0 for k in ITEMS} for _ in range(30)] for _ in (0, 1)]
    prev = [dict(g.telemetry(p)['sold_units_items']) for p in (0, 1)]
    while not g.done:
        t = g.step_count
        acts = [a0[t] if isinstance(a0, list) else a0(g.observe(0)), a1[t] if isinstance(a1, list) else a1(g.observe(1))]
        g.step(*acts)
        for p in (0, 1):
            cur = g.telemetry(p)['sold_units_items']
            for k in ITEMS: daily[p][t // 24][k] += cur[k] - prev[p][k]
            prev[p] = dict(cur)
    return daily
A = [tape, e1] if s == 0 else [e1, tape]
d1 = run(A[0], A[1], [m1])
d2 = run(e1, e1, [m1, m1])
print('day | FQ sold (S,Mi,W,Me) | E090 vs FQ | E090 mirror')
for d in range(30):
    f_ = d1[s][d]; e_ = d1[1 - s][d]; m_ = d2[0][d]
    print(f"{d:2d} | {[f_[k] for k in ITEMS]} | {[e_[k] for k in ITEMS]} | {[m_[k] for k in ITEMS]}")
