"""同じ世界で (a) FQ テープ vs E090、(b) E090 vs E090 を回し、品目別の売上・単価を比べる。"""
import json, sys
sys.path.insert(0, 'tests')
import kagsim
from kag_eval import load, fresh, PASS
ITEMS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
f = sys.argv[1]; team = 'Fourth Quadrant'
r = json.load(open(f)); st = r['steps']; s = r['info']['TeamNames'].index(team)
tape = [st[t + 1][s].get('action') or PASS for t in range(719)]
shops = st[-1][0]['observation']['town']['unlocked_shops']; seed = r['info']['seed']
e1, m1 = load('agents/e090/main.py')
def run(a0, a1, mods):
    for m in mods: fresh(m)
    g = kagsim.Game(seed, 720, shops)
    while not g.done:
        t = g.step_count
        acts = [a0[t] if isinstance(a0, list) else a0(g.observe(0)), a1[t] if isinstance(a1, list) else a1(g.observe(1))]
        g.step(*acts)
    return g
def show(label, g, p):
    tel = g.telemetry(p)
    rev = tel['sell_revenue_items']; u = tel['sold_units_items']
    print(f"{label:14s} money={g.reward(p):7.0f} spend={tel['total_spend']:7.0f} " + ' '.join(f"{k[:4]}:{int(rev[k])//1000}k/{u[k]}@{rev[k]/u[k]:.0f}" for k in ITEMS if u[k]))
A = [tape, e1] if s == 0 else [e1, tape]
g = run(A[0], A[1], [m1]); show('FQtape', g, s); show('E090 vs FQ', g, 1 - s)
g = run(e1, e1, [m1, m1]); show('E090 mirror0', g, 0); show('E090 mirror1', g, 1)
