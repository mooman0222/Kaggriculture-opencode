"""E090 鏡像戦で、日ごとの空き労働 (PASS のユニット手番)・空きマス (解放済み)・所持金・雇用数を出す。"""
import sys
sys.path.insert(0, 'tests')
import kagsim
from kag_eval import load, fresh
e, m = load('agents/e090/main.py')
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
fresh(m)
g = kagsim.Game(seed)
Q = {'NW': (0, 0), 'NE': (5, 0), 'SW': (0, 5), 'SE': (5, 5)}
rows = []
idle = [0] * 30; units = [0] * 30; moves = [0] * 30
while not g.done:
    t = g.step_count
    o0 = g.observe(0); o1 = g.observe(1)
    a0 = e(o0); a1 = e(o1)
    for u in [a0.get('farmer') or ['PASS']] + list(a0.get('hands') or []):
        units[t // 24] += 1
        if not u or u[0] == 'PASS': idle[t // 24] += 1
        elif u[0] in ('NORTH', 'SOUTH', 'EAST', 'WEST'): moves[t // 24] += 1
    if t % 24 == 12:
        f = o0['farms'][0]
        empty = 0
        for q in f['unlocked_quadrants']:
            x0, y0 = Q[q]
            for y in range(y0, y0 + 5):
                for x in range(x0, x0 + 5):
                    if f['tiles'][y][x] is None: empty += 1
        rows.append((t // 24, int(f['money']), len(f['hands']), f['unlocked_quadrants'], empty))
    g.step(a0, a1)
for d, money, hands, q, empty in rows:
    print(f"day {d:2d} money {money:6d} hands {hands:2d} quads {len(q)} empty {empty:2d} idle {idle[d]:3d}/{units[d]:3d} moves {moves[d]:3d}")
print('final', g.reward(0), g.reward(1))
