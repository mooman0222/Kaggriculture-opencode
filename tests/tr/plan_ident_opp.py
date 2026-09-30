"""指定チームの対戦相手の席について、E090 の農場行動と何手目まで一致するかを出す (相手の相手 = 指定チームの記録テープ)。"""
import json, glob, sys, collections
sys.path.insert(0, 'tests')
import kagsim
from kag_eval import load, fresh, PASS
teams = sys.argv[1].split(','); pattern = sys.argv[2]
ag, m = load('agents/e090/main.py')
def plan(a):
    a = a or PASS; return json.dumps([a.get('farmer') or ['PASS'], a.get('hands') or []])
buckets = collections.Counter(); rows = []
for f in sorted(glob.glob(pattern)):
    r = json.load(open(f)); st = r['steps']; names = r['info']['TeamNames']
    k = next((t for t in teams if t in names), None)
    if k is None: continue
    s = 1 - names.index(k)
    rec = [plan(st[t + 1][s].get('action')) for t in range(719)]
    tape = [st[t + 1][1 - s].get('action') or PASS for t in range(719)]
    fresh(m); g = kagsim.Game(r['info']['seed'], 720, st[-1][0]['observation']['town']['unlocked_shops']); first = 719
    while not g.done:
        t = g.step_count; a = ag(g.observe(s))
        if plan(a) != rec[t]: first = t; break
        acts = [None, None]; acts[s] = a; acts[1 - s] = tape[t]; g.step(*acts)
    b = 'chassis(>=144)' if first >= 144 else ('opening(17-143)' if first >= 17 else 'custom(<17)')
    buckets[b] += 1; rows.append((names[s][:16], first))
print(dict(buckets)); print(sorted(rows, key=lambda x: -x[1]))
