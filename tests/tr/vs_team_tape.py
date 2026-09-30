"""指定チームの記録テープを相手に、その対戦相手の席へ候補を入れて対戦させる (世界は記録どおり固定)。"""
import json, glob, sys, statistics as S
sys.path.insert(0, 'tests')
import kagsim
from kag_eval import load, fresh, PASS
team, pattern = sys.argv[1], sys.argv[2]; cands = sys.argv[3:]
mods = [(c, *load(c)) for c in cands]
res = {c: [] for c in cands}
for f in sorted(glob.glob(pattern)):
    r = json.load(open(f)); st = r['steps']; names = r['info']['TeamNames']
    if team not in names: continue
    s = names.index(team); me = 1 - s
    tape = [st[t + 1][s].get('action') or PASS for t in range(719)]
    shops = st[-1][0]['observation']['town']['unlocked_shops']
    line = []
    for c, ag, m in mods:
        fresh(m); g = kagsim.Game(r['info']['seed'], 720, shops)
        while not g.done:
            t = g.step_count; acts = [None, None]; acts[s] = tape[t]; acts[me] = ag(g.observe(me)); g.step(*acts)
        res[c].append(g.reward(me) - g.reward(s)); line.append(f"{c.split('/')[1]} {res[c][-1]:+7.0f}")
    print(r['info']['EpisodeId'], 'rec', r['rewards'][me] - r['rewards'][s], '|', ' '.join(line), flush=True)
for c in cands:
    v = res[c]; print(f"{c}: n={len(v)} mean {S.mean(v):+.0f} se {S.stdev(v)/len(v)**.5:.0f} wins {sum(x>0 for x in v)}")
d = [a - b for a, b in zip(res[cands[1]], res[cands[0]])]
print(f"diff {cands[1]} - {cands[0]}: {S.mean(d):+.0f} ± {S.stdev(d)/len(d)**.5:.0f}")
