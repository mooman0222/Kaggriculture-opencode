"""track の記録を kagsim で再生し、小麦・肥料の往復の発動条件 (日・時刻・数量・倉庫空き・所持金・価格) を集める。"""
import json, glob, sys, collections
sys.path.insert(0, 'tests')
import kagsim
from kag_eval import PASS
team = sys.argv[1]; pattern = sys.argv[2]
W = []; F = []; first = []
for f in sorted(glob.glob(pattern)):
    r = json.load(open(f)); st = r['steps']; names = r['info']['TeamNames']
    if team not in names: continue
    s = names.index(team)
    acts = [[st[t + 1][p].get('action') or PASS for p in (0, 1)] for t in range(719)]
    g = kagsim.Game(r['info']['seed'], 720, st[-1][0]['observation']['town']['unlocked_shops'])
    fw = None
    while not g.done:
        t = g.step_count; o = g.observe(s)
        mk = acts[t][s].get('market') or []
        shed = sum((o['private']['shed'] or {}).values())
        money = o['farms'][s]['money']
        for i, od in enumerate(mk):
            if od and od[0] == 'BUY_PRODUCT' and od[1] == 'WHEAT' and len(od) > 2 and int(od[2]) >= 20:
                nxt = acts[t + 1][s].get('market') or [] if t + 1 < 719 else []
                sold = [x for x in nxt if x and x[0] == 'SELL' and x[1] == 'WHEAT' and int(x[2]) > 0]
                W.append(dict(t=t, mod=t % 4, h=t % 24, d=t // 24, q=int(od[2]), slot=i, n=len(mk), shed=shed, money=int(money), px=o['market']['prices']['WHEAT'],
                              inv=o['market']['inventory']['WHEAT'], next_sell=[int(x[2]) for x in sold], next_slot=[nxt.index(x) for x in sold]))
                fw = fw if fw is not None else t
            if od and od[0] == 'BUY_PRODUCT' and od[1] == 'FERTILIZER' and len(od) > 2 and int(od[2]) >= 20:
                same = [int(x[2]) for x in mk if x and x[0] == 'SELL' and x[1] == 'FERTILIZER']
                F.append(dict(t=t, h=t % 24, d=t // 24, q=int(od[2]), slot=i, sells=same, money=int(money), px=o['market']['prices']['FERTILIZER']))
        g.step(*acts[t])
    first.append(fw)
print('wheat trips', len(W), 'first step per game', first)
print('mod', collections.Counter(x['mod'] for x in W), 'slot==last', sum(x['slot'] == x['n'] - 1 for x in W))
print('q vs 100-shed:', collections.Counter((x['q'] == 100 - x['shed'], x['q'] >= 90) for x in W))
print('q dist', collections.Counter(x['q'] // 10 * 10 for x in W))
print('next sell slot', collections.Counter(tuple(x['next_slot']) for x in W).most_common(5))
print('money min at trip', min(x['money'] for x in W), 'q*px - money', max(x['q'] * x['px'] - x['money'] for x in W))
print('by day', sorted(collections.Counter(x['d'] for x in W).items()))
print('sample', W[:3])
print('fert trips', len(F), 'hours', collections.Counter(x['h'] for x in F), 'q', collections.Counter(x['q'] for x in F).most_common(6))
print('fert by day', sorted(collections.Counter(x['d'] for x in F).items()))
print('fert sample', F[:4])
