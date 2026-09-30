"""固定テープ (切替なし) を複数の世界 (seed+ショップ列を固定) で E090 と対戦させ、テープごとの平均 margin を出す。
  python tests/tr/fixed_tape.py LIB SEAT N_TAPES N_WORLDS OUT
"""
import json, sys, random, statistics as S, time
sys.path.insert(0, 'tests')
import kagsim
from kag_eval import load, fresh, PASS
lib = json.load(open(sys.argv[1])); seat = int(sys.argv[2]); nt = int(sys.argv[3]); nw = int(sys.argv[4]); out = sys.argv[5]
acts = lib['actions']
opp, omod = load('agents/e090/main.py')
tapes = [tp for tp in lib['tapes'] if tp['s'] == seat]
tapes.sort(key=lambda tp: -(tp['own'] - tp['opp']))
worlds = [(tp['seed'], tp['shops']) for tp in lib['tapes']]
random.Random(7).shuffle(worlds)
worlds = worlds[:nw]
res = {}
for tp in tapes[:nt]:
    tape = [acts[i] for i in tp['ids']]
    ms = []
    for seed, shops in worlds:
        fresh(omod)
        g = kagsim.Game(seed, 720, shops)
        while not g.done:
            t = g.step_count
            a = [None, None]; a[seat] = tape[t]; a[1 - seat] = opp(g.observe(1 - seat))
            g.step(a[0], a[1])
        ms.append((g.reward(seat) - g.reward(1 - seat), g.reward(seat)))
        time.sleep(0.2)
    m = [x[0] for x in ms]
    res[tp['eid']] = ms
    print(f"tape {tp['eid']} rec_m={tp['own']-tp['opp']:+7.0f} shops={tp['shops'][:2]} vsE090 mean {S.mean(m):+8.0f} se {S.stdev(m)/len(m)**.5:6.0f} win {sum(x>0 for x in m)}/{len(m)} own {S.mean(x[1] for x in ms):7.0f}", flush=True)
json.dump(dict(worlds=worlds, res=res), open(out, 'w'))
