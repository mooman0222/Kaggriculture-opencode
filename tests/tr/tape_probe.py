"""上位の記録行動 (テープ) を別の相手に対して再生したときの所持金を測る。"""
import json, glob, sys
sys.path.insert(0, 'tests')
import kagsim
from kag_eval import load, fresh, PASS
team = sys.argv[1]; opp_path = sys.argv[2]; pattern = sys.argv[3]
opp, omod = load(opp_path)
for f in sorted(glob.glob(pattern)):
    r = json.load(open(f)); st = r['steps']
    names = r['info']['TeamNames']
    if team not in names: continue
    s = names.index(team)
    tape = [st[t + 1][s].get('action') or PASS for t in range(719)]
    rec_tape_opp = [st[t + 1][1 - s].get('action') or PASS for t in range(719)]
    shops = st[-1][0]['observation']['town']['unlocked_shops']
    seed = r['info']['seed']
    out = []
    for label, other in (('rec_opp', rec_tape_opp), ('live', opp)):
        fresh(omod)
        g = kagsim.Game(seed, 720, shops)
        while not g.done:
            t = g.step_count
            a_me = tape[t]
            a_ot = other[t] if isinstance(other, list) else other(g.observe(1 - s))
            acts = [None, None]; acts[s] = a_me; acts[1 - s] = a_ot
            g.step(acts[0], acts[1])
        out.append(f"{label}: tape={g.reward(s):7.0f} opp={g.reward(1-s):7.0f}")
    print(f.split('-')[1], 'seat', s, 'recorded', r['rewards'][s], r['rewards'][1 - s], '|', ' | '.join(out), flush=True)
