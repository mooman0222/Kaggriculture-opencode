"""Credit-assignment check. Play bc (greedy Policy3) vs v41 on pinned shops; at step t delete one kind of market order
(seed / anim / sell / hire) from our action, replay to the end, and compare
  true  = final margin(base) - final margin(branch)          (causal value of taking the order)
  adv_c = sum_k c^k [r_{t+k}(base) - r_{t+k}(branch)]          (what the PPO advantage credits to it, constant critic)
with r = dense * d(own-opp)/1000 per step (+ terminal +-1), c = gamma*lam (0.947) and c = gamma (0.997).
usage: branch.py CKPT KIND SEED0 NSEEDS PER_GAME"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), "rl"))
import kagsim
from play3 import BCAgent3, load_agent

ck, kind, seed0, nseeds, per = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
agent = BCAgent3(ck); opp = load_agent("third_party/public_agents/v41/main.py")
MATCH = {"seed": ("BUY_SEED",), "anim": ("BUY_ANIMAL",), "sell": ("SELL",), "hire": ("HIRE",), "wheat": ("BUY_PRODUCT",)}[kind]


def play(seed, seat, shops=None, cut=None):
    for name in ("_LIVE", "_POLICY"):
        if hasattr(opp, name): setattr(opp, name, None)
    g = kagsim.Game(seed, 720, shops) if shops else kagsim.Game(seed)
    m = [0.0]; hits = []
    while not g.done:
        t = g.step_count; o = g.observe(seat)
        a = agent.act(o, seat)
        mk = a.get("market") or []
        if any(x and x[0] in MATCH for x in mk) and 24 <= t < 24 * 14: hits.append(t)
        if cut == t: a["market"] = [x for x in mk if not (x and x[0] in MATCH)]
        b = opp.agent(g.observe(1 - seat))
        g.step(*((a, b) if seat == 0 else (b, a)))
        o2 = g.observe(seat); m.append(o2["farms"][seat]["money"] - o2["farms"][1 - seat]["money"])
    fin = g.reward(seat) - g.reward(1 - seat); m[-1] = fin
    return np.array(m), fin, hits, g.observe(seat)["town"]["unlocked_shops"]


def adv(mb, ma, t, c, dense=0.2):
    rb = dense * np.diff(mb[t:]) / 1000; ra = dense * np.diff(ma[t:]) / 1000
    rb[-1] += np.sign(mb[-1]); ra[-1] += np.sign(ma[-1])
    w = c ** np.arange(len(rb))
    return float((w * (rb - ra)).sum())


rows = []
for s in range(nseeds):
    seed = seed0 + s; seat = s % 2
    mb, fb, hits, shops = play(seed, seat)
    mb2, fb2, _, _ = play(seed, seat, shops)
    assert fb2 == fb, (fb, fb2)   # pinning the drawn shops reproduces the base game
    for t in [hits[i] for i in np.linspace(0, len(hits) - 1, min(per, len(hits))).astype(int)] if hits else []:
        ma, fa, _, _ = play(seed, seat, shops, cut=t)
        rows.append(dict(seed=seed, t=t, true=fb - fa, a947=adv(mb, ma, t, 0.997 * 0.95), a997=adv(mb, ma, t, 0.997)))
        print(json.dumps(rows[-1]), flush=True)
