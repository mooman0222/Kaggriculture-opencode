"""指定ペア (先頭 2 店を固定、残り 6 店は seed 由来) の世界で、候補を同じ相手・seed・席で比べる。
  python tests/tr/pair_panel.py CANDS OPPS PAIRS SEED0 NSEED OUT   (PAIRS = 'A+B;C+D;...' または 'E090PATCH')
"""
import sys, os, json, math, random, importlib.util, hashlib
from multiprocessing import Pool
import kagsim
SHOPS = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP", "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"]
M = {}
def load(p):
    d = os.path.dirname(os.path.abspath(p)); sys.path.insert(0, d)
    s = importlib.util.spec_from_file_location("q_" + hashlib.md5(p.encode()).hexdigest()[:8], p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def run(job):
    cand, opp, pair, seed, seat = job
    for p in (cand, opp):
        if p not in M: M[p] = load(p)
    A, B = M[cand], M[opp]
    shops = list(pair) + [SHOPS[random.Random(seed * 31 + i).randrange(8)] for i in range(6)]
    g = kagsim.Game(seed, 720, shops)
    while not g.done:
        x = A.agent(g.observe(seat)); y = B.agent(g.observe(1 - seat))
        g.step(*((x, y) if seat == 0 else (y, x)))
    return cand, opp, pair, seed, seat, g.reward(seat) - g.reward(1 - seat)
if __name__ == "__main__":
    cands = sys.argv[1].split(","); opps = sys.argv[2].split(","); seed0 = int(sys.argv[4]); n = int(sys.argv[5]); out = sys.argv[6]
    if sys.argv[3] == 'E090PATCH':
        m = load('agents/e094/main.py'); pairs = sorted(m._E074_PATCH)
    else:
        pairs = [tuple(p.split('+')) for p in sys.argv[3].split(';')]
    jobs = [(c, o, p, s, seat) for o in opps for p in pairs for s in range(seed0, seed0 + n) for seat in (0, 1) for c in cands]
    res = {}
    with Pool(int(os.environ.get("PANEL_JOBS", "2"))) as pool:
        for c, o, p, s, seat, mg in pool.imap_unordered(run, jobs, chunksize=1):
            res[(c, o, p, s, seat)] = mg
    json.dump([[c, o, '+'.join(p), s, seat, v] for (c, o, p, s, seat), v in res.items()], open(out, "w"))
    for o in opps:
        for c in cands[1:]:
            d = [res[(c, o, p, s, seat)] - res[(cands[0], o, p, s, seat)] for p in pairs for s in range(seed0, seed0 + n) for seat in (0, 1)]
            md = sum(d) / len(d); se = math.sqrt(sum((x - md) ** 2 for x in d) / max(1, len(d) - 1) / len(d))
            w0 = sum(res[(cands[0], o, p, s, seat)] > 0 for p in pairs for s in range(seed0, seed0 + n) for seat in (0, 1))
            w1 = sum(res[(c, o, p, s, seat)] > 0 for p in pairs for s in range(seed0, seed0 + n) for seat in (0, 1))
            print(f"{os.path.basename(os.path.dirname(o)):16s} {os.path.basename(os.path.dirname(c))} - {os.path.basename(os.path.dirname(cands[0]))}: d {md:+6.0f} ± {se:4.0f}  wins {w0} -> {w1} of {len(d)}")
        for p in pairs:
            dd = [res[(cands[1], o, p, s, seat)] - res[(cands[0], o, p, s, seat)] for s in range(seed0, seed0 + n) for seat in (0, 1)]
            print(f"    {'+'.join(p):32s} d {sum(dd)/len(dd):+7.0f}")
