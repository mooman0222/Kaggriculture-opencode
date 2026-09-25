"""候補 (複数) を同じ相手・同じ seed・同じ席で対戦させ、先頭候補とのペア差と勝数を出す (相手は反応する実物)。

  .venv/bin/python tests/panel_eval.py agents/e079/main.py,agents/e081/main.py agents/pub_cha22/main.py,agents/e072/main.py 3100000 12 tmp/panel.json
  引数: 候補 (カンマ区切り、先頭が基準) / 相手 (カンマ区切り) / seed0 / seed 数 (各 seed 両席) / 結果 JSON。agents/ 配下のみ実行する。
"""
import sys, os, json, math, random, importlib.util, hashlib
from multiprocessing import Pool
import kagsim
SHOPS = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP", "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"]
M = {}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load(p):
    assert os.path.abspath(p).startswith(os.path.join(ROOT, "agents") + os.sep), "only agents/ are executed"
    d = os.path.dirname(os.path.abspath(p)); sys.path.insert(0, d)
    s = importlib.util.spec_from_file_location("p_" + hashlib.md5(p.encode()).hexdigest()[:8], p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def run(job):
    cand, opp, seed, seat = job
    for p in (cand, opp):
        if p not in M: M[p] = load(p)
    A, B = M[cand], M[opp]
    shops = [SHOPS[random.Random(seed * 31 + i).randrange(8)] for i in range(8)]
    g = kagsim.Game(seed, 720, shops)
    while not g.done:
        x = A.agent(g.observe(seat)); y = B.agent(g.observe(1 - seat))
        g.step(*((x, y) if seat == 0 else (y, x)))
    return cand, opp, seed, seat, g.reward(seat) - g.reward(1 - seat)
if __name__ == "__main__":
    cands = sys.argv[1].split(","); opps = sys.argv[2].split(","); seed0 = int(sys.argv[3]); n = int(sys.argv[4]); out = sys.argv[5]
    jobs = [(c, o, s, seat) for o in opps for s in range(seed0, seed0 + n) for seat in (0, 1) for c in cands]
    res = {}
    with Pool(4) as p:
        for c, o, s, seat, m in p.imap_unordered(run, jobs, chunksize=1):
            res[(c, o, s, seat)] = m
    json.dump([[*k, v] for k, v in res.items()], open(out, "w"))
    tot = {c: [] for c in cands}
    for o in opps:
        keys = [(s, seat) for s in range(seed0, seed0 + n) for seat in (0, 1)]
        line = f"{os.path.basename(os.path.dirname(o)):18s}"
        for c in cands:
            ms = [res[(c, o, s, seat)] for s, seat in keys]; tot[c] += ms
            d = [res[(c, o, s, seat)] - res[(cands[0], o, s, seat)] for s, seat in keys]
            md = sum(d) / len(d); se = math.sqrt(sum((x - md) ** 2 for x in d) / max(1, len(d) - 1) / len(d))
            line += f" | {os.path.basename(os.path.dirname(c))} W{sum(x > 0 for x in ms):2d}/{len(ms)} m{sum(ms) / len(ms):+6.0f}" + (f" d{md:+5.0f}±{se:3.0f}" if c != cands[0] else "")
        print(line, flush=True)
    for c in cands:
        ms = tot[c]; print(f"TOTAL {c}: W {sum(x > 0 for x in ms)}/{len(ms)} mean {sum(ms) / len(ms):+.0f}")
