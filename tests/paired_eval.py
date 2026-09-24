"""2 候補を同じ seed・同じ世界・同じ相手で走らせ、ペア差をショップペア別 (パッチ有無) に出す。

  .venv/bin/python tests/paired_eval.py --a agents/e076/main.py --b agents/e077/main.py --opp agents/e072/main.py --seeds 60 --seed0 940000 [--pairs-json tmp/x.json]
"""
import argparse, json, math, os, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kagsim
import mirror_eval as ME

M = {}


def init(paths):
    for k, p in paths.items():
        M[k] = ME.load(p, k)


def run(job):
    seed, seat, key, shops = job
    g = kagsim.Game(seed, 720, shops)
    A, O = M[key], M['opp']
    while not g.done:
        x = A.agent(g.observe(seat)); y = O.agent(g.observe(1 - seat))
        g.step(*((x, y) if seat == 0 else (y, x)))
    return seed, seat, key, g.reward(seat) - g.reward(1 - seat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--opp", required=True)
    ap.add_argument("--seeds", type=int, default=60); ap.add_argument("--seed0", type=int, default=940000)
    ap.add_argument("--seats", default="0,1"); ap.add_argument("--jobs", type=int, default=11)
    ap.add_argument("--dump", help="結果を保存/再利用する json")
    ap.add_argument("--patched", help="agents/X/main.py の _E077_PATCH などを読んでペアをパッチ有無で分ける (モジュール変数名)", default="_E077_PATCH")
    a = ap.parse_args()
    worlds = {s: [ME.SHOPS[__import__('random').Random(s * 31 + i).randrange(8)] for i in range(8)] for s in range(a.seed0, a.seed0 + a.seeds)}
    jobs = [(s, seat, k, worlds[s]) for s in worlds for seat in map(int, a.seats.split(",")) for k in ("a", "b")]
    t0 = time.time()
    if a.dump and os.path.exists(a.dump):
        res = [tuple(x) for x in json.load(open(a.dump))]
    else:
        res = None
    with Pool(a.jobs, initializer=init, initargs=({"a": a.a, "b": a.b, "opp": a.opp},)) as p:
        res = res or p.map(run, jobs)
    if a.dump: json.dump(res, open(a.dump, "w"))
    m = {(s, seat, k): v for s, seat, k, v in res}
    probe = ME.load(a.b, "probe")
    patch = getattr(probe, a.patched, {}) or {}
    patched = {tuple(k.split("+")) if isinstance(k, str) else k for k in patch}
    groups = {"patched": [], "other": []}
    for s in worlds:
        for seat in map(int, a.seats.split(",")):
            d = m[(s, seat, "b")] - m[(s, seat, "a")]
            groups["patched" if tuple(worlds[s][:2]) in patched else "other"].append(d)
    for k, d in groups.items():
        if not d: continue
        n = len(d); mu = sum(d) / n; se = math.sqrt(sum((x - mu) ** 2 for x in d) / max(1, n - 1) / n)
        print(f"{k:8s} n={n:3d} b-a {mu:+7.0f} ± {se:5.0f} (t={mu / se if se else 0:+.1f}) +{sum(x > 0 for x in d)}/-{sum(x < 0 for x in d)}")
    flips = {"a_win_b_lose": 0, "a_lose_b_win": 0}
    for s in worlds:
        for seat in map(int, a.seats.split(",")):
            x, y = m[(s, seat, "a")], m[(s, seat, "b")]
            if x > 0 >= y: flips["a_win_b_lose"] += 1
            if x <= 0 < y: flips["a_lose_b_win"] += 1
    print("flips", flips)
    allv = [m[(s, seat, 'a')] for s in worlds for seat in map(int, a.seats.split(","))]
    allb = [m[(s, seat, 'b')] for s in worlds for seat in map(int, a.seats.split(","))]
    print(f"a mean {sum(allv) / len(allv):+.0f} W{sum(x > 0 for x in allv)}  b mean {sum(allb) / len(allb):+.0f} W{sum(x > 0 for x in allb)}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
