"""ショップペア × ルートの総当たり: 自分 (--us) の step144 のルートだけを差し替え、鏡像の相手 (--opp、既定表) と両席で対戦する。

  .venv/bin/python tests/route_sweep.py --us agents/e074/main.py --opp agents/e072/main.py --seeds 4 --seed0 700000 --out tmp/sweep.json
  --pairs 'A+B,C+D' / --routes '0,9,126' で絞れる。ショップ列は先頭 2 つをペアに固定し、残り 6 つは seed から乱択。
  自分側は agents/*/main.py の _E074_PATCH (step144-647 のペア別ルート上書き) を書き換えて使う。
"""
import argparse, hashlib, importlib.util, json, os, random, sys, time
from multiprocessing import Pool
import kagsim

SHOPS = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP", "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U = O = None


def load(p, tag):
    ap = os.path.abspath(p)
    assert ap.startswith(os.path.join(ROOT, "agents") + os.sep), "only our own agents"
    s = importlib.util.spec_from_file_location(tag + hashlib.md5(ap.encode()).hexdigest()[:6], ap)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def init(us, opp):
    global U, O
    U, O = load(us, "u"), load(opp, "o")


def world(pair, seed):
    rng = random.Random(seed * 7919 + SHOPS.index(pair[0]) * 8 + SHOPS.index(pair[1]))
    return list(pair) + [rng.choice(SHOPS) for _ in range(6)]


def run(job):
    pair, route, seed, seat = job
    saved = dict(U._E074_PATCH)
    U._E074_PATCH.clear()
    if route is not None: U._E074_PATCH[pair] = route
    else: U._E074_PATCH.update(saved)
    try:
        g = kagsim.Game(seed, 720, world(pair, seed))
        while not g.done:
            a = U.agent(g.observe(seat)); b = O.agent(g.observe(1 - seat))
            g.step(*((a, b) if seat == 0 else (b, a)))
        return pair, route, seed, seat, g.reward(seat) - g.reward(1 - seat), g.reward(seat)
    finally:
        U._E074_PATCH.clear(); U._E074_PATCH.update(saved)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--us", required=True); ap.add_argument("--opp", required=True)
    ap.add_argument("--pairs", default="all"); ap.add_argument("--routes", default="all")
    ap.add_argument("--seeds", type=int, default=4); ap.add_argument("--seed0", type=int, default=700000)
    ap.add_argument("--jobs", type=int, default=11); ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", action="store_true", help="ルート None (= 既定表) も入れる")
    ap.add_argument("--seats", default="0", help="鏡像戦は両席で同値になるので既定は 0 のみ")
    ap.add_argument("--cands", help="sweep_report --json の出力 (ペアごとの候補ルートだけを回す)")
    ap.add_argument("--top", type=int, default=3)
    a = ap.parse_args()
    m = load(a.us, "probe")
    routes = sorted(m._IMPL.chassis.routes) if a.routes == "all" else [int(x) for x in a.routes.split(",")]
    pairs = [(x, y) for x in SHOPS for y in SHOPS] if a.pairs == "all" else [tuple(p.split("+")) for p in a.pairs.split(",")]
    rset = ([None] if a.baseline else []) + routes
    per = {p: rset for p in pairs}
    if a.cands:
        cj = json.load(open(a.cands))
        per = {tuple(k.split("+")): ([None] if a.baseline else []) + [c["route"] for c in v[:a.top]] for k, v in cj.items()}
    jobs = [(p, r, s, seat) for p, rs in per.items() for s in range(a.seed0, a.seed0 + a.seeds) for r in rs for seat in map(int, a.seats.split(","))]
    random.Random(1).shuffle(jobs)
    t0 = time.time(); res = []
    with Pool(a.jobs, initializer=init, initargs=(a.us, a.opp)) as pool:
        for i, x in enumerate(pool.imap_unordered(run, jobs, chunksize=4)):
            res.append(x)
            if (i + 1) % 2000 == 0:
                print(f"{i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
                json.dump(res, open(a.out, "w"))
    json.dump(res, open(a.out, "w"))
    print(f"done {len(res)} games {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
