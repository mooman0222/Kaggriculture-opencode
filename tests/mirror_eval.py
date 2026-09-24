"""自前エージェント同士の並列対戦 (ショップ列は seed から乱択して固定)。鏡像戦の A/B 用。

  .venv/bin/python tests/mirror_eval.py --a agents/e076/main.py --b agents/e072/main.py --seeds 40 [--seats 0,1] [--env K=V ...]
"""
import argparse, hashlib, importlib.util, math, os, random, sys, time
from multiprocessing import Pool
import kagsim

SHOPS = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP", "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = B = None


def load(p, tag):
    ap = os.path.abspath(p)
    assert ap.startswith(os.path.join(ROOT, "agents") + os.sep), "only our own agents"
    s = importlib.util.spec_from_file_location(tag + hashlib.md5(ap.encode()).hexdigest()[:6], ap)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def init(a, b, env):
    global A, B
    os.environ.update(env)
    A, B = load(a, "a"), load(b, "b")


def run(job):
    seed, seat = job
    shops = [random.Random(seed).choice(SHOPS) for _ in range(8)] if False else [SHOPS[random.Random(seed * 31 + i).randrange(8)] for i in range(8)]
    g = kagsim.Game(seed, 720, shops)
    while not g.done:
        x = A.agent(g.observe(seat)); y = B.agent(g.observe(1 - seat))
        g.step(*((x, y) if seat == 0 else (y, x)))
    return seed, seat, g.reward(seat) - g.reward(1 - seat), g.reward(seat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--seeds", type=int, default=40); ap.add_argument("--seed0", type=int, default=900000)
    ap.add_argument("--seats", default="0"); ap.add_argument("--jobs", type=int, default=11)
    ap.add_argument("--env", nargs="*", default=[])
    a = ap.parse_args()
    env = dict(kv.split("=", 1) for kv in a.env)
    jobs = [(s, seat) for s in range(a.seed0, a.seed0 + a.seeds) for seat in map(int, a.seats.split(","))]
    t0 = time.time()
    with Pool(a.jobs, initializer=init, initargs=(a.a, a.b, env)) as p:
        res = p.map(run, jobs)
    ms = [m for _, _, m, _ in res]; n = len(ms); mu = sum(ms) / n
    se = math.sqrt(sum((x - mu) ** 2 for x in ms) / max(1, n - 1) / n)
    print(f"{a.a} vs {a.b} {env}: n={n} mean {mu:+.0f} ± {se:.0f} (t={mu / se if se else 0:+.1f}) W {sum(x > 0 for x in ms)} L {sum(x < 0 for x in ms)} own {sum(o for *_, o in res) / n:.0f} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
