"""E058 GA: (1+λ) 山登り。変異演算子は evolve.py を再利用、適応度は fit58 (v41 chassis + ライブ層で L1)。
L1-pool (記録80席、偶数=train) で λ 候補をふるい、上位 --top を反応クローン付きで確認して採用。
使い方: .venv/bin/python tests/ga/evolve58.py --init agents/sr0909_base/actions.json --out tmp/e058/ga/best.json --iters 150 --lam 12"""
import argparse, json, random, sys, time
from collections import defaultdict
sys.path.insert(0, "tests/ga")
import evolve, fit58


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--init", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=150); ap.add_argument("--lam", type=int, default=12); ap.add_argument("--ops", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--top", type=int, default=2)
    a = ap.parse_args(); rng = random.Random(a.seed)
    pool = json.load(open(fit58.POOL)); eps = sorted({str(p["episode"]) for p in pool})
    train = [p for p in pool if eps.index(str(p["episode"])) % 2 == 0]; hold = [p for p in pool if eps.index(str(p["episode"])) % 2 == 1]
    ws = defaultdict(list)
    for p in pool:
        if p["lineage"] == "SR0909": ws[tuple(p["shops"][:2])].append(p["stream"])
    evolve._WORLD_STREAMS = ws  # splice donors: the current family's recorded streams
    best = json.load(open(a.init))
    bw = fit58.evaluate(best, pool=train); bws = fit58.score(bw)
    print(f"init {fit58.fmt(bw)} score {bws:+.0f}", flush=True)
    t0 = time.time(); acc = 0
    for it in range(a.iters):
        cands = [evolve.mutate(best, rng, a.ops) for _ in range(a.lam)]
        scored = sorted(((fit58.score(fit58.evaluate(c, live=False, pool=train)), i) for i, c in enumerate(cands)), reverse=True)
        for s0, i in scored[: a.top]:
            w = fit58.evaluate(cands[i], pool=train); s = fit58.score(w)
            if s > bws + 25:
                best, bws, bw = cands[i], s, w; acc += 1
                h = fit58.evaluate(best, pool=hold, seeds=range(2100, 2112))
                print(f"it {it} ACCEPT {fit58.fmt(w)} score {bws:+.0f} | holdout {fit58.fmt(h)} score {fit58.score(h):+.0f} [{time.time()-t0:.0f}s]", flush=True)
                json.dump(best, open(a.out, "w")); break
        if it % 10 == 0: print(f"it {it} best {bws:+.0f} [{time.time()-t0:.0f}s]", flush=True)
    print("accepted", acc)


if __name__ == "__main__":
    main()
