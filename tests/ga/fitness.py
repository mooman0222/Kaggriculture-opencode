"""13本テープ (0909 形式) の適応度: プール全対戦を kagsim L0 (固定 vs 固定) で回し、系統別に集計する。"""
import json, sys, time, importlib.util
from collections import defaultdict
import kagsim
sys.path.insert(0, "agents/sr0909_base")
_r = importlib.util.spec_from_file_location("sr0909_router_f", "agents/sr0909_base/main.py"); router = importlib.util.module_from_spec(_r); _r.loader.exec_module(router)
SHOP_PLANS, ROUTE_STEP, FINAL = router.SHOP_PLANS, router.ROUTE_STEP, router.FINAL_PLAN_STEP


def build_stream(tapes, shops):
    plan = SHOP_PLANS.get(tuple(shops[:2]), 0)
    return tapes[plan][:FINAL] + tapes[2][FINAL:]


class Fitness:
    WEIGHTS = {"SR0909live": 0.35, "E055live": 0.25, "SR0909": 0.15, "other": 0.12, "ThomasOpen": 0.05, "Griefer": 0.05, "SR0908": 0.02, "KaitoOpen": 0.01}

    def __init__(self, pool_path=("tmp/ga/pool.json", "tmp/ga/pool_live.json"), lineages=None, max_per_lineage=None, split=None):
        pool = [p for path in ((pool_path,) if isinstance(pool_path, str) else pool_path) for p in json.load(open(path))]
        if split:  # "train" = even episode index, "holdout" = odd (by distinct episode)
            eps = sorted({str(p["episode"]) for p in pool}); keep = {e for i, e in enumerate(eps) if (i % 2 == 0) == (split == "train")}
            pool = [p for p in pool if str(p["episode"]) in keep]
        if lineages: pool = [p for p in pool if p["lineage"] in lineages]
        if max_per_lineage:
            cnt = defaultdict(int); kept = []
            for p in pool:
                if cnt[p["lineage"]] < max_per_lineage: kept.append(p); cnt[p["lineage"]] += 1
            pool = kept
        self.pool = pool
        self.opp = [kagsim.Stream(p["stream"]) for p in pool]

    def evaluate(self, tapes, weights=None):
        """returns dict: mean margin, win rate, per-lineage, and per-game margins."""
        streams = {}
        jobs = []
        for p in self.pool:
            key = tuple(p["shops"][:2])
            if key not in streams: streams[key] = kagsim.Stream(build_stream(tapes, p["shops"]))
            me = 1 - p["seat"]
            a, b = (streams[key], None) if me == 0 else (None, streams[key])
            jobs.append((streams[key], self.opp[len(jobs)], p["seed"]) if me == 0 else (self.opp[len(jobs)], streams[key], p["seed"]))
        res = kagsim.run_many(jobs)
        margins = []; by = defaultdict(list)
        for p, (b0, b1) in zip(self.pool, res):
            me = 1 - p["seat"]; m = (b0 - b1) if me == 0 else (b1 - b0)
            margins.append(m); by[p["lineage"]].append(m)
        lin_mean = {k: sum(v) / len(v) for k, v in by.items()}
        lin_wr = {k: sum(1 for m in v if m > 0) / len(v) for k, v in by.items()}
        # lineage-balanced score so 76 SR0909 games don't drown the rest
        bal = sum(lin_mean.values()) / len(lin_mean)
        w = {k: self.WEIGHTS.get(k, 0.05) for k in lin_mean}; ws = sum(w.values())
        weighted = sum(w[k] * lin_mean[k] for k in lin_mean) / ws
        wwr = sum(w[k] * lin_wr[k] for k in lin_mean) / ws
        return {"mean": sum(margins) / len(margins), "wr": sum(1 for m in margins if m > 0) / len(margins),
                "balanced": bal, "weighted": weighted, "wwr": wwr, "score": weighted + 3000 * wwr,
                "lin_mean": lin_mean, "lin_wr": lin_wr, "margins": margins}


if __name__ == "__main__":
    tapes = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "agents/sr0909_base/actions.json"))
    f = Fitness(); t0 = time.time(); r = f.evaluate(tapes)
    print(f"pool {len(f.pool)} games in {time.time()-t0:.2f}s: mean {r['mean']:+.0f} wr {r['wr']:.2f} balanced {r['balanced']:+.0f}")
    for k in r["lin_mean"]: print(f"  {k:<11} n={len([p for p in f.pool if p['lineage']==k]):3d} mean {r['lin_mean'][k]:+8.0f} wr {r['lin_wr'][k]:.2f}")
