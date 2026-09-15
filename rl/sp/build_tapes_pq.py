"""Large recorded-opponent pool from the public replay parquet (all September episodes): seats whose rating >= --min-rating.
Usage: .venv/bin/python rl/sp/build_tapes_pq.py --min-rating 2800 --max-per-team 400 --out tmp/rl/tapes_top.pkl"""
import argparse, csv, json, pickle
from collections import Counter
import pyarrow.parquet as pq
ap = argparse.ArgumentParser(); ap.add_argument("--parquet", default="tmp/data/replays_2026-09.parquet"); ap.add_argument("--episodes", default="tmp/data/episodes.csv")
ap.add_argument("--min-rating", type=float, default=2800); ap.add_argument("--max-per-team", type=int, default=400); ap.add_argument("--out", default="tmp/rl/tapes_top.pkl"); ap.add_argument("--limit", type=int, default=6000)
a = ap.parse_args()
want = {}  # episode_id -> list of (seat, team, rating)
for r in csv.DictReader(open(a.episodes)):
    for s in (0, 1):
        try: rt = float(r[f"rating_{s}"])
        except Exception: continue
        if rt >= a.min_rating: want.setdefault(int(r["episode_id"]), []).append((s, r[f"team_{s}"], rt))
print(f"candidate episodes {len(want)}, seats {sum(len(v) for v in want.values())}", flush=True)
PASS = {"farmer": ["PASS"], "hands": [], "market": []}
f = pq.ParquetFile(a.parquet); tapes = []; cnt = Counter(); scanned = 0
for rg in range(f.metadata.num_row_groups):
    ids = f.read_row_group(rg, columns=["episode_id"]).column(0).to_pylist()
    if not any(i in want for i in ids): continue
    tb = f.read_row_group(rg); 
    for eid, rj in zip(tb.column("episode_id").to_pylist(), tb.column("replay_json").to_pylist()):
        if eid not in want: continue
        r = json.loads(rj); seed = r.get("info", {}).get("seed"); steps = r.get("steps", [])
        if not seed or len(steps) != 720: continue
        for seat, team, rt in want[eid]:
            if cnt[team] >= a.max_per_team: continue
            acts = [steps[t][seat]["action"] or PASS for t in range(1, 720)]
            tapes.append({"seed": int(seed), "seat": seat, "team": team, "rating": rt, "final": float(steps[-1][seat]["observation"]["farms"][seat]["money"]), "acts": acts}); cnt[team] += 1
        scanned += 1
    if len(tapes) >= a.limit: break
pickle.dump(tapes, open(a.out, "wb"))
print(f"{len(tapes)} tapes from {scanned} episodes -> {a.out}; teams: {cnt.most_common(10)}")
