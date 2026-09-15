"""BC data v2: winner seats with rating >= R, from the replay parquet (episodes.csv metadata) and from replay JSON folders (winner = higher final bank).
Usage: .venv/bin/python rl/extract_winners.py --out tmp/rl/bcw --min-rating 2500 --parquet tmp/data/replays_2026-09.parquet --json 'tmp/top0915/**/episode-*.json' 'tmp/e061_live/*.json' 'tmp/e060_live/*.json'"""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, pyarrow.parquet as pq
from extract import extract_replay
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--min-rating", type=float, default=2500); ap.add_argument("--parquet"); ap.add_argument("--json", nargs="*", default=[])
ap.add_argument("--episodes-csv", default="tmp/data/episodes.csv"); ap.add_argument("--exclude-team", default="MMN0222", help="skip our own seats in JSON folders")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time(); done = 0
def save(r, seat, eid, tag):
    dst = os.path.join(a.out, f"{tag}_{eid}_{seat}.npz")
    if os.path.exists(dst) or len(r["steps"]) < 720: return 0
    np.savez_compressed(dst, **extract_replay(r, seat)); return 1
if a.parquet:
    ep = pd.read_csv(a.episodes_csv); ep = ep[ep["type"].str.contains("PUBLIC")]; want = {}
    for _, r in ep.iterrows():
        for s in (0, 1):
            if r[f"rating_{s}"] >= a.min_rating and r[f"bank_{s}"] > r[f"bank_{1-s}"]: want.setdefault(int(r.episode_id), []).append(s)
    pf = pq.ParquetFile(a.parquet); print("parquet candidate seats:", sum(len(v) for v in want.values()), flush=True)
    for g in range(pf.num_row_groups):
        ids = pf.read_row_group(g, columns=["episode_id"]).column(0).to_pylist(); hit = [i for i, e in enumerate(ids) if e in want]
        if not hit: continue
        tbl = pf.read_row_group(g)
        for i in hit:
            try: r = json.loads(tbl.column("replay_json")[i].as_py())
            except Exception: continue
            for s in want[ids[i]]: done += save(r, s, ids[i], "pq")
        if done % 100 < 2: print(f"{done} shards [{time.time()-t0:.0f}s]", flush=True)
for pat in a.json:
    for f in sorted(glob.glob(pat, recursive=True)):
        try: r = json.load(open(f))
        except Exception: continue
        if not isinstance(r, dict) or "steps" not in r or len(r["steps"]) < 720: continue
        names = r["info"].get("TeamNames", ["?", "?"]); banks = [r["steps"][-1][p]["observation"]["farms"][p]["money"] for p in (0, 1)]
        w = int(banks[1] > banks[0])
        if names[w] == a.exclude_team or banks[w] <= banks[1 - w]: continue
        done += save(r, w, r["id"], "js")
print(f"done {done} shards [{time.time()-t0:.0f}s] -> {a.out}")
