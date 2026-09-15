"""BC data v5: quality loser seats (recovery demos) from the replay parquet + all opponent seats from live JSON folders.
Usage: .venv/bin/python rl/extract_losers.py --out tmp/rl/bcl --parquet tmp/data/replays_2026-09.parquet --json 'tmp/e058_live/episode-*.json' 'tmp/e060_live/episode-*.json'
(parquet losers: rating>=2500, lost, bank>=80k; JSON: every non-MMN0222 seat, tag jsl/jsw by loss/win)"""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, pyarrow.parquet as pq
from extract import extract_replay

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--parquet", default="tmp/data/replays_2026-09.parquet")
ap.add_argument("--episodes-csv", default="tmp/data/episodes.csv")
ap.add_argument("--json", nargs="*", default=[])
ap.add_argument("--min-rating", type=float, default=2500)
ap.add_argument("--min-bank", type=float, default=80000)
ap.add_argument("--exclude-team", default="MMN0222")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
t0 = time.time(); done = 0


def save(r, seat, eid, tag):
    global done
    dst = os.path.join(a.out, f"{tag}_{eid}_{seat}.npz")
    if os.path.exists(dst) or len(r.get("steps", [])) < 720:
        return
    np.savez_compressed(dst, **extract_replay(r, seat))
    done += 1


if a.parquet:
    ep = pd.read_csv(a.episodes_csv); ep = ep[ep["type"].str.contains("PUBLIC")]
    want = {}
    for _, r in ep.iterrows():
        for s in (0, 1):
            if r[f"rating_{s}"] >= a.min_rating and r[f"bank_{s}"] < r[f"bank_{1 - s}"] and r[f"bank_{s}"] >= a.min_bank:
                want.setdefault(int(r.episode_id), []).append(s)
    print("parquet loser candidate seats:", sum(len(v) for v in want.values()), flush=True)
    pf = pq.ParquetFile(a.parquet)
    for g in range(pf.num_row_groups):
        ids = pf.read_row_group(g, columns=["episode_id"]).column(0).to_pylist()
        hit = [i for i, e in enumerate(ids) if e in want]
        if not hit:
            continue
        tbl = pf.read_row_group(g)
        for i in hit:
            try:
                r = json.loads(tbl.column("replay_json")[i].as_py())
            except Exception:
                continue
            for s in want[ids[i]]:
                save(r, s, ids[i], "pql")
        if done % 100 < 2:
            print(f"{done} shards [{time.time()-t0:.0f}s]", flush=True)
for pat in a.json:
    for f in sorted(glob.glob(pat, recursive=True)):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if not isinstance(r, dict) or "steps" not in r or len(r["steps"]) < 720:
            continue
        names = r["info"].get("TeamNames", ["?", "?"])
        banks = [r["steps"][-1][p]["observation"]["farms"][p]["money"] for p in (0, 1)]
        for s in (0, 1):
            if names[s] == a.exclude_team:
                continue
            save(r, s, r.get("id", os.path.basename(f).split("-")[1]), "jsl" if banks[s] < banks[1 - s] else "jsw")
print(f"done {done} shards [{time.time()-t0:.0f}s] -> {a.out}")
