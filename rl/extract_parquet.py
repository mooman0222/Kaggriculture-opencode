"""公開データセットの月次 parquet から上位チームの席を shard 化。使い方: .venv/bin/python rl/extract_parquet.py --parquet tmp/data/replays_2026-09.parquet --top 40 --out tmp/rl/pq"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, pyarrow.parquet as pq
from extract import extract_replay
ap = argparse.ArgumentParser(); ap.add_argument("--parquet", required=True); ap.add_argument("--top", type=int, default=40); ap.add_argument("--out", required=True); ap.add_argument("--teams-csv", default="tmp/data/teams.csv"); ap.add_argument("--episodes-csv", default="tmp/data/episodes.csv")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
teams = pd.read_csv(a.teams_csv).sort_values("ladder_score", ascending=False).head(a.top); top = set(teams.team_id); name = dict(zip(teams.team_id, teams.team_name))
ep = pd.read_csv(a.episodes_csv); ep = ep[ep["type"].str.contains("PUBLIC")]
want = {}
for _, r in ep.iterrows():
    seats = [s for s in (0, 1) if r[f"team_{s}"] in top]
    if seats: want[int(r.episode_id)] = [(s, int(r[f"team_{s}"])) for s in seats]
print("episodes with a top team seat:", len(want), flush=True)
pf = pq.ParquetFile(a.parquet); t0 = time.time(); done = 0
for g in range(pf.num_row_groups):
    ids = pf.read_row_group(g, columns=["episode_id"]).column(0).to_pylist()
    hit = [i for i, e in enumerate(ids) if e in want]
    if not hit: continue
    tbl = pf.read_row_group(g)
    for i in hit:
        eid = ids[i]
        for seat, tid in want[eid]:
            dst = os.path.join(a.out, f"{tid}", f"{eid}.npz")
            if os.path.exists(dst): continue
            try:
                r = json.loads(tbl.column("replay_json")[i].as_py())
                if len(r["steps"]) < 720: continue
                os.makedirs(os.path.dirname(dst), exist_ok=True); np.savez_compressed(dst, **extract_replay(r, seat)); done += 1
            except Exception as e: print("skip", eid, e, flush=True)
    if done and done % 100 == 0: print(f"{done} shards [{time.time()-t0:.0f}s] rg {g}/{pf.num_row_groups}", flush=True)
print(f"done {done} shards [{time.time()-t0:.0f}s]")
json.dump({str(k): v for k, v in name.items()}, open(os.path.join(a.out, "team_names.json"), "w"), ensure_ascii=False)
