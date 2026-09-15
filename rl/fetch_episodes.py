"""提出 id のエピソードを Kaggle から取り、対象チームの席を shard 化して JSON は捨てる (ストリーミング)。
使い方: .venv/bin/python rl/fetch_episodes.py --sub 56156662 --team Majkel1337 --out tmp/rl/majkel [--n 400]"""
import argparse, glob, json, os, shlex, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from extract import extract_replay
K = shlex.split(os.environ.get("KAGGLE_BIN", ".venv/bin/kaggle"))
ap = argparse.ArgumentParser(); ap.add_argument("--sub", required=True); ap.add_argument("--team", required=True); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=400)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
ids = [l.split()[0] for l in subprocess.run([*K, "competitions", "episodes", a.sub], capture_output=True, text=True).stdout.split("\n")[2:] if l.strip() and l.split()[0].isdigit()][: a.n]
have = {os.path.basename(p)[:-4] for p in glob.glob(a.out + "/*.npz")}
tmp = tempfile.mkdtemp(prefix="ep_"); done = 0; skipped = 0
for eid in ids:
    if eid in have: skipped += 1; continue
    subprocess.run([*K, "competitions", "replay", eid, "-p", tmp], capture_output=True)
    fs = glob.glob(f"{tmp}/episode-{eid}-replay.json")
    if not fs: continue
    try:
        r = json.load(open(fs[0])); names = r["info"]["TeamNames"]
        if a.team in names and len(r["steps"]) >= 720:
            np.savez_compressed(os.path.join(a.out, f"{eid}.npz"), **extract_replay(r, names.index(a.team))); done += 1
    except Exception as e:
        print("skip", eid, e, flush=True)
    os.remove(fs[0])
    if done % 25 == 0 and done: print(f"{a.team}: {done} shards", flush=True)
print(f"{a.team}: +{done} shards (had {skipped}), total {len(glob.glob(a.out + '/*.npz'))}")
