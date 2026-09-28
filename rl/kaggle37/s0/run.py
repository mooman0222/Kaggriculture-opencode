"""Kaggle: hold-oracle data generation for the learned HOLD layer (2026-09-28).

Each kernel takes one shard of the tapes (SHARD of NSHARDS, both seats of the 09-27 top-20 games plus our
09-27 games) and runs tests/overlay_headroom.py --family hold --feat (hindsight greedy + per-day features),
then a block of reactive games vs the public me4 chassis. Outputs hold_tapes_<shard>.json / hold_me4_<shard>.json.
Dry: SKIP_BUILD=1 KAGGLE_ROOT=<fake tree> KAGGLE_DRY=1 python run_hold_oracle.py
"""
import glob, os, shutil, subprocess, sys, time
SHARD, NSHARDS = 0, 5
ROOT = os.environ.get("KAGGLE_ROOT", "/kaggle"); INPUT = os.path.join(ROOT, "input"); output = os.path.join(ROOT, "working"); DRY = os.environ.get("KAGGLE_DRY")
JOBS = 2 if DRY else 4


def first(suffix, prefer):
    m = sorted(glob.glob(os.path.join(INPUT, "**", suffix), recursive=True)); m = [x for x in m if prefer in x] or m
    assert m, suffix; return m[0]


code = first("code/overlay_headroom.py", "hold-oracle"); D = os.path.dirname(os.path.dirname(code))
kagsim_src = os.path.dirname(os.path.dirname(first("sim/sim.hpp", "v41self")))
os.makedirs(output, exist_ok=True)
if os.environ.get("SKIP_BUILD"):
    env = dict(os.environ)
else:
    sim = os.path.join(output, "kagsim_src"); shutil.copytree(kagsim_src, sim, dirs_exist_ok=True)
    r = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=sim, capture_output=True, text=True); assert r.returncode == 0, r.stderr[-300:]
    env = dict(os.environ, PYTHONPATH=sim + ":" + os.environ.get("PYTHONPATH", ""))
env["OMP_NUM_THREADS"] = "1"

tapes = sorted(glob.glob(os.path.join(D, "tapes", "slim0927", "*.json"))) + sorted(glob.glob(os.path.join(D, "tapes", "late0928", "*.json")))
mine = tapes[SHARD::NSHARDS]
if DRY: mine = mine[:1]
shard_dir = os.path.join(output, "shard"); os.makedirs(shard_dir, exist_ok=True)
for f in mine: shutil.copy(f, shard_dir)
print(f"shard {SHARD}/{NSHARDS}: {len(mine)} replays (both seats; our own games use --team via separate pass)", flush=True)
t0 = time.time()
base = os.path.join(D, "agents", "e087", "main.py"); me4 = os.path.join(D, "agents", "pub_me4", "main.py")
days = "12-14" if DRY else "9-29"
# 1) tapes: top-20 games → both seats; our games → only our seat (the other seat is us, no point)
top = [f for f in mine if "late0928" not in f]
if top:
    tdir = os.path.join(shard_dir, "top"); os.makedirs(tdir, exist_ok=True)
    for f in top: shutil.move(os.path.join(shard_dir, os.path.basename(f)), tdir)
    r = subprocess.run([sys.executable, code, base, "--replays", os.path.join(tdir, "episode-*.json"), "--family", "hold", "--days", days, "--feat",
                        "--jobs", str(JOBS), "--out", os.path.join(output, f"hold_tapes_{SHARD}.json")], env=env, cwd=output)
    print("tapes rc", r.returncode, f"[{time.time() - t0:.0f}s]", flush=True)
ours = [f for f in mine if "late0928" in f]
if ours:
    r = subprocess.run([sys.executable, code, base, "--replays", os.path.join(shard_dir, "episode-*.json"), "--team", "MMN0222", "--family", "hold", "--days", days, "--feat",
                        "--jobs", str(JOBS), "--out", os.path.join(output, f"hold_ours_{SHARD}.json")], env=env, cwd=output)
    print("ours rc", r.returncode, f"[{time.time() - t0:.0f}s]", flush=True)
# 2) reactive games vs the public me4 chassis (distinct seeds per shard)
r = subprocess.run([sys.executable, code, base, "--vs", me4, "--seeds", "2" if DRY else "24", "--seed0", str(6000 + 100 * SHARD), "--family", "hold", "--days", days, "--feat",
                    "--jobs", str(JOBS), "--out", os.path.join(output, f"hold_me4_{SHARD}.json")], env=env, cwd=output)
print("me4 rc", r.returncode, f"[{time.time() - t0:.0f}s]", flush=True)
shutil.rmtree(shard_dir, ignore_errors=True)
if not os.environ.get("SKIP_BUILD"): shutil.rmtree(sim, ignore_errors=True)
print("DONE", flush=True)
