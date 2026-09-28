"""Kaggle: route sweep of our patched pairs against the public me4 mirror (2026-09-28). Shard over pairs."""
import glob, os, shutil, subprocess, sys, time
SHARD, NSHARDS = 1, 5
ROOT = os.environ.get("KAGGLE_ROOT", "/kaggle"); INPUT = os.path.join(ROOT, "input"); output = os.path.join(ROOT, "working"); DRY = os.environ.get("KAGGLE_DRY")


def first(suffix, prefer):
    m = sorted(glob.glob(os.path.join(INPUT, "**", suffix), recursive=True)); m = [x for x in m if prefer in x] or m
    assert m, suffix; return m[0]


code = first("code/route_sweep.py", "route-sweep"); D = os.path.dirname(os.path.dirname(code))
kagsim_src = os.path.dirname(os.path.dirname(first("sim/sim.hpp", "v41self")))
os.makedirs(output, exist_ok=True)
if os.environ.get("SKIP_BUILD"):
    env = dict(os.environ)
else:
    sim = os.path.join(output, "kagsim_src"); shutil.copytree(kagsim_src, sim, dirs_exist_ok=True)
    r = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=sim, capture_output=True, text=True); assert r.returncode == 0, r.stderr[-300:]
    env = dict(os.environ, PYTHONPATH=sim + ":" + os.environ.get("PYTHONPATH", ""))
env["OMP_NUM_THREADS"] = "1"
pairs = open(os.path.join(D, "code", "pairs.txt")).read().strip().split(",")
mine = pairs[SHARD::NSHARDS]
if DRY: mine = mine[:1]
print(f"shard {SHARD}/{NSHARDS}: {len(mine)} pairs", flush=True); t0 = time.time()
r = subprocess.run([sys.executable, code, "--us", os.path.join(D, "agents", "e085", "main.py"), "--opp", os.path.join(D, "agents", "pub_me4", "main.py"),
                    "--pairs", ",".join(mine), "--routes", "0,1,2" if DRY else "all", "--seeds", "1" if DRY else "3", "--seed0", "800000", "--seats", "0,1", "--baseline",
                    "--jobs", "4", "--out", os.path.join(output, f"sweep_{SHARD}.json")], env=env, cwd=output)
print("sweep rc", r.returncode, f"[{time.time() - t0:.0f}s]", flush=True)
if not os.environ.get("SKIP_BUILD"): shutil.rmtree(sim, ignore_errors=True)
print("DONE", flush=True)
