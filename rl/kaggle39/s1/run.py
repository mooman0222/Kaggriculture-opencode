"""Kaggle: held-out verification of the swept route candidates against the public me4 mirror (2026-09-28).
Shard over pairs; candidates from code/cands.json (sweep_report --json); seeds 900000+ (never used in the sweep)."""
import glob, json, os, shutil, subprocess, sys, time
SHARD, NSHARDS = 1, 3
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
cands = json.load(open(os.path.join(D, "code", "cands.json")))
pairs = sorted(cands); mine = pairs[SHARD::NSHARDS]
if DRY: mine = mine[:1]
sub = {p: cands[p] for p in mine}; cj = os.path.join(output, "cands_shard.json"); json.dump(sub, open(cj, "w"))
print(f"shard {SHARD}/{NSHARDS}: {len(mine)} pairs", flush=True); t0 = time.time()
r = subprocess.run([sys.executable, code, "--us", os.path.join(D, "agents", "e085", "main.py"), "--opp", os.path.join(D, "agents", "pub_me4", "main.py"),
                    "--pairs", ",".join(mine), "--cands", cj, "--top", "2", "--seeds", "1" if DRY else "5", "--seed0", "900000", "--seats", "0,1", "--baseline",
                    "--jobs", "4", "--out", os.path.join(output, f"verify_{SHARD}.json")], env=env, cwd=output)
print("verify rc", r.returncode, f"[{time.time() - t0:.0f}s]", flush=True)
if not os.environ.get("SKIP_BUILD"): shutil.rmtree(sim, ignore_errors=True)
print("DONE", flush=True)
