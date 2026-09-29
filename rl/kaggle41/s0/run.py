"""Kaggle: seat-swap of five candidates on the rank 50-110 band tapes (both seats), 2026-09-29."""
import glob, os, shutil, subprocess, sys, time
ROOT = os.environ.get("KAGGLE_ROOT", "/kaggle"); INPUT = os.path.join(ROOT, "input"); output = os.path.join(ROOT, "working")
def first(suffix, prefer):
    m = sorted(glob.glob(os.path.join(INPUT, "**", suffix), recursive=True)); m = [x for x in m if prefer in x] or m
    assert m, suffix; return m[0]
code = first("code/swap_eval.py", "band-swap"); D = os.path.dirname(os.path.dirname(code))
kagsim_src = os.path.dirname(os.path.dirname(first("sim/sim.hpp", "v41self")))
os.makedirs(output, exist_ok=True)
sim = os.path.join(output, "kagsim_src"); shutil.copytree(kagsim_src, sim, dirs_exist_ok=True)
r = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=sim, capture_output=True, text=True); assert r.returncode == 0, r.stderr[-300:]
env = dict(os.environ, PYTHONPATH=sim, OMP_NUM_THREADS="1")
cands = [os.path.join(D, "agents", a, "main.py") for a in ("e087", "e081", "e082", "e089", "e090")]
t0 = time.time()
r = subprocess.run([sys.executable, code, "--replays", os.path.join(D, "tapes", "episode-*.json"), "--cands", *cands, "--jobs", "4", "--quiet", "--out", os.path.join(output, "band_swap.json")], env=env, cwd=output)
print("rc", r.returncode, f"[{time.time() - t0:.0f}s]", flush=True)
shutil.rmtree(sim, ignore_errors=True); print("DONE", flush=True)
