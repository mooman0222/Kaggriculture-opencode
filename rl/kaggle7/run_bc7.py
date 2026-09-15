"""Kaggle GPU run: behaviour cloning bc7 (winners + quality losers) + closed-loop eval vs v41 in kagsim. Outputs in /kaggle/working."""
import glob, os, subprocess, sys, shutil, time
root = None
for cand in glob.glob("/kaggle/input/**/train_bc2.py", recursive=True): root = os.path.dirname(os.path.dirname(os.path.dirname(cand))); break
assert root, "dataset not found"; print("dataset root:", root, flush=True)
code = os.path.join(root, "code"); rl = os.path.join(code, "rl"); data = os.path.join(root, "data", "bc7data"); out = "/kaggle/working"
print("shards:", len(glob.glob(os.path.join(data, "*.npz"))), flush=True)
# optional: build kagsim for the closed-loop eval (needs pybind11; Kaggle images ship it in most cases)
have_sim = False
try:
    import pybind11  # noqa
    ksrc = os.path.join(out, "kagsim_src"); shutil.rmtree(ksrc, ignore_errors=True); shutil.copytree(os.path.join(code, "kagsim"), ksrc)
    r = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=ksrc, capture_output=True, text=True); print("kagsim build rc", r.returncode, r.stderr[-400:], flush=True)
    have_sim = r.returncode == 0
except Exception as e: print("kagsim build skipped:", e, flush=True)
# resume support: a previous run's output attached as a dataset provides bc7k.pt.state
prev = glob.glob("/kaggle/input/**/bc7k.pt.state", recursive=True)
if prev: shutil.copy(prev[0], os.path.join(out, "bc7k.pt.state")); print("resuming from", prev[0], flush=True)
cmd = [sys.executable, os.path.join(rl, "train_bc2.py"), "--data", os.path.join(data, "*.npz"), "--out", os.path.join(out, "bc7k.pt"), "--epochs", "6", "--bs", "512", "--lr", "5e-4"] + (["--resume"] if prev else [])
t0 = time.time(); r = subprocess.run(cmd, cwd=rl); print("train rc", r.returncode, f"[{time.time()-t0:.0f}s]", flush=True)
if have_sim:
    env = dict(os.environ, PYTHONPATH=ksrc + ":" + rl)
    with open(os.path.join(out, "eval.txt"), "a") as f:
        for ck in sorted(glob.glob(os.path.join(out, "bc7k_ep*.pt"))):
            r = subprocess.run([sys.executable, os.path.join(rl, "play2.py"), ck, "--games", "32", "--vs", os.path.join(code, "v41_main.py")], cwd=rl, env=env, capture_output=True, text=True)
            line = (r.stdout.strip().splitlines() or [r.stderr[-300:]])[-1]; print(os.path.basename(ck), line, flush=True); f.write(f"{os.path.basename(ck)} {line}\n")
print("ALL DONE", flush=True)
