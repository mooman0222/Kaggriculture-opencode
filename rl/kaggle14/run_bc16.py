"""Kaggle GPU: bc16 = continue bc15_ep3 on the same Majkel 713 shards for 4 more epochs (lr 1e-4), evaluate each epoch vs v41 (32 games)."""
import glob
import os
import shutil
import subprocess
import sys
import time

output = "/kaggle/working"


def first(pattern):
    matches = sorted(glob.glob(pattern, recursive=True))
    if not matches:
        for root, dirs, files in os.walk("/kaggle/input"):
            if root.count("/") <= 5:
                print("input:", root, dirs[:8], files[:8], flush=True)
    assert matches, f"input not found: {pattern}"
    return matches[0]


rl = os.path.dirname(first("/kaggle/input/**/rollout3.py"))
kagsim_src = os.path.dirname(first("/kaggle/input/**/kagsim/setup.py"))
v41 = first("/kaggle/input/**/v41_main.py")
init = first("/kaggle/input/**/bc15_ep3.pt")
data = os.path.dirname(first("/kaggle/input/**/majkel_all/*.npz"))
print("rl:", rl, "init:", init, "shards:", len(glob.glob(os.path.join(data, "*.npz"))), flush=True)

simulator = os.path.join(output, "kagsim_src")
shutil.copytree(kagsim_src, simulator, dirs_exist_ok=True)
result = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=simulator, capture_output=True, text=True)
print("kagsim build rc", result.returncode, result.stderr[-300:], flush=True)
have_simulator = result.returncode == 0

command = [sys.executable, os.path.join(rl, "train_bc3.py"), "--data", os.path.join(data, "*.npz"), "--out", os.path.join(output, "bc16k.pt"),
           "--init", init, "--epochs", "4", "--bs", "128", "--lr", "1e-4", "--max-games", "100000"]
started = time.time()
result = subprocess.run(command, cwd=rl)
print("train rc", result.returncode, f"[{time.time() - started:.0f}s]", flush=True)

if have_simulator and result.returncode == 0:
    environment = dict(os.environ, PYTHONPATH=simulator + ":" + rl)
    with open(os.path.join(output, "eval.txt"), "w") as report:
        for checkpoint in sorted(glob.glob(os.path.join(output, "bc16k_ep*.pt"))):
            result = subprocess.run([sys.executable, os.path.join(rl, "play3.py"), checkpoint, "--games", "32", "--vs", v41], cwd=rl, env=environment, capture_output=True, text=True)
            line = (result.stdout.strip().splitlines() or [result.stderr[-300:]])[-1]
            print(os.path.basename(checkpoint), line, flush=True)
            report.write(f"{os.path.basename(checkpoint)} {line}\n")
print("ALL DONE", flush=True)
