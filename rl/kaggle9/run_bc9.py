"""Kaggle GPU pilot for Policy3 joint destination-operation options."""
import glob
import os
import shutil
import subprocess
import sys
import time


def parent_with(pattern, levels):
    matches = glob.glob(pattern, recursive=True)
    assert matches, f"input not found: {pattern}"
    path = matches[0]
    for _ in range(levels):
        path = os.path.dirname(path)
    return path


bc7_root = parent_with("/kaggle/input/**/data/bc7data", 2)
initial = glob.glob("/kaggle/input/**/ckpt/bc5_ep3.pt", recursive=True)[0]
code = os.path.join(bc7_root, "code")
rl = os.path.join(code, "rl")
data = os.path.join(bc7_root, "data", "bc7data")
output = "/kaggle/working"
print("root:", bc7_root, "shards:", len(glob.glob(os.path.join(data, "*.npz"))), "init:", initial, flush=True)

have_simulator = False
try:
    import pybind11  # noqa: F401

    simulator = os.path.join(output, "kagsim_src")
    shutil.copytree(os.path.join(code, "kagsim"), simulator, dirs_exist_ok=True)
    result = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=simulator, capture_output=True, text=True)
    print("kagsim build rc", result.returncode, result.stderr[-400:], flush=True)
    have_simulator = result.returncode == 0
except Exception as exc:
    print("kagsim build skipped:", exc, flush=True)

command = [
    sys.executable,
    os.path.join(rl, "train_bc3.py"),
    "--data", os.path.join(data, "*.npz"),
    "--out", os.path.join(output, "bc9k.pt"),
    "--init", initial,
    "--epochs", "4",
    "--bs", "128",
    "--lr", "5e-4",
    "--max-games", "1200",
]
started = time.time()
result = subprocess.run(command, cwd=rl)
print("train rc", result.returncode, f"[{time.time() - started:.0f}s]", flush=True)

if have_simulator and result.returncode == 0:
    environment = dict(os.environ, PYTHONPATH=simulator + ":" + rl)
    with open(os.path.join(output, "eval.txt"), "w") as report:
        for checkpoint in sorted(glob.glob(os.path.join(output, "bc9k_ep*.pt"))):
            result = subprocess.run(
                [sys.executable, os.path.join(rl, "play3.py"), checkpoint, "--games", "16", "--vs", os.path.join(code, "v41_main.py")],
                cwd=rl,
                env=environment,
                capture_output=True,
                text=True,
            )
            line = (result.stdout.strip().splitlines() or [result.stderr[-300:]])[-1]
            print(os.path.basename(checkpoint), line, flush=True)
            report.write(f"{os.path.basename(checkpoint)} {line}\n")
print("ALL DONE", flush=True)