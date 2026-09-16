"""Continue Policy3 (bc9k_ep3) for 4 epochs on the same 1,200 demo games plus closed-loop correction shards (rl/rollout3.py, dataset kaggriculture-rl-dagger3)."""
import glob
import os
import shutil
import subprocess
import sys
import time


def parent_with(pattern, levels):
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        for root, dirs, files in os.walk("/kaggle/input"):
            if root.count("/") <= 4:
                print("input:", root, dirs[:8], files[:8], flush=True)
    assert matches, f"input not found: {pattern}"
    path = matches[0]
    for _ in range(levels):
        path = os.path.dirname(path)
    return path


bc7_root = parent_with("/kaggle/input/**/data/bc7data", 2)
initial = glob.glob("/kaggle/input/**/bc9k_ep3.pt", recursive=True)[0]
code = os.path.join(bc7_root, "code")
rl = parent_with("/kaggle/input/**/rollout3.py", 1)  # the rl-code dataset is mounted flat (no rl/ folder); rollout3.py is absent from the stale bc7 copy
import random
output = "/kaggle/working"
data = os.path.join(output, "data12")
os.makedirs(data, exist_ok=True)
demo = sorted(glob.glob(os.path.join(bc7_root, "data", "bc7data", "*.npz")))
random.Random(0).shuffle(demo)
dagger = sorted(glob.glob("/kaggle/input/**/dg_*.npz", recursive=True))
for path in demo[:1200] + dagger:
    link = os.path.join(data, os.path.basename(path))
    if not os.path.exists(link):
        os.symlink(path, link)
print("demo shards", min(1200, len(demo)), "correction shards", len(dagger), flush=True)
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
    "--out", os.path.join(output, "bc12k.pt"),
    "--init", initial,
    "--epochs", "4",
    "--bs", "128",
    "--lr", "2e-4",
    "--max-games", "100000",
]
started = time.time()
result = subprocess.run(command, cwd=rl)
print("train rc", result.returncode, f"[{time.time() - started:.0f}s]", flush=True)

if have_simulator and result.returncode == 0:
    environment = dict(os.environ, PYTHONPATH=simulator + ":" + rl)
    with open(os.path.join(output, "eval.txt"), "w") as report:
        for checkpoint in sorted(glob.glob(os.path.join(output, "bc12k_ep*.pt"))):
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