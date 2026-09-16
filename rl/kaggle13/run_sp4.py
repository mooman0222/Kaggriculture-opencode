"""Kaggle GPU: self-play PPO for Policy3 (rl/sp/train3.py) from bc13_ep3, then evaluate checkpoints against v41.

Inputs: rl-code dataset (patched rl/, flat), rl-bc7 dataset (code/kagsim, code/v41_main.py), rl-majkel0916 dataset (ckpt/bc13_ep3.pt,
tapes/tapes_top.pkl). A previous run of this kernel (kernel_sources) provides sp4k.pt.state for --resume.
"""
import glob
import os
import shutil
import subprocess
import sys
import time

output = "/kaggle/working"
MINUTES = 150           # 2.5 h of training per kernel run; chain runs with --resume (kernel_sources) instead of one long run
EVAL_EVERY = 25         # evaluate every saved checkpoint (train3 saves _itK.pt every 25 iters)
EVAL_GAMES = 32


def first(pattern):
    matches = sorted(glob.glob(pattern, recursive=True))
    if not matches:
        for root, dirs, files in os.walk("/kaggle/input"):
            if root.count("/") <= 5:
                print("input:", root, dirs[:8], files[:8], flush=True)
    assert matches, f"input not found: {pattern}"
    return matches[0]


rl = os.path.dirname(first("/kaggle/input/**/rollout3.py"))          # patched code (flat dataset)
kagsim_src = os.path.dirname(first("/kaggle/input/**/kagsim/setup.py"))
v41 = first("/kaggle/input/**/v41_main.py")
init = first("/kaggle/input/**/bc13_ep3.pt")
tapes = first("/kaggle/input/**/tapes_top.pkl")
print("rl:", rl, "kagsim:", kagsim_src, "init:", init, "tapes:", tapes, flush=True)

simulator = os.path.join(output, "kagsim_src")
shutil.copytree(kagsim_src, simulator, dirs_exist_ok=True)
result = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=simulator, capture_output=True, text=True)
print("kagsim build rc", result.returncode, result.stderr[-300:], flush=True)
assert result.returncode == 0
environment = dict(os.environ, PYTHONPATH=simulator + ":" + rl + ":" + os.path.join(rl, "sp"))

out = os.path.join(output, "sp4k.pt")
resume = []
prior = glob.glob("/kaggle/input/**/sp4k.pt.state", recursive=True)
if prior:
    shutil.copy(prior[0], out + ".state")
    for f in glob.glob(os.path.join(os.path.dirname(prior[0]), "sp4k*.pt")):
        shutil.copy(f, output)
    resume = ["--resume"]
    print("resuming from", prior[0], flush=True)

command = [sys.executable, os.path.join(rl, "sp", "train3.py"), "--init", init, "--out", out, "--dev", "cuda", "--workers", "4", "--games", "48", "--T", "96",
           "--lr", "1e-5", "--kl", "0.02", "--temp", "0.7", "--dense", "0.2", "--death", "0.05", "--escape", "0.2",
           "--tapes", tapes, "--tape-frac", "0.25", "--critic-warmup", "60", "--max-minutes", str(MINUTES), *resume]
started = time.time()
result = subprocess.run(command, cwd=rl, env=environment)
print("train rc", result.returncode, f"[{time.time() - started:.0f}s]", flush=True)

checkpoints = sorted(glob.glob(os.path.join(output, "sp4k_it*.pt")), key=lambda p: int(p.rsplit("_it", 1)[1][:-3]))
picked = [c for c in checkpoints if int(c.rsplit("_it", 1)[1][:-3]) % EVAL_EVERY == 0] or checkpoints[-1:]
with open(os.path.join(output, "eval.txt"), "a") as report:
    for checkpoint in picked + [out]:
        result = subprocess.run([sys.executable, os.path.join(rl, "play3.py"), checkpoint, "--games", str(EVAL_GAMES), "--vs", v41], cwd=rl, env=environment, capture_output=True, text=True)
        line = (result.stdout.strip().splitlines() or [result.stderr[-300:]])[-1]
        print(os.path.basename(checkpoint), line, flush=True)
        report.write(f"{os.path.basename(checkpoint)} {line}\n")
print("ALL DONE", flush=True)
