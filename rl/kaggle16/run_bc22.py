"""Kaggle GPU (夜間): bc22 = Majkel 854 局のみ・初期値なしで 4 epoch lr 5e-4 (bc20 レシピ)。
学習後、bc22 の各 epoch と既存の基準 3 本 (bc15/bc18/bc20_ep3) を対 v41 **64 戦**で評価する。

ねらい: 09-17 の結論はすべて 32 戦 (SE ±3.1k) の中に収まっており順位がつかない。
64 戦で SE を半分にし、同時に別ハードで bc20 (92.1k) を再現するかを見る。
乾式実行: KAGGLE_ROOT=<偽ツリー> KAGGLE_DRY=1 python run_bc22.py
"""
import glob
import os
import shutil
import subprocess
import sys
import time

ROOT = os.environ.get("KAGGLE_ROOT", "/kaggle")
INPUT = os.path.join(ROOT, "input")
output = os.path.join(ROOT, "working")
DRY = os.environ.get("KAGGLE_DRY")
GAMES = "2" if DRY else "64"


def first(suffix):
    matches = sorted(glob.glob(os.path.join(INPUT, "**", suffix), recursive=True))
    if not matches:
        for root, dirs, files in os.walk(INPUT):
            if root.count("/") <= INPUT.count("/") + 3:
                print("input:", root, dirs[:8], files[:8], flush=True)
    assert matches, f"input not found: {suffix}"
    return matches[0]


rl = os.path.dirname(first("rollout3.py"))
kagsim_src = os.path.dirname(first("kagsim/setup.py"))
v41 = first("v41_main.py")
majkel = os.path.dirname(first("majkel_all/*.npz"))
refs = sorted(glob.glob(os.path.join(os.path.dirname(first("ckpt/bc18_ep3.pt")), "*.pt")))
print("rl:", rl, "shards:", len(glob.glob(os.path.join(majkel, "*.npz"))), "refs:", [os.path.basename(p) for p in refs], flush=True)

os.makedirs(output, exist_ok=True)
simulator = os.path.join(output, "kagsim_src")
shutil.copytree(kagsim_src, simulator, dirs_exist_ok=True)
result = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=simulator, capture_output=True, text=True)
print("kagsim build rc", result.returncode, result.stderr[-300:], flush=True)
have_simulator = result.returncode == 0

command = [sys.executable, os.path.join(rl, "train_bc3.py"), "--data", os.path.join(majkel, "*.npz"),
           "--out", os.path.join(output, "bc22.pt"), "--epochs", "1" if DRY else "4", "--bs", "128",
           "--lr", "5e-4", "--max-games", "4" if DRY else "100000"]   # --init なし = スクラッチ
started = time.time()
result = subprocess.run(command, cwd=rl)
print(f"bc22 rc {result.returncode} [{time.time() - started:.0f}s]", flush=True)
assert result.returncode == 0

if have_simulator:
    environment = dict(os.environ, PYTHONPATH=simulator + ":" + rl)
    with open(os.path.join(output, "eval.txt"), "w") as report:
        for checkpoint in sorted(glob.glob(os.path.join(output, "bc22_ep*.pt"))) + refs:
            started = time.time()
            result = subprocess.run([sys.executable, os.path.join(rl, "play3.py"), checkpoint, "--games", GAMES, "--vs", v41],
                                    cwd=rl, env=environment, capture_output=True, text=True)
            line = (result.stdout.strip().splitlines() or [result.stderr[-300:]])[-1]
            print(f"{os.path.basename(checkpoint)} {line} [{time.time() - started:.0f}s]", flush=True)
            report.write(f"{os.path.basename(checkpoint)} {line}\n")
            report.flush()
print("ALL DONE", flush=True)
