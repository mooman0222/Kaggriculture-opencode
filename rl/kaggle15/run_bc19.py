"""Kaggle GPU: bc19 = 修正ラベルで初期値から作り直す 2 段。
  1 段目 bc9k19 : 公開データの上位 40 チーム shard (pq) で bc5_ep3 から 4 epoch lr 5e-4  (bc9 レシピ、局数 634 -> 3,500)
  2 段目 bc19   : Majkel 854 局で bc9k19_ep3 から 4 epoch lr 2e-4                        (bc15/bc18 レシピ)
各 checkpoint を対 v41 32 戦で評価して eval.txt に書き出す。
乾式実行: KAGGLE_ROOT=/path/to/fake python run_bc19.py
"""
import glob
import os
import shutil
import subprocess
import sys
import time

ROOT = os.environ.get("KAGGLE_ROOT", "/kaggle")   # 乾式実行で偽の入力ツリーを指すため
INPUT = os.path.join(ROOT, "input")
output = os.path.join(ROOT, "working")
MIXED_GAMES = "2000"   # train_bc3 は全 shard を RAM に載せる (~7 MB/局)。2,000 局 ~ 14 GB。


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
init = first("ckpt/bc5_ep3.pt")
mixed = os.path.dirname(os.path.dirname(first("pq/*/*.npz")))
majkel = os.path.dirname(first("majkel_all/*.npz"))
print("rl:", rl, "init:", init, flush=True)
print("mixed:", mixed, len(glob.glob(os.path.join(mixed, "*", "*.npz"))), "shards", flush=True)
print("majkel:", majkel, len(glob.glob(os.path.join(majkel, "*.npz"))), "shards", flush=True)

os.makedirs(output, exist_ok=True)
simulator = os.path.join(output, "kagsim_src")
shutil.copytree(kagsim_src, simulator, dirs_exist_ok=True)
result = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=simulator, capture_output=True, text=True)
print("kagsim build rc", result.returncode, result.stderr[-300:], flush=True)
have_simulator = result.returncode == 0

DRY = os.environ.get("KAGGLE_DRY")   # 乾式実行: 1 epoch / 少数局 / 評価 2 戦


def train(tag, data, initial, lr, max_games):
    command = [sys.executable, os.path.join(rl, "train_bc3.py"), "--data", data, "--out", os.path.join(output, tag + ".pt"),
               "--init", initial, "--epochs", "1" if DRY else "4", "--bs", "128", "--lr", lr,
               "--max-games", "4" if DRY else max_games]
    started = time.time()
    result = subprocess.run(command, cwd=rl)
    print(f"{tag} rc {result.returncode} [{time.time() - started:.0f}s]", flush=True)
    assert result.returncode == 0, tag
    return os.path.join(output, tag + ("_ep0.pt" if DRY else "_ep3.pt"))


stage1 = train("bc9k19", os.path.join(mixed, "*", "*.npz"), init, "5e-4", MIXED_GAMES)
train("bc19", os.path.join(majkel, "*.npz"), stage1, "2e-4", "100000")

if have_simulator:
    environment = dict(os.environ, PYTHONPATH=simulator + ":" + rl)
    with open(os.path.join(output, "eval.txt"), "w") as report:
        for checkpoint in sorted(glob.glob(os.path.join(output, "bc19_ep*.pt"))) + [stage1]:
            result = subprocess.run([sys.executable, os.path.join(rl, "play3.py"), checkpoint, "--games", "2" if DRY else "32", "--vs", v41],
                                    cwd=rl, env=environment, capture_output=True, text=True)
            line = (result.stdout.strip().splitlines() or [result.stderr[-300:]])[-1]
            print(os.path.basename(checkpoint), line, flush=True)
            report.write(f"{os.path.basename(checkpoint)} {line}\n")
print("ALL DONE", flush=True)
