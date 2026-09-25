"""Kaggle GPU: BC パイプラインは既知の方策を複製できるか (2026-09-24)。

教師 = v41 (評価相手そのもの) の自己対戦 1,510 局 (rl/gen_selfplay.py、1 局 1 席)。v41 のクローン対 v41 は
完全な複製なら margin 0 なので、測った margin がそのまま「表現 + 学習 + 閉ループの複合誤差」の損失になる
(Majkel 版の −15k は教師自身の対 v41 の強さが分からず、損失を切り分けられなかった)。
レシピは bc20/bc22 (初期値なし・4 epoch・lr 5e-4・bs 128) と同一。データ量は Majkel と同じ 854 局 (train 769 / val 85 も同数)。
ep1 と ep3 を対 v41 64 戦 (seed0 5000、bc15/bc22 と同じ席・seed) で評価する。

乾式実行: KAGGLE_ROOT=<偽ツリー> KAGGLE_DRY=1 .venv/bin/python rl/kaggle24/run_clone.py
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
ARMS = [("clone854", "8" if DRY else "854")]   # 短縮版 (09-24 16:10): 判断を早めるため 1 アーム・評価は ep1/ep3 のみ


def first(suffix):
    matches = sorted(glob.glob(os.path.join(INPUT, "**", suffix), recursive=True))
    matches = [m for m in matches if "v41self" in m] or matches   # 同梱の現行コードを優先 (bc7 dataset には 09-16 の古い複製がある)
    assert matches, f"input not found: {suffix}"
    return matches[0]


rl = os.path.dirname(first("act_common.py"))
kagsim_src = os.path.dirname(os.path.dirname(first("sim/sim.hpp")))
v41 = first("v41_main.py")
shards = os.path.dirname(first("200000_s0.npz"))
print("rl:", rl, "shards:", len(glob.glob(os.path.join(shards, "*.npz"))), flush=True)

os.makedirs(output, exist_ok=True)
simulator = os.path.join(output, "kagsim_src")
shutil.copytree(kagsim_src, simulator, dirs_exist_ok=True)
result = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=simulator, capture_output=True, text=True)
print("kagsim build rc", result.returncode, result.stderr[-300:], flush=True)
assert result.returncode == 0
environment = dict(os.environ, PYTHONPATH=simulator + ":" + rl)

for name, max_games in ARMS:
    started = time.time()
    result = subprocess.run([sys.executable, os.path.join(rl, "train_bc3.py"), "--data", os.path.join(shards, "*.npz"),
                             "--out", os.path.join(output, f"{name}.pt"), "--epochs", "1" if DRY else "4", "--bs", "128",
                             "--lr", "5e-4", "--max-games", max_games], cwd=rl)
    print(f"{name} rc {result.returncode} [{time.time() - started:.0f}s]", flush=True)
    assert result.returncode == 0
    with open(os.path.join(output, "eval.txt"), "a") as report:
        for checkpoint in [os.path.join(output, f"{name}_ep{k}.pt") for k in ((0,) if DRY else (1, 3))]:
            started = time.time()
            result = subprocess.run([sys.executable, os.path.join(rl, "play3.py"), checkpoint, "--games", GAMES, "--vs", v41],
                                    cwd=rl, env=environment, capture_output=True, text=True)
            line = (result.stdout.strip().splitlines() or [result.stderr[-300:]])[-1]
            print(f"{os.path.basename(checkpoint)} {line} [{time.time() - started:.0f}s]", flush=True)
            report.write(f"{os.path.basename(checkpoint)} {line}\n"); report.flush()
print("ALL DONE", flush=True)
