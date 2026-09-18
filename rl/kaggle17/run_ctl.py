"""Kaggle GPU: 「PPO が崩れるのは規模不足か、信号が壊れているか」の対照実験。

SP3 と同じレシピ (rl/sp/train.py、Policy2、bc5_ep3 初期値、tape-frac 0.25、T 96) を 2 本走らせる:
  real : そのまま
  shuf : --shuffle-adv = advantage の並びだけを無作為に壊す (critic の ret・KL・比率の分布・対戦列は同一)
両者の劣化の傾き (vsTAPE / vsT の window、対 v41 32 戦) を比べる。
  傾きが同じ    -> advantage は何も運んでいない = 信号側の問題。規模の話は成立しない
  real が緩い   -> 信号は乗っている = そこで初めて規模の議論ができる

乾式実行: KAGGLE_ROOT=<偽ツリー> KAGGLE_DRY=1 .venv/bin/python rl/kaggle17/run_ctl.py
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
DRY = bool(os.environ.get("KAGGLE_DRY"))

ITERS = "3" if DRY else "45"
MINUTES = "3" if DRY else "150"      # 1 本あたりの上限。カーネルは完了時にしか出力を保存しない
WORKERS, GAMES = ("2", "4") if DRY else ("4", "120")   # 4x120 = 480 局 = Mac の 10x48 と同じ標本数
EVAL_GAMES = "4" if DRY else "32"


def first(suffix):
    matches = sorted(glob.glob(os.path.join(INPUT, "**", suffix), recursive=True))
    if not matches:
        for root, dirs, files in os.walk(INPUT):
            if root.count("/") <= INPUT.count("/") + 3:
                print("input:", root, dirs[:8], files[:8], flush=True)
    assert matches, f"input not found: {suffix}"
    return matches[0]


def patched_rl_dir():
    """--shuffle-adv を持つ train.py がある rl/ を選ぶ (bc7 dataset に古い複製が同居しているため)。"""
    for path in sorted(glob.glob(os.path.join(INPUT, "**", "sp", "train.py"), recursive=True)):
        if "shuffle_adv" in open(path).read():
            return os.path.dirname(os.path.dirname(path))
    raise AssertionError("patched sp/train.py (--shuffle-adv) not found in any input dataset")


rl = patched_rl_dir()
kagsim_src = os.path.dirname(first("kagsim/setup.py"))
v41 = first("v41_main.py")
init = first("ckpt/bc5_ep3.pt")
tapes = first("tapes_top.pkl")
print("rl:", rl, "\nkagsim:", kagsim_src, "\ninit:", init, "\ntapes:", tapes, "\nv41:", v41, flush=True)

os.makedirs(output, exist_ok=True)
simulator = os.path.join(output, "kagsim_src")
shutil.copytree(kagsim_src, simulator, dirs_exist_ok=True)
result = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=simulator, capture_output=True, text=True)
print("kagsim build rc", result.returncode, result.stderr[-300:], flush=True)
assert result.returncode == 0
environment = dict(os.environ, PYTHONPATH=simulator + ":" + rl + ":" + os.path.join(rl, "sp"))
device = "cpu" if DRY else "cuda"

common = ["--init", init, "--tapes", tapes, "--tape-frac", "0.25", "--workers", WORKERS, "--games", GAMES,
          "--T", "96", "--iters", ITERS, "--max-minutes", MINUTES, "--dev", device]
arms = [("real", []), ("shuf", ["--shuffle-adv"])]

for name, extra in arms:
    out = os.path.join(output, f"ctl_{name}.pt")
    started = time.time()
    result = subprocess.run([sys.executable, os.path.join(rl, "sp", "train.py"), "--out", out, *common, *extra],
                            cwd=rl, env=environment)
    print(f"arm {name}: rc {result.returncode} [{time.time() - started:.0f}s]", flush=True)
    assert result.returncode == 0

# 対 v41 の接地した物差し: 各 arm の保存済み checkpoint (25 iter ごと) と最終
with open(os.path.join(output, "eval.txt"), "a") as report:
    for name, _ in arms:
        out = os.path.join(output, f"ctl_{name}.pt")
        for checkpoint in sorted(glob.glob(os.path.join(output, f"ctl_{name}_it*.pt"))) + [out]:
            result = subprocess.run([sys.executable, os.path.join(rl, "play2.py"), checkpoint, "--games", EVAL_GAMES, "--vs", v41],
                                    cwd=rl, env=environment, capture_output=True, text=True)
            line = (result.stdout.strip().splitlines() or [result.stderr[-300:]])[-1]
            print(os.path.basename(checkpoint), line, flush=True)
            report.write(f"{os.path.basename(checkpoint)} {line}\n")
    # 基準点 (未学習の初期値)
    result = subprocess.run([sys.executable, os.path.join(rl, "play2.py"), init, "--games", EVAL_GAMES, "--vs", v41],
                            cwd=rl, env=environment, capture_output=True, text=True)
    line = (result.stdout.strip().splitlines() or [result.stderr[-300:]])[-1]
    print("bc5_ep3 (baseline)", line, flush=True)
    report.write(f"bc5_ep3 (baseline) {line}\n")
print("ALL DONE", flush=True)
