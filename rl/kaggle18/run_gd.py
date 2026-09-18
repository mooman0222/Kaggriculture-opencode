"""Kaggle GPU: 目的地を貪欲に固定した PPO (--greedy-dest) が、素の SP3 レシピと違って崩壊せず伸びるか。

4d の測定 (experiments.md) で分かったこと:
  得点 ≈ 63,072 × (1−ε)^10。うち目的地ノイズが損失の 85% (単独で (1−ε)^7.6)、作業ノイズは ε を倍にしても
  −15.2k → −16.6k と飽和する。つまり per-step で目的地をサンプリングする探索は代償が大きすぎる。
  温度を下げても戻らない (temp 0.05〜0.7 で own 45〜56k、非単調 = 差はノイズ)。

そこで探索を作業・数量・市場ヘッドだけに限る。目的地は貪欲、log-prob 0 で PPO から自動的に外れる。
エントロピー賞与も op ヘッドへ移す (目的地を固定したまま dest のエントロピーを上げても鎖を壊す方向に押すだけ)。

2 本を順に走らせて比べる:
  base : SP3 と同じレシピ (対照)
  gd   : --greedy-dest

再開: 出力を dataset にして dataset_sources に足すと <out>.state を見つけて --resume する (run_sp4 と同じ形)。
乾式実行: KAGGLE_ROOT=<偽ツリー> KAGGLE_DRY=1 .venv/bin/python rl/kaggle18/run_gd.py
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

ITERS = "3" if DRY else "60"
MINUTES = "3" if DRY else "150"        # 1 本あたり。カーネルは完了時にしか出力を保存しない
WORKERS, GAMES = ("2", "4") if DRY else ("4", "120")   # 4x120 = 480 局 = Mac の 10x48 と同じ標本数 (70k/iter)
EVAL_GAMES = "4" if DRY else "32"
ARMS = [("base", []), ("gd", ["--greedy-dest"])]


def first(suffix):
    matches = sorted(glob.glob(os.path.join(INPUT, "**", suffix), recursive=True))
    if not matches:
        for root, dirs, files in os.walk(INPUT):
            if root.count("/") <= INPUT.count("/") + 3:
                print("input:", root, dirs[:8], files[:8], flush=True)
    assert matches, f"input not found: {suffix}"
    return matches[0]


def patched_rl_dir():
    """--greedy-dest を持つ train.py がある rl/ を選ぶ (bc7 dataset に古い複製が同居しているため)。"""
    for path in sorted(glob.glob(os.path.join(INPUT, "**", "sp", "train.py"), recursive=True)):
        if "greedy_dest" in open(path).read():
            return os.path.dirname(os.path.dirname(path))
    raise AssertionError("patched sp/train.py (--greedy-dest) not found in any input dataset")


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

for name, extra in ARMS:
    out = os.path.join(output, f"ppo_{name}.pt")
    resume = []
    prior = glob.glob(os.path.join(INPUT, "**", f"ppo_{name}.pt.state"), recursive=True)
    if prior:
        shutil.copy(prior[0], out + ".state")
        for f in glob.glob(os.path.join(os.path.dirname(prior[0]), f"ppo_{name}*.pt")):
            shutil.copy(f, output)
        resume = ["--resume"]
        print(f"arm {name}: resuming from {prior[0]}", flush=True)
    started = time.time()
    result = subprocess.run([sys.executable, os.path.join(rl, "sp", "train.py"), "--out", out, *common, *extra, *resume],
                            cwd=rl, env=environment)
    print(f"arm {name}: rc {result.returncode} [{time.time() - started:.0f}s]", flush=True)
    assert result.returncode == 0

# 対 v41 の接地した物差し (貪欲評価)。train.py は 25 iter ごとに _itK.pt を残す
with open(os.path.join(output, "eval.txt"), "a") as report:
    targets = []
    for name, _ in ARMS:
        targets += sorted(glob.glob(os.path.join(output, f"ppo_{name}_it*.pt"))) + [os.path.join(output, f"ppo_{name}.pt")]
    for checkpoint in targets + [init]:
        result = subprocess.run([sys.executable, os.path.join(rl, "play2.py"), checkpoint, "--games", EVAL_GAMES, "--vs", v41],
                                cwd=rl, env=environment, capture_output=True, text=True)
        line = (result.stdout.strip().splitlines() or [result.stderr[-300:]])[-1]
        label = "bc5_ep3 (baseline)" if checkpoint == init else os.path.basename(checkpoint)
        print(label, line, flush=True)
        report.write(f"{label} {line}\n")
print("ALL DONE", flush=True)
