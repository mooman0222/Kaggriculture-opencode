"""Kaggle GPU: value head を事前学習してから PPO (09-19 診断への最後の処方)。

診断と `--vlr` の結果:
  value head は乱数初期化のまま定数 (R²≈0) で、adv が生 outcome になる。`--vlr 1e-3` で相関は
  0.13→0.55 まで改善したがスケールが 10 分の 1 のまま (R²_mc 0.05) で崩壊は残った (final own 33.2k)。
  probe (同じ凍結特徴・教師あり) なら数分で R²_val 0.6 に届くので、PPO 開始時点で critic を
  まともな状態にしておくのが最後の切り分け。

このカーネル:
  1. `rl/vpretrain.py` — bc5_ep3 の方策で 8 窓 (T=96) 収集 (self-play + tape)、凍結トランクの
     H['g'] で value head だけを 2000 step 回帰 (lr 1e-3) → `init_v.pt`。トランク凍結なので方策は不変。
  2. `rl/sp/train.py` — SP3 と同一 + `--vlr 1e-3 --vwarmup 20`、45 iter、init は `init_v.pt`。
  3. 対 v41 32 戦 (it25・最終・基準 bc5_ep3)。

判定: それでも崩れる (final own ≪ 63.1k) なら RL を凍結して本線 (chassis) へ。

乾式実行: KAGGLE_ROOT=<偽ツリー> KAGGLE_DRY=1 .venv/bin/python rl/kaggle22/run_vpt.py
結果取得: .venv/bin/python -m kaggle kernels output mmn0222/kaggriculture-ppo-vpretrain -p tmp/kaggle_out_vpt --force
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
MINUTES = "3" if DRY else "150"
WORKERS, GAMES = ("2", "4") if DRY else ("4", "120")
EVAL_GAMES = "4" if DRY else "32"
PT_WINDOWS, PT_STEPS, PT_CAP = ("1", "50", "10000") if DRY else ("8", "2000", "400000")
T = "24" if DRY else "96"


def first(suffix):
    matches = sorted(glob.glob(os.path.join(INPUT, "**", suffix), recursive=True))
    if not matches:
        for root, dirs, files in os.walk(INPUT):
            if root.count("/") <= INPUT.count("/") + 3:
                print("input:", root, dirs[:8], files[:8], flush=True)
    assert matches, f"input not found: {suffix}"
    return matches[0]


def patched_rl_dir():
    """--vlr を持つ train.py がある rl/ を選ぶ (bc7 dataset に古い複製が同居しているため)。"""
    for path in sorted(glob.glob(os.path.join(INPUT, "**", "sp", "train.py"), recursive=True)):
        if "--vlr" in open(path).read():
            return os.path.dirname(os.path.dirname(path))
    raise AssertionError("patched sp/train.py (--vlr) not found in any input dataset")


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

started = time.time()
pretrained = os.path.join(output, "init_v.pt")
result = subprocess.run([sys.executable, os.path.join(rl, "vpretrain.py"), "--init", init, "--out", pretrained,
                         "--tapes", tapes, "--tape-frac", "0.25", "--workers", WORKERS, "--games", GAMES, "--T", T,
                         "--windows", PT_WINDOWS, "--steps", PT_STEPS, "--cap", PT_CAP, "--dev", device],
                        cwd=rl, env=environment)
print(f"vpretrain: rc {result.returncode} [{time.time() - started:.0f}s]", flush=True)
assert result.returncode == 0

out = os.path.join(output, "ppo_vpt.pt")
common = ["--tapes", tapes, "--tape-frac", "0.25", "--workers", WORKERS, "--games", GAMES,
          "--T", T, "--iters", ITERS, "--max-minutes", MINUTES, "--dev", device, "--init", pretrained]
started = time.time()
result = subprocess.run([sys.executable, os.path.join(rl, "sp", "train.py"), "--out", out, "--vlr", "1e-3", "--vwarmup", "20", *common],
                        cwd=rl, env=environment)
print(f"train vpt: rc {result.returncode} [{time.time() - started:.0f}s]", flush=True)
assert result.returncode == 0

with open(os.path.join(output, "eval.txt"), "a") as report:
    targets = sorted(glob.glob(os.path.join(output, "ppo_vpt_it*.pt"))) + [out]
    for checkpoint in targets + [init]:
        result = subprocess.run([sys.executable, os.path.join(rl, "play2.py"), checkpoint, "--games", EVAL_GAMES, "--vs", v41],
                                cwd=rl, env=environment, capture_output=True, text=True)
        line = (result.stdout.strip().splitlines() or [result.stderr[-300:]])[-1]
        label = "bc5_ep3 (baseline)" if checkpoint == init else os.path.basename(checkpoint)
        print(label, line, flush=True)
        report.write(f"{label} {line}\n")
print("ALL DONE", flush=True)
