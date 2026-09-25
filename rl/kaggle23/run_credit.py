"""Kaggle GPU: PPO の崩壊は信用配分の地平線が原因か (2026-09-24)。

これまでの 5 走 (SP3 レシピ: gamma 0.997, lam 0.95, T 96) は critic がほぼ定数だったため、
advantage が見ていたのは直後 ~1/(1-gamma*lam) = 19 手の現金の出入りだけだった。
分岐実験 (bc15、192 分岐) では、この advantage と「その注文を消したときの最終 margin の差」の相関が
種 +0.00 / 家畜 +0.06 / 小麦 -0.00 (gamma のみなら +0.89〜0.94)。買いは一貫して負、売りは正に採点されていた。

2 アームを別カーネルで同条件に回す (ARM を下で切替、tmp/ に生成した 2 ディレクトリから push):
  mc     : --gamma 1.0 --lam 1.0   全局の実リターン (定数 critic でも最終 margin に接地)
  ctl720 : --gamma 0.997 --lam 0.95  同じ 720 手窓の対照 (窓の変更だけでは直らないことの確認)
共通: SP3 レシピ (bc5_ep3 初期値, tape 0.25, temp 0.5, lr 1e-5, KL 0.005, 昇格 0.53)。
T 720 = 1 iter が全局 1 局 (VecEnv の局は同期しているので窓が局と一致し、局の途中で役割が替わる混入も起きない)。
optimizer step 数を過去の走と揃える: minibatches 8 × 約 23 iter ≈ 184 step、critic 暖機 60 step。
6 iter ごとに ckpt を残し、play2 貪欲で対 v41 32 戦 (基準 bc5_ep3 own 63,072 / margin -71,469、過去 5 走の final 12〜33k)。

乾式実行: KAGGLE_ROOT=<偽ツリー> KAGGLE_DRY=1 .venv/bin/python rl/kaggle23/run_credit.py
"""
import glob
import os
import shutil
import subprocess
import sys
import time

ARM = "mc"
ARGS = {"mc": ["--gamma", "1.0", "--lam", "1.0"], "ctl720": ["--gamma", "0.997", "--lam", "0.95"]}[ARM]

ROOT = os.environ.get("KAGGLE_ROOT", "/kaggle")
INPUT = os.path.join(ROOT, "input")
output = os.path.join(ROOT, "working")
DRY = bool(os.environ.get("KAGGLE_DRY"))

T = "24" if DRY else "720"
SEGMENT = 1 if DRY else 6          # iters between saved checkpoints
BUDGET_MIN = 2 if DRY else 150     # training wall clock; the kernel only saves output on completion
WORKERS, GAMES = ("2", "2") if DRY else ("4", "48")
EVAL_GAMES = "2" if DRY else "32"


def first(suffix):
    matches = sorted(glob.glob(os.path.join(INPUT, "**", suffix), recursive=True))
    assert matches, f"input not found: {suffix}"
    return matches[0]


def patched_rl_dir():
    for path in sorted(glob.glob(os.path.join(INPUT, "**", "sp", "train.py"), recursive=True)):
        if "--vlr" in open(path).read():
            return os.path.dirname(os.path.dirname(path))
    raise AssertionError("sp/train.py with --vlr not found")


rl = patched_rl_dir()
kagsim_src = os.path.dirname(first("kagsim/setup.py"))
v41 = first("v41_main.py")
init = first("ckpt/bc5_ep3.pt")
tapes = first("tapes_top.pkl")
print("arm", ARM, "\nrl:", rl, "\nkagsim:", kagsim_src, "\ninit:", init, "\ntapes:", tapes, flush=True)

os.makedirs(output, exist_ok=True)
simulator = os.path.join(output, "kagsim_src")
shutil.copytree(kagsim_src, simulator, dirs_exist_ok=True)
result = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=simulator, capture_output=True, text=True)
print("kagsim build rc", result.returncode, result.stderr[-300:], flush=True)
assert result.returncode == 0
environment = dict(os.environ, PYTHONPATH=simulator + ":" + rl + ":" + os.path.join(rl, "sp"))
device = "cpu" if DRY else "cuda"

out = os.path.join(output, f"credit_{ARM}.pt")
common = ["--init", init, "--out", out, "--tapes", tapes, "--tape-frac", "0.25", "--workers", WORKERS, "--games", GAMES,
          "--T", T, "--minibatches", "8", "--critic-warmup", "60", "--dev", device, *ARGS]
started = time.time(); iters = 0; snapshots = []
while time.time() - started < BUDGET_MIN * 60:
    iters += SEGMENT
    left = max(1.0, BUDGET_MIN - (time.time() - started) / 60)
    result = subprocess.run([sys.executable, os.path.join(rl, "sp", "train.py"), *common, "--iters", str(iters), "--max-minutes", f"{left:.1f}",
                             *(["--resume"] if iters > SEGMENT else [])], cwd=rl, env=environment)
    assert result.returncode == 0, result.returncode
    snap = out.replace(".pt", f"_seg{iters}.pt"); shutil.copy(out, snap); snapshots.append(snap)
    print(f"segment -> iter {iters} [{time.time() - started:.0f}s]", flush=True)
    if DRY and iters >= 2: break

with open(os.path.join(output, "eval.txt"), "a") as report:
    for checkpoint in snapshots + [init]:
        result = subprocess.run([sys.executable, os.path.join(rl, "play2.py"), checkpoint, "--games", EVAL_GAMES, "--vs", v41],
                                cwd=rl, env=environment, capture_output=True, text=True)
        line = (result.stdout.strip().splitlines() or [result.stderr[-300:]])[-1]
        print(os.path.basename(checkpoint), line, flush=True)
        report.write(f"{ARM} {os.path.basename(checkpoint)} {line}\n")
print("ALL DONE", flush=True)
