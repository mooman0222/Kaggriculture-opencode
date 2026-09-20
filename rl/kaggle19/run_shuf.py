"""Kaggle GPU: advantage シャッフル対照の shuf アームだけを再走する。

kaggle17/run_ctl.py は real -> shuf の順に回すが、shuf が it0 でキャンセルされた
(CANCEL_ACKNOWLEDGED)。real は 45 iter 完走済みなのでやり直さず、shuf だけを同一条件
(SP3 レシピ、bc5_ep3 初期値、tape-frac 0.25、T 96、45 iter、150 分上限) で走らせる。

比較は kaggle17 の ctl_real.pt.log との**劣化の傾き** (毎 iter の vsTAPE / vsT window)。
接地の物差しとして shuf の `_itK.pt` と最終を対 v41 32 戦で評価する。

乾式実行: KAGGLE_ROOT=<偽ツリー> KAGGLE_DRY=1 .venv/bin/python rl/kaggle19/run_shuf.py
結果取得: .venv/bin/python -m kaggle kernels output mmn0222/kaggriculture-ppo-adv-shuf -p tmp/kaggle_out_shuf --force
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

out = os.path.join(output, "ctl_shuf.pt")
common = ["--init", init, "--tapes", tapes, "--tape-frac", "0.25", "--workers", WORKERS, "--games", GAMES,
          "--T", "96", "--iters", ITERS, "--max-minutes", MINUTES, "--dev", device]
started = time.time()
result = subprocess.run([sys.executable, os.path.join(rl, "sp", "train.py"), "--out", out, "--shuffle-adv", *common],
                        cwd=rl, env=environment)
print(f"arm shuf: rc {result.returncode} [{time.time() - started:.0f}s]", flush=True)
assert result.returncode == 0

with open(os.path.join(output, "eval.txt"), "a") as report:
    targets = sorted(glob.glob(os.path.join(output, "ctl_shuf_it*.pt"))) + [out]
    for checkpoint in targets + [init]:
        result = subprocess.run([sys.executable, os.path.join(rl, "play2.py"), checkpoint, "--games", EVAL_GAMES, "--vs", v41],
                                cwd=rl, env=environment, capture_output=True, text=True)
        line = (result.stdout.strip().splitlines() or [result.stderr[-300:]])[-1]
        label = "bc5_ep3 (baseline)" if checkpoint == init else os.path.basename(checkpoint)
        print(label, line, flush=True)
        report.write(f"{label} {line}\n")
print("ALL DONE", flush=True)
