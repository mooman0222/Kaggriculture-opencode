"""Kaggle GPU: BC on the raw action representation (rl/raw.py), data generated in-kernel (2026-09-24).

rl/raw.py の表現は記録 10 局 (v41 4 局・Majkel 実戦 6 局) を kagsim で完全再現する (旧 Policy3 表現は v41 を 125k → 30k に落とした)。
ARM (tmp/ に生成した 2 ディレクトリから push):
  v41  : 教師 v41 の自己対戦 (旧表現の複製実験 rl/kaggle24 の再走。旧: val 99.8% で対 v41 −130k)
  e079 : 教師 = 我々の提出系 E079、相手は実戦に寄せた混成 (鏡像 E079×2・E072・E074・herd2700 / 旧系 v46・v48・v41)
手順: kagsim をビルド → 4 プロセスで教師の席の shard を生成 → train_raw (1,000 局、初期値なし・4 epoch・lr 5e-4・bs 256)
→ ep1 / ep3 を評価 (v41 アームは対 v41 64 戦、e079 アームは対 E072・対 v41 各 32 戦。教師自身は対 E072 +392・対 v41 +6,397)。
乾式実行: KAGGLE_ROOT=<偽ツリー> KAGGLE_DRY=1 .venv/bin/python rl/kaggle27/run_raw.py
"""
import glob
import os
import shutil
import subprocess
import sys
import time

ARM = "v41"
ROOT = os.environ.get("KAGGLE_ROOT", "/kaggle")
INPUT = os.path.join(ROOT, "input")
output = os.path.join(ROOT, "working")
DRY = os.environ.get("KAGGLE_DRY")
PROCS, PER = (2, 3) if DRY else (4, 250)
TRAIN_GAMES = "5" if DRY else "1000"


def first(suffix, prefer):
    matches = sorted(glob.glob(os.path.join(INPUT, "**", suffix), recursive=True))
    matches = [m for m in matches if prefer in m] or matches
    assert matches, f"input not found: {suffix}"
    return matches[0]


rl = os.path.dirname(first("raw.py", "panel-agents"))
kagsim_src = os.path.dirname(os.path.dirname(first("sim/sim.hpp", "v41self")))
agents = os.path.dirname(os.path.dirname(first("e079/main.py", "panel-agents")))
A = lambda name: os.path.join(agents, name, "main.py")
teacher, panel, evals = {"v41": (A("v41"), [A("v41")], [("v41", "2" if DRY else "64")]),
                         "e079": (A("e079"), [A(n) for n in ("e079", "e079", "e072", "e074", "pub_herd2700", "v46", "v48", "v41")],
                                  [("e072", "2" if DRY else "32"), ("v41", "2" if DRY else "32")]),
                         "e081": (A("e081"), [A(n) for n in ("e081", "e081", "e072", "e074", "pub_herd2700", "v46", "v48", "v41")],
                                  [("e072", "2" if DRY else "32"), ("v41", "2" if DRY else "32")])}[ARM]
print("arm", ARM, "rl:", rl, flush=True)

os.makedirs(output, exist_ok=True)
simulator = os.path.join(output, "kagsim_src")
shutil.copytree(kagsim_src, simulator, dirs_exist_ok=True)
result = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=simulator, capture_output=True, text=True)
print("kagsim build rc", result.returncode, result.stderr[-300:], flush=True)
assert result.returncode == 0
environment = dict(os.environ, PYTHONPATH=simulator + ":" + rl)

import numpy as np   # 前回カーネルの出力を kernel_sources で渡せば生成を省く。v41self dataset にも同名の旧形式 shard があるので中身 (uop) で選ぶ
reuse = [p for p in sorted(glob.glob(os.path.join(INPUT, "**", "200000_s0.npz"), recursive=True)) if "uop" in np.load(p).files]
print("reuse shards from:", reuse[:1], flush=True)
shards = os.path.dirname(reuse[0]) if reuse else os.path.join(output, "shards"); started = time.time()
procs = [] if reuse else [subprocess.Popen([sys.executable, os.path.join(rl, "gen_selfplay.py"), "--raw", "--agent", teacher, "--opp", *panel,
                           "--seed0", str(200000 + p * PER), "--games", str(PER), "--out", shards], cwd=rl, env=environment,
                          stdout=open(os.path.join(output, f"gen_{p}.log"), "w"), stderr=subprocess.STDOUT) for p in range(PROCS)]
print("gen rc", [p.wait() for p in procs], "shards", len(glob.glob(os.path.join(shards, "*.npz"))), f"[{time.time() - started:.0f}s]", flush=True)

started = time.time(); out = os.path.join(output, f"raw_{ARM}.pt")
result = subprocess.run([sys.executable, os.path.join(rl, "train_raw.py"), "--data", os.path.join(shards, "*.npz"), "--out", out,
                         "--epochs", "1" if DRY else "4", "--bs", "512", "--lr", "7e-4", "--max-games", TRAIN_GAMES], cwd=rl, env=environment)
print(f"train rc {result.returncode} [{time.time() - started:.0f}s]", flush=True)
assert result.returncode == 0

with open(os.path.join(output, "eval.txt"), "a") as report:
    for k in ((0,) if DRY else (1, 3)):
        for opp, games in evals:
            started = time.time()
            result = subprocess.run([sys.executable, os.path.join(rl, "raw.py"), out.replace(".pt", f"_ep{k}.pt"), "--games", games, "--vs", A(opp)],
                                    cwd=rl, env=environment, capture_output=True, text=True)
            line = (result.stdout.strip().splitlines() or [result.stderr[-300:]])[-1]
            print(f"{ARM} ep{k} vs {opp}: {line} [{time.time() - started:.0f}s]", flush=True)
            report.write(f"{ARM} ep{k} vs {opp}: {line}\n"); report.flush()
shutil.rmtree(simulator, ignore_errors=True)
print("ALL DONE", flush=True)
