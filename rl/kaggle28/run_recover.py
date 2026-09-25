"""Kaggle GPU: DAgger iteration 1 for the raw v41 clone — recovery data on the student's own mistakes (2026-09-24).

学習前の予測 (手元の測定): いまの複製 ep3 は対 v41 −61k。最初の分岐後を教師に任せると −0.8k (楽観側)、
外れるたびに教師が 24 手だけ回復して ep3 に戻すと平均 −23.6k (悲観側)。1 回の反復で −5k〜−15k を見込み、−20k より悪ければ失敗。
手順: kagsim をビルド → rl/gen_recover.py 4 プロセス × 250 局 (student = raw2-bc-v41 の ep3、影の教師 v41、相手 v41)
→ 元の 1,000 局 (raw-bc-v41 の出力) + 回復 1,000 局で ep3 から 3 epoch 追加学習 (lr 3e-4、bs 512) → ep0 / ep2 を対 v41 64 戦。
乾式実行: KAGGLE_ROOT=<偽ツリー> KAGGLE_DRY=1 .venv/bin/python rl/kaggle28/run_recover.py
"""
import glob
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.environ.get("KAGGLE_ROOT", "/kaggle")
INPUT = os.path.join(ROOT, "input")
output = os.path.join(ROOT, "working")
DRY = os.environ.get("KAGGLE_DRY")
PROCS, PER = (2, 2) if DRY else (4, 250)


def first(suffix, prefer):
    matches = sorted(glob.glob(os.path.join(INPUT, "**", suffix), recursive=True))
    matches = [m for m in matches if prefer in m] or matches
    assert matches, f"input not found: {suffix}"
    return matches[0]


rl = os.path.dirname(first("gen_recover.py", "panel-agents"))
kagsim_src = os.path.dirname(os.path.dirname(first("sim/sim.hpp", "v41self")))
v41 = first("v41/main.py", "panel-agents")
student = first("raw_v41_ep3.pt", "raw2-bc-v41")
clean = os.path.dirname([p for p in sorted(glob.glob(os.path.join(INPUT, "**", "200000_s0.npz"), recursive=True)) if "uop" in np.load(p).files][0])
print("rl:", rl, "\nstudent:", student, "\nclean shards:", clean, len(glob.glob(os.path.join(clean, "*.npz"))), flush=True)

os.makedirs(output, exist_ok=True)
simulator = os.path.join(output, "kagsim_src")
shutil.copytree(kagsim_src, simulator, dirs_exist_ok=True)
result = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=simulator, capture_output=True, text=True)
print("kagsim build rc", result.returncode, result.stderr[-300:], flush=True)
assert result.returncode == 0
environment = dict(os.environ, PYTHONPATH=simulator + ":" + rl)

recover = os.path.join(output, "recover"); started = time.time()
procs = [subprocess.Popen([sys.executable, os.path.join(rl, "gen_recover.py"), "--student", student, "--agent", v41,
                           "--seed0", str(220000 + p * PER), "--games", str(PER), "--out", recover], cwd=rl,
                          env=dict(environment, OMP_NUM_THREADS="1"), stdout=open(os.path.join(output, f"gen_{p}.log"), "w"), stderr=subprocess.STDOUT)
         for p in range(PROCS)]
print("gen rc", [p.wait() for p in procs], "recovery games", len(glob.glob(os.path.join(recover, "*.npz"))), f"[{time.time() - started:.0f}s]", flush=True)

started = time.time(); out = os.path.join(output, "raw_v41_dagger1.pt")
clean_glob = os.path.join(clean, "*.npz") if not DRY else os.path.join(clean, "20000[0-3]_s*.npz")
result = subprocess.run([sys.executable, os.path.join(rl, "train_raw.py"), "--data", clean_glob, os.path.join(recover, "*.npz"), "--out", out,
                         "--init", student, "--epochs", "1" if DRY else "3", "--bs", "512", "--lr", "3e-4"], cwd=rl, env=environment)
print(f"train rc {result.returncode} [{time.time() - started:.0f}s]", flush=True)
assert result.returncode == 0

with open(os.path.join(output, "eval.txt"), "a") as report:
    for k in ((0,) if DRY else (0, 2)):
        started = time.time()
        result = subprocess.run([sys.executable, os.path.join(rl, "raw.py"), out.replace(".pt", f"_ep{k}.pt"), "--games", "2" if DRY else "64", "--vs", v41],
                                cwd=rl, env=environment, capture_output=True, text=True)
        line = (result.stdout.strip().splitlines() or [result.stderr[-300:]])[-1]
        print(f"dagger1 ep{k} vs v41: {line} [{time.time() - started:.0f}s]", flush=True)
        report.write(f"dagger1 ep{k} vs v41: {line}\n"); report.flush()
shutil.rmtree(simulator, ignore_errors=True)
print("ALL DONE", flush=True)
