"""Kaggle GPU: raw BC with a discrete clock (day/hour embeddings) on the existing 2,000 v41 shards (2026-09-25).
仮説: 複製が毎局同じ手番で外すのは「日・時刻・ルート」で決まる教師の予定表 (肥料は 12/13/15/21... 日目の 0 時に売る、
家畜は 9 日目 1 時に買う) を連続値の day/29・hour/23 から鋭く作れないため。
予測: 最初の分岐の中央値が 320 手より後ろ、検証 mstep > 92.5%。分岐が 145 / 218 / 320 のままなら仮説は外れ。
データ: raw-bc-v41 の出力 (元 1,000 局) + raw-dagger1-v41 の出力 recover/ (回復 1,000 局)。初期値なし・4 epoch・bs 512・lr 7e-4。
乾式実行: KAGGLE_ROOT=<偽ツリー> KAGGLE_DRY=1 .venv/bin/python rl/kaggle29/run_clock.py
"""
import glob, os, shutil, subprocess, sys, time
import numpy as np
ROOT = os.environ.get("KAGGLE_ROOT", "/kaggle"); INPUT = os.path.join(ROOT, "input"); output = os.path.join(ROOT, "working"); DRY = os.environ.get("KAGGLE_DRY")


def first(suffix, prefer):
    m = sorted(glob.glob(os.path.join(INPUT, "**", suffix), recursive=True)); m = [x for x in m if prefer in x] or m
    assert m, suffix; return m[0]


rl = os.path.dirname(first("raw.py", "panel-agents")); kagsim_src = os.path.dirname(os.path.dirname(first("sim/sim.hpp", "v41self")))
v41 = first("v41/main.py", "panel-agents")
raw_dirs = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(INPUT, "**", "*_s*.npz"), recursive=True) if "uop" in np.load(p).files})
print("shard dirs:", raw_dirs, [len(glob.glob(os.path.join(d, "*.npz"))) for d in raw_dirs], flush=True)
os.makedirs(output, exist_ok=True); sim = os.path.join(output, "kagsim_src"); shutil.copytree(kagsim_src, sim, dirs_exist_ok=True)
r = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=sim, capture_output=True, text=True); assert r.returncode == 0, r.stderr[-300:]
env = dict(os.environ, PYTHONPATH=sim + ":" + rl); out = os.path.join(output, "raw_v41_clock.pt"); t0 = time.time()
r = subprocess.run([sys.executable, os.path.join(rl, "train_raw.py"), "--data", *[os.path.join(d, "*.npz") for d in raw_dirs], "--out", out,
                    "--epochs", "1" if DRY else "4", "--bs", "512", "--lr", "7e-4", "--max-games", "8" if DRY else "100000"], cwd=rl, env=env)
print(f"train rc {r.returncode} [{time.time() - t0:.0f}s]", flush=True); assert r.returncode == 0
with open(os.path.join(output, "eval.txt"), "a") as rep:
    for k in ((0,) if DRY else (1, 3)):
        t0 = time.time(); r = subprocess.run([sys.executable, os.path.join(rl, "raw.py"), out.replace(".pt", f"_ep{k}.pt"), "--games", "2" if DRY else "64", "--vs", v41], cwd=rl, env=env, capture_output=True, text=True)
        line = (r.stdout.strip().splitlines() or [r.stderr[-300:]])[-1]; print(f"clock ep{k} vs v41: {line} [{time.time() - t0:.0f}s]", flush=True); rep.write(f"clock ep{k}: {line}\n")
shutil.rmtree(sim, ignore_errors=True); print("ALL DONE", flush=True)
