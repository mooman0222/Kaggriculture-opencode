"""Kaggle GPU: E081 stage 3 — data from weedy farms (teacher skips watering one day), 8 epochs (2026-09-25).
第 2 段階の対 v41 の悪化は 12 局中 1 局の全面崩壊 (-170k): 10 日目から作物 29 -> 0・雑草 27・家畜 10 -> 1 で戻らない。
教師 E081 の畑にはほぼ雑草が出ないので、雑草だらけの畑は学習データに無い。教師は 1 日水やりを抜かれても
雑草 13 マスを 3 日で片付け植え直す (手元 4 局) ので、その手順をデータに入れる。
予測: 全面崩壊の局が消え、対 v41 は -17k 前後 (±3k 前後)、対 E072 は -21k 以上を保つ。
外れ: 12 局の再生で「作物 0・雑草 20 以上」の局が残る。
手順: gen_selfplay --raw --dry 4,20 (1 局につき 1 日、教師の WATER を PASS で実行、ラベルは WATER) 4 × 250 局
-> 元 1,000 + 回復 1,000 + 雑草 1,000 局で最初から 8 epoch (bs 512、lr 7e-4) -> ep3 / ep5 / ep7 を対 E072・対 v41 各 32 戦。
乾式実行: KAGGLE_ROOT=<偽ツリー> KAGGLE_DRY=1 .venv/bin/python rl/kaggle31/run_dry_e081.py
"""
import glob, os, shutil, subprocess, sys, time
import numpy as np
ROOT = os.environ.get("KAGGLE_ROOT", "/kaggle"); INPUT = os.path.join(ROOT, "input"); output = os.path.join(ROOT, "working"); DRY = os.environ.get("KAGGLE_DRY")
PROCS, PER = (2, 2) if DRY else (4, 250)


def first(suffix, prefer):
    m = sorted(glob.glob(os.path.join(INPUT, "**", suffix), recursive=True)); m = [x for x in m if prefer in x] or m
    assert m, suffix; return m[0]


rl = os.path.dirname(first("gen_selfplay.py", "panel-agents")); kagsim_src = os.path.dirname(os.path.dirname(first("sim/sim.hpp", "v41self")))
agents = os.path.dirname(os.path.dirname(first("e081/main.py", "panel-agents"))); A = lambda n: os.path.join(agents, n, "main.py")
panel = [A(n) for n in ("e081", "e081", "e072", "e074", "pub_herd2700", "v46", "v48", "v41")]
old = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(INPUT, "**", "*_s*.npz"), recursive=True) if "v41self" not in p and "uop" in np.load(p).files})
print("existing shard dirs:", old, [len(glob.glob(os.path.join(d, "*.npz"))) for d in old], flush=True)
os.makedirs(output, exist_ok=True); sim = os.path.join(output, "kagsim_src"); shutil.copytree(kagsim_src, sim, dirs_exist_ok=True)
r = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=sim, capture_output=True, text=True); assert r.returncode == 0, r.stderr[-300:]
env = dict(os.environ, PYTHONPATH=sim + ":" + rl); dry = os.path.join(output, "dry"); t0 = time.time()
procs = [subprocess.Popen([sys.executable, os.path.join(rl, "gen_selfplay.py"), "--raw", "--dry", "4,20", "--agent", A("e081"), "--opp", *panel,
                           "--seed0", str(240000 + p * PER), "--games", str(PER), "--out", dry], cwd=rl, env=dict(env, OMP_NUM_THREADS="1"),
                          stdout=open(os.path.join(output, f"gen_{p}.log"), "w"), stderr=subprocess.STDOUT) for p in range(PROCS)]
print("gen rc", [p.wait() for p in procs], "dry games", len(glob.glob(os.path.join(dry, "*.npz"))), f"[{time.time() - t0:.0f}s]", flush=True)
out = os.path.join(output, "raw_e081_dry.pt"); t0 = time.time(); E = 1 if DRY else 8
r = subprocess.run([sys.executable, os.path.join(rl, "train_raw.py"), "--data", *[os.path.join(d, "*.npz") for d in old + [dry]], "--out", out,
                    "--epochs", str(E), "--bs", "512", "--lr", "7e-4", "--max-games", "12" if DRY else "100000"], cwd=rl, env=env)
print(f"train rc {r.returncode} [{time.time() - t0:.0f}s]", flush=True); assert r.returncode == 0
with open(os.path.join(output, "eval.txt"), "a") as rep:
    for k in ((0,) if DRY else (3, 5, 7)):
        for opp in ("e072", "v41"):
            t0 = time.time(); r = subprocess.run([sys.executable, os.path.join(rl, "raw.py"), out.replace(".pt", f"_ep{k}.pt"), "--games", "2" if DRY else "32", "--vs", A(opp)], cwd=rl, env=env, capture_output=True, text=True)
            line = (r.stdout.strip().splitlines() or [r.stderr[-300:]])[-1]; print(f"dry ep{k} vs {opp}: {line} [{time.time() - t0:.0f}s]", flush=True); rep.write(f"dry ep{k} vs {opp}: {line}\n"); rep.flush()
shutil.rmtree(sim, ignore_errors=True); print("ALL DONE", flush=True)
