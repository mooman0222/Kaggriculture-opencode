"""Kaggle GPU: E081 stage 2 — recovery data on the E081 clone's own mistakes, then retrain from scratch (2026-09-25).
v41 で効いた手順 (回復データ + 時刻の埋め込み + 最初から学習、-61k -> -14.5k) を教師 E081 に当てる。
第 1 段階 (kernel raw-bc-e081) の ep3: 対 v41 -21.7k / 対 E072 -55.8k (教師 E081 は +4.9k / -385)。
予測: 対 v41 -5k〜-15k、対 E072 -20k〜-40k。対 v41 が -21.7k より良くならなければ外れ。
手順: gen_recover 4 プロセス × 250 局 (student = raw_e081_ep3、影の教師 E081、相手は第 1 段階と同じ 8 枠の混成)
-> 元 1,000 + 回復 1,000 局で最初から 4 epoch (bs 512、lr 7e-4) -> ep1 / ep3 を対 E072・対 v41 各 32 戦。
乾式実行: KAGGLE_ROOT=<偽ツリー> KAGGLE_DRY=1 .venv/bin/python rl/kaggle30/run_recover_e081.py
"""
import glob, os, shutil, subprocess, sys, time
import numpy as np
ROOT = os.environ.get("KAGGLE_ROOT", "/kaggle"); INPUT = os.path.join(ROOT, "input"); output = os.path.join(ROOT, "working"); DRY = os.environ.get("KAGGLE_DRY")
PROCS, PER = (2, 2) if DRY else (4, 250)


def first(suffix, prefer):
    m = sorted(glob.glob(os.path.join(INPUT, "**", suffix), recursive=True)); m = [x for x in m if prefer in x] or m
    assert m, suffix; return m[0]


rl = os.path.dirname(first("gen_recover.py", "panel-agents")); kagsim_src = os.path.dirname(os.path.dirname(first("sim/sim.hpp", "v41self")))
agents = os.path.dirname(os.path.dirname(first("e081/main.py", "panel-agents"))); A = lambda n: os.path.join(agents, n, "main.py")
student = first("raw_e081_ep3.pt", "raw-bc-e081")
clean = os.path.dirname([p for p in sorted(glob.glob(os.path.join(INPUT, "**", "200000_s0.npz"), recursive=True)) if "uop" in np.load(p).files][0])
panel = [A(n) for n in ("e081", "e081", "e072", "e074", "pub_herd2700", "v46", "v48", "v41")]
print("student:", student, "\nclean:", clean, len(glob.glob(os.path.join(clean, "*.npz"))), flush=True)
os.makedirs(output, exist_ok=True); sim = os.path.join(output, "kagsim_src"); shutil.copytree(kagsim_src, sim, dirs_exist_ok=True)
r = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=sim, capture_output=True, text=True); assert r.returncode == 0, r.stderr[-300:]
env = dict(os.environ, PYTHONPATH=sim + ":" + rl); recover = os.path.join(output, "recover"); t0 = time.time()
procs = [subprocess.Popen([sys.executable, os.path.join(rl, "gen_recover.py"), "--student", student, "--agent", A("e081"), "--opp", *panel,
                           "--seed0", str(230000 + p * PER), "--games", str(PER), "--out", recover], cwd=rl, env=dict(env, OMP_NUM_THREADS="1"),
                          stdout=open(os.path.join(output, f"gen_{p}.log"), "w"), stderr=subprocess.STDOUT) for p in range(PROCS)]
print("gen rc", [p.wait() for p in procs], "recovery games", len(glob.glob(os.path.join(recover, "*.npz"))), f"[{time.time() - t0:.0f}s]", flush=True)
out = os.path.join(output, "raw_e081_rec.pt"); t0 = time.time()
cg = os.path.join(clean, "*.npz") if not DRY else os.path.join(clean, "20000[0-3]_s*.npz")
r = subprocess.run([sys.executable, os.path.join(rl, "train_raw.py"), "--data", cg, os.path.join(recover, "*.npz"), "--out", out,
                    "--epochs", "1" if DRY else "4", "--bs", "512", "--lr", "7e-4"], cwd=rl, env=env)
print(f"train rc {r.returncode} [{time.time() - t0:.0f}s]", flush=True); assert r.returncode == 0
with open(os.path.join(output, "eval.txt"), "a") as rep:
    for k in ((0,) if DRY else (1, 3)):
        for opp in ("e072", "v41"):
            t0 = time.time(); r = subprocess.run([sys.executable, os.path.join(rl, "raw.py"), out.replace(".pt", f"_ep{k}.pt"), "--games", "2" if DRY else "32", "--vs", A(opp)], cwd=rl, env=env, capture_output=True, text=True)
            line = (r.stdout.strip().splitlines() or [r.stderr[-300:]])[-1]; print(f"rec ep{k} vs {opp}: {line} [{time.time() - t0:.0f}s]", flush=True); rep.write(f"rec ep{k} vs {opp}: {line}\n"); rep.flush()
shutil.rmtree(sim, ignore_errors=True); print("ALL DONE", flush=True)
