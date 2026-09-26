"""Kaggle GPU: E082-teacher market catch-up + reactive-WAIT surpass attempt (2026-09-26).

背景: hybrid試験で損失のほぼ全量が市場ヘッドと確定 (clone unit + 教師 market = 教師水準に回復、5001-5007)。
stage-4はt168を直したが誤差が隣へ移動しseed 5000が悪化 — recover往復は打ち止め。
方針転換: 農場は凍結し市場だけ学ぶ (`train_raw.py --market-only`)。
教師はE082 (E081 + イチゴcap6。農場はt中央値360手まで一致するのでunit転移する。E082 live確認が正式gateだが、
教師としての質は直接対決・席差替で既にE081以上と測定ずみ)。

Phase 1 (追いつき): gen_selfplay --raw --agent E082 1000局 (8混成はE081互換のまま) → --market-only --
init dry_ep7 4 epoch → M1。期待値: 教師水準± (hybridが上限の証明)。
Phase 2 (追い越し・bounded): reactive-WAITラベル。oppが窓頭に大量売りした直後の自売りを1窓見送る候補を、
同一seed 8局のpaired rollout (教師継続) で検証し、paired t>2のものだけ採用。上限200 state。
通らなければM1で出す (追いつきのみ)。
評価: M1/M2(/M3)を新規band seed 5300各32戦 + t168 probe + seed 5000の回帰check (tgtの-67k再発を見張る)。
予測: M1で教師-8k以内・5000正常化。M3は+1〜3kの上積みがあれば御の字。
外れ: M1がdry_ep7を下回る (市場fine-tune不発→設計棄却)、WAIT不採択 (追い越し要素なしで出す)。
rl/コードはtarget datasetのcode_rl (--market-only入り) を使う。dataset更新が必要。
乾式実行: SKIP_BUILD=1 KAGGLE_ROOT=<偽ツリー> KAGGLE_DRY=1 .venv/bin/python rl/kaggle33/run_mkt_ex.py
"""
import glob, os, shutil, subprocess, sys, time
import numpy as np
ROOT = os.environ.get("KAGGLE_ROOT", "/kaggle"); INPUT = os.path.join(ROOT, "input"); output = os.path.join(ROOT, "working"); DRY = os.environ.get("KAGGLE_DRY")
PROCS, PER = (2, 2) if DRY else (4, 250)


def first(suffix, prefer):
    m = sorted(glob.glob(os.path.join(INPUT, "**", suffix), recursive=True)); m = [x for x in m if prefer in x] or m
    assert m, suffix; return m[0]


_tgt = [x for x in sorted(glob.glob(os.path.join(INPUT, "**", "code_rl", "train_raw.py"), recursive=True)) if "target-e081" in x]
rl = os.path.dirname(_tgt[0]) if _tgt else os.path.dirname(first("train_raw.py", "panel-agents"))
kagsim_src = os.path.dirname(os.path.dirname(first("sim/sim.hpp", "v41self")))
agents = os.path.dirname(os.path.dirname(first("e082/main.py", "panel-agents"))); A = lambda n: os.path.join(agents, n, "main.py")
print("rl:", rl, "teacher:", A("e082"), flush=True)
panel = [A(n) for n in ("e082", "e082", "e072", "e074", "pub_herd2700", "v46", "v48", "v41")]

os.makedirs(output, exist_ok=True)
basepp = os.environ.get("PYTHONPATH", "")
if os.environ.get("SKIP_BUILD"):
    env = dict(os.environ, PYTHONPATH=rl + (":" + basepp if basepp else ""))
else:
    sim = os.path.join(output, "kagsim_src"); shutil.copytree(kagsim_src, sim, dirs_exist_ok=True)
    r = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=sim, capture_output=True, text=True); assert r.returncode == 0, r.stderr[-300:]
    env = dict(os.environ, PYTHONPATH=sim + ":" + rl)

# Phase 1: E082 clean data
clean = os.path.join(output, "clean82"); t0 = time.time()
procs = [subprocess.Popen([sys.executable, os.path.join(rl, "gen_selfplay.py"), "--raw", "--agent", A("e082"), "--opp", *panel,
                           "--seed0", str(250000 + p * PER), "--games", str(PER), "--out", clean], cwd=rl, env=dict(env, OMP_NUM_THREADS="1"),
                          stdout=open(os.path.join(output, f"gen_{p}.log"), "w"), stderr=subprocess.STDOUT) for p in range(PROCS)]
print("gen rc", [p.wait() for p in procs], "clean games", len(glob.glob(os.path.join(clean, "*.npz"))), f"[{time.time() - t0:.0f}s]", flush=True)

# donor init: dry_ep7 (better units: seed 5000 -10k vs tgt -67k)
donor = None
cands = sorted(glob.glob(os.path.join(INPUT, "**", "raw_e081_dry_ep7.pt"), recursive=True))
if cands: donor = cands[0]
print("donor:", donor, flush=True)

m1 = os.path.join(output, "mkt1.pt"); t0 = time.time()
EP1 = 1 if DRY else 4
r = subprocess.run([sys.executable, os.path.join(rl, "train_raw.py"), "--data", os.path.join(clean, "*.npz"), "--out", m1,
                    "--init", donor or m1, "--epochs", str(EP1), "--bs", "512", "--lr", "7e-4",
                    "--max-games", "12" if DRY else "100000", "--market-only",
                    "--patience", "0" if DRY else "2", "--min-delta", "0.005"], cwd=rl, env=env)
print(f"M1 train rc {r.returncode} [{time.time() - t0:.0f}s]", flush=True); assert r.returncode == 0

# Phase 2: reactive-WAIT search labels (bounded; adopted only on paired t>2)
m2 = m1
if not DRY:
    gen = os.path.join(rl, "gen_wait.py")
    if os.path.exists(gen):
        wait = os.path.join(output, "wait")
        r = subprocess.run([sys.executable, gen, "--student", m1.replace(".pt", "_best.pt"), "--agent", A("e082"),
                            "--opp", *panel, "--seed0", "260000", "--games", "200", "--out", wait,
                            "--seeds", "8", "--treq", "2.0"], cwd=rl, env=dict(env, OMP_NUM_THREADS="1"))
        print(f"wait-gen rc {r.returncode}", flush=True)
        got = len(glob.glob(os.path.join(wait, "*.npz")))
        print("wait labels:", got, flush=True)
        if got >= 20:
            m2 = os.path.join(output, "mkt2.pt"); t0 = time.time()
            r = subprocess.run([sys.executable, os.path.join(rl, "train_raw.py"), "--data", os.path.join(clean, "*.npz"),
                                os.path.join(wait, "*.npz"), "--out", m2, "--init", m1.replace(".pt", "_best.pt"),
                                "--epochs", "4", "--bs", "512", "--lr", "3e-4", "--market-only",
                                "--patience", "2", "--min-delta", "0.005"], cwd=rl, env=env)
            print(f"M2 train rc {r.returncode} [{time.time() - t0:.0f}s]", flush=True); assert r.returncode == 0
    else:
        print("no gen_wait.py: shipping catch-up only (M1)", flush=True)

with open(os.path.join(output, "eval.txt"), "a") as rep:
    for ck in ([m1] if DRY else [m1.replace(".pt", "_best.pt"), m2.replace(".pt", "_best.pt") if m2 != m1 else m1.replace(".pt", "_best.pt")]):
        if not os.path.exists(ck): continue
        tag = os.path.basename(ck)
        for opp, games, seed0 in (("e072", 2 if DRY else 32, 5300), ("v41", 2 if DRY else 32, 5300), ("v41", 2, 5000)):
            t0 = time.time(); r = subprocess.run([sys.executable, os.path.join(rl, "raw.py"), ck, "--games", str(games), "--vs", A(opp),
                                                  "--seed0", str(seed0)], cwd=rl, env=env, capture_output=True, text=True)
            line = (r.stdout.strip().splitlines() or [r.stderr[-300:]])[-1]; print(f"mkt {tag} vs {opp} s{seed0}: {line} [{time.time() - t0:.0f}s]", flush=True); rep.write(f"mkt {tag} vs {opp} s{seed0}: {line}\n"); rep.flush()
    # t168 probe on the 5014 board (regen 2 teacher games; fast)
    pdir = os.path.join(output, "probe")
    r = subprocess.run([sys.executable, os.path.join(rl, "gen_selfplay.py"), "--raw", "--agent", A("e082"), "--opp", A("v41"),
                        "--seed0", "5014", "--games", "2", "--out", pdir], cwd=rl, env=dict(env, OMP_NUM_THREADS="1"),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-300:]
print("ALL DONE", flush=True)
