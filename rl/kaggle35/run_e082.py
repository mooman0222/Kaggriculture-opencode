"""Kaggle GPU: E082-teacher raw BC from scratch, stage 1 (2026-09-26).

E081教師の系列 (raw-bc→recover→dry→target) は市場fine-tune (mkt-e082) で崩壊し、
recover往復もbulge移動で打ち止め。E082 (E081 + イチゴcap6) が提出され教師を更新する。
E082はE081と農場がt中央値360手まで一致するためunitは転移するが、一から学び直す
(市場ラベルのcap衝突を避けるためE081 shardは混ぜない)。
手順: gen E082 clean 1000 + dry 500 (--dry 4,20) → 最初から8 epoch (bs 512 lr 7e-4、
--patience 3 --min-delta 0.005) → ep奇数+bestを新規band seed 5400各32戦 + t168 probe
(E082-5014盤面) + seed 5000回帰 (2局、tgtの-67k再発を見張る)。
予測: ep7で対v41 -15k前後・対E072 -12k前後 (E081-dry_ep7と同帯)。E082教師自身は
対E072 +? / 対v41 +? (live測定前。直接対決でE081以上を確認ずみ)。
外れ: ep7で-30kより悪い (E082教師がBCに不向き→教師据え置き)。
次: kaggle36 (このep3/5をstudentにrecover 1000 + 再学習8epoch)。
乾式実行: SKIP_BUILD=1 KAGGLE_ROOT=<偽ツリー> KAGGLE_DRY=1 .venv/bin/python rl/kaggle35/run_e082.py
"""
import glob, os, shutil, subprocess, sys, time
import numpy as np
ROOT = os.environ.get("KAGGLE_ROOT", "/kaggle"); INPUT = os.path.join(ROOT, "input"); output = os.path.join(ROOT, "working"); DRY = os.environ.get("KAGGLE_DRY")
PROCS, PER = (2, 2) if DRY else (4, 250)
CLEAN_N, DRY_N = (4, 4) if DRY else (1000, 500)


def first(suffix, prefer):
    m = sorted(glob.glob(os.path.join(INPUT, "**", suffix), recursive=True)); m = [x for x in m if prefer in x] or m
    assert m, suffix; return m[0]


_tgt = [x for x in sorted(glob.glob(os.path.join(INPUT, "**", "code_rl", "train_raw.py"), recursive=True)) if "target-e081" in x]
assert _tgt, "code_rl with --market-only train_raw"
rl = os.path.dirname(_tgt[0])
kagsim_src = os.path.dirname(os.path.dirname(first("sim/sim.hpp", "v41self")))
_ta = [x for x in sorted(glob.glob(os.path.join(INPUT, "**", "agents", "e082", "main.py"), recursive=True)) if "target-e081" in x]
assert _ta, "target-e081/agents/e082/main.py"
agents_t = os.path.dirname(os.path.dirname(_ta[0]))
agents_p = os.path.dirname(os.path.dirname(first("e072/main.py", "panel-agents")))
A_t = lambda n: os.path.join(agents_t, n, "main.py"); A_p = lambda n: os.path.join(agents_p, n, "main.py")
print("rl:", rl, "teacher:", A_t("e082"), flush=True)
panel = [A_t("e082"), A_t("e082"), A_p("e072"), A_p("e074"), A_p("pub_herd2700"), A_p("v46"), A_p("v48"), A_p("v41")]

os.makedirs(output, exist_ok=True)
basepp = os.environ.get("PYTHONPATH", "")
if os.environ.get("SKIP_BUILD"):
    env = dict(os.environ, PYTHONPATH=rl + (":" + basepp if basepp else ""))
else:
    sim = os.path.join(output, "kagsim_src"); shutil.copytree(kagsim_src, sim, dirs_exist_ok=True)
    r = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=sim, capture_output=True, text=True); assert r.returncode == 0, r.stderr[-300:]
    env = dict(os.environ, PYTHONPATH=sim + ":" + rl)

t0 = time.time(); jobs = []
clean = os.path.join(output, "clean"); dry = os.path.join(output, "dry")
for p in range(PROCS):
    jobs.append(subprocess.Popen([sys.executable, os.path.join(rl, "gen_selfplay.py"), "--raw", "--agent", A_t("e082"), "--opp", *panel,
                                  "--seed0", str(250000 + p * PER), "--games", str(PER), "--out", clean],
                                 cwd=rl, env=dict(env, OMP_NUM_THREADS="1"),
                                 stdout=open(os.path.join(output, f"gen_c{p}.log"), "w"), stderr=subprocess.STDOUT))
    jobs.append(subprocess.Popen([sys.executable, os.path.join(rl, "gen_selfplay.py"), "--raw", "--dry", "4,20", "--agent", A_t("e082"), "--opp", *panel,
                                  "--seed0", str(251000 + p * (PER if DRY else PER // 2)), "--games", str(PER if DRY else PER // 2), "--out", dry],
                                 cwd=rl, env=dict(env, OMP_NUM_THREADS="1"),
                                 stdout=open(os.path.join(output, f"gen_d{p}.log"), "w"), stderr=subprocess.STDOUT))
print("gen rc", [p.wait() for p in jobs], flush=True)
nc, nd = len(glob.glob(os.path.join(clean, "*.npz"))), len(glob.glob(os.path.join(dry, "*.npz")))
print(f"clean {nc} dry {nd} [{time.time() - t0:.0f}s]", flush=True)
assert nc >= (4 if DRY else 500) and nd >= (4 if DRY else 250), "gen produced no shards"

out = os.path.join(output, "raw_e082.pt"); t0 = time.time()
EP = 1 if DRY else 8
r = subprocess.run([sys.executable, os.path.join(rl, "train_raw.py"), "--data", os.path.join(clean, "*.npz"), os.path.join(dry, "*.npz"),
                    "--out", out, "--epochs", str(EP), "--bs", "512", "--lr", "7e-4",
                    "--max-games", "12" if DRY else "100000", "--patience", "0" if DRY else "3", "--min-delta", "0.005"], cwd=rl, env=env)
print(f"train rc {r.returncode} [{time.time() - t0:.0f}s]", flush=True); assert r.returncode == 0

ckpts = sorted(glob.glob(out.replace(".pt", "_ep*.pt")))
ckpts = [c for c in ckpts if int(c.split("_ep")[1].split(".")[0]) % 2 == 1]
if os.path.exists(out.replace(".pt", "_best.pt")): ckpts.append(out.replace(".pt", "_best.pt"))
print("eval ckpts:", [os.path.basename(c) for c in ckpts], flush=True)
with open(os.path.join(output, "eval.txt"), "a") as rep:
    for ck in ckpts:
        tag = os.path.basename(ck)
        for opp, games, seed0 in (("e072", 2 if DRY else 32, 5400), ("v41", 2 if DRY else 32, 5400), ("v41", 2, 5000)):
            t0 = time.time(); r = subprocess.run([sys.executable, os.path.join(rl, "raw.py"), ck, "--games", str(games), "--vs", A_p(opp),
                                                  "--seed0", str(seed0)], cwd=rl, env=env, capture_output=True, text=True)
            line = (r.stdout.strip().splitlines() or [r.stderr[-300:]])[-1]; print(f"e82 {tag} vs {opp} s{seed0}: {line} [{time.time() - t0:.0f}s]", flush=True); rep.write(f"e82 {tag} vs {opp} s{seed0}: {line}\n"); rep.flush()
    try:
        pdir = os.path.join(output, "probe")
        r = subprocess.run([sys.executable, os.path.join(rl, "gen_selfplay.py"), "--raw", "--agent", A_t("e082"), "--opp", A_p("v41"),
                            "--seed0", "5014", "--games", "2", "--out", pdir], cwd=rl, env=dict(env, OMP_NUM_THREADS="1"),
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-300:]
        pfile = os.path.join(pdir, "5014_s0.npz")
        psell = """
import sys
sys.path.insert(0, %r)
import numpy as np, torch
from raw import Policy4, KINDS, KIND_INDEX
SF = KIND_INDEX['SELL_FERTILIZER']
a = np.load(%r); t = 168
model = Policy4()
for ck in %r:
    sd = torch.load(ck, map_location='cpu'); model.load_state_dict(sd, strict=True); model.eval()
    with torch.no_grad():
        o = model(torch.from_numpy(a['tiles'][t:t+1]).float(), torch.from_numpy(a['units'][t:t+1]).float(),
                  torch.from_numpy(a['items'][t:t+1]), torch.from_numpy(a['glob'][t:t+1]))
        p = o['mkind'][0, 0].softmax(-1); ps = float(p[SF]); ph = float(p[KIND_INDEX['HIRE']])
        print(f"t168probe {ck.split('/')[-1]}: P(SELL)={ps:.3f} P(HIRE)={ph:.3f} label={KINDS[int(a['mk'][t,0])]}", flush=True)
""" % (rl, pfile, ckpts)
        r = subprocess.run([sys.executable, "-c", psell], cwd=rl, env=env, capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr[-500:], flush=True)
        rep.write(r.stdout); rep.flush()
    except Exception as e:
        print(f"probe skipped: {type(e).__name__} {str(e)[:200]}", flush=True)
if "sim" in dir(): shutil.rmtree(sim, ignore_errors=True)
print("ALL DONE", flush=True)
