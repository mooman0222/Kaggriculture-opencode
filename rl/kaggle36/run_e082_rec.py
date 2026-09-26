"""Kaggle GPU: E082-teacher raw BC stage 2 — recover + retrain from scratch (2026-09-26).

K35 (raw-e082, E082 clean1000 + dry500, 8epoch最初から): ep7で対v41 −33.0k / 対E072 −37.8k。
E081-s1 (ep3 −21.7k/−55.8k) と同帯で教師は viable。ただしseed 5000は全epochで−100k〜−190k
(d8所持金8まで落ちる資金ショート→農場死。tgtの−67kより悪い)。E081の道筋どおりrecoverで直す。
手順: student=e82_ep7でrecover 1000局 (takeover 24) → clean1000 + dry500 (K35出力をkernel_sourcesで再利用)
+ recover1000 = 2500局で最初から8 epoch (bs 512 lr 7e-4、--patience 3 --min-delta 0.005)
→ ep奇数+bestを新規band seed 5500各32戦 + t168 probe (E082-5014盤面) + seed 5000回帰。
予測: ep7で対v41 −15k前後・対E072 −12k前後 (E081-dry_ep7同帯)、seed 5000は−30k以内。
外れ: ep7で−30kより悪いか5000が−50kより悪いまま (E082教師のd8資金繰りがBC不能→打ち止め)。
乾式実行: SKIP_BUILD=1 KAGGLE_ROOT=<偽ツリー> KAGGLE_DRY=1 .venv/bin/python rl/kaggle36/run_e082_rec.py
"""
import glob, os, shutil, subprocess, sys, time
import numpy as np
ROOT = os.environ.get("KAGGLE_ROOT", "/kaggle"); INPUT = os.path.join(ROOT, "input"); output = os.path.join(ROOT, "working"); DRY = os.environ.get("KAGGLE_DRY")
PROCS, PER = (2, 2) if DRY else (4, 250)


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
student = first("raw_e082_ep7.pt", "raw-e082")
clean_dirs = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(INPUT, "**", "25*_s*.npz"), recursive=True) if "uop" in np.load(p).files})
print("rl:", rl, "teacher:", A_t("e082"), "student:", student, "clean_dirs:", clean_dirs, flush=True)
assert clean_dirs, "K35 shards via kernel_sources"
panel = [A_t("e082"), A_t("e082"), A_p("e072"), A_p("e074"), A_p("pub_herd2700"), A_p("v46"), A_p("v48"), A_p("v41")]

os.makedirs(output, exist_ok=True)
basepp = os.environ.get("PYTHONPATH", "")
if os.environ.get("SKIP_BUILD"):
    env = dict(os.environ, PYTHONPATH=rl + (":" + basepp if basepp else ""))
else:
    sim = os.path.join(output, "kagsim_src"); shutil.copytree(kagsim_src, sim, dirs_exist_ok=True)
    r = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=sim, capture_output=True, text=True); assert r.returncode == 0, r.stderr[-300:]
    env = dict(os.environ, PYTHONPATH=sim + ":" + rl)

recover = os.path.join(output, "recover"); t0 = time.time()
procs = [subprocess.Popen([sys.executable, os.path.join(rl, "gen_recover.py"), "--student", student, "--agent", A_t("e082"), "--opp", *panel,
                           "--seed0", str(270000 + p * PER), "--games", str(PER), "--out", recover], cwd=rl, env=dict(env, OMP_NUM_THREADS="1"),
                          stdout=open(os.path.join(output, f"rec_{p}.log"), "w"), stderr=subprocess.STDOUT) for p in range(PROCS)]
print("rec rc", [p.wait() for p in procs], "recover games", len(glob.glob(os.path.join(recover, "*.npz"))), f"[{time.time() - t0:.0f}s]", flush=True)
if not DRY: assert len(glob.glob(os.path.join(recover, "*.npz"))) >= 500, "recover gen failed"

out = os.path.join(output, "raw_e082_rec.pt"); t0 = time.time()
EP = 1 if DRY else 8
datas = [os.path.join(d, "*.npz") for d in clean_dirs] + [os.path.join(recover, "*.npz")]
r = subprocess.run([sys.executable, os.path.join(rl, "train_raw.py"), "--data", *datas, "--out", out,
                    "--epochs", str(EP), "--bs", "512", "--lr", "7e-4",
                    "--max-games", "12" if DRY else "100000", "--patience", "0" if DRY else "3", "--min-delta", "0.005"], cwd=rl, env=env)
print(f"train rc {r.returncode} [{time.time() - t0:.0f}s]", flush=True); assert r.returncode == 0

ckpts = sorted(glob.glob(out.replace(".pt", "_ep*.pt")))
ckpts = [c for c in ckpts if int(c.split("_ep")[1].split(".")[0]) % 2 == 1]
if os.path.exists(out.replace(".pt", "_best.pt")): ckpts.append(out.replace(".pt", "_best.pt"))
print("eval ckpts:", [os.path.basename(c) for c in ckpts], flush=True)
with open(os.path.join(output, "eval.txt"), "a") as rep:
    for ck in ckpts:
        tag = os.path.basename(ck)
        for opp, games, seed0 in (("e072", 2 if DRY else 32, 5500), ("v41", 2 if DRY else 32, 5500), ("v41", 2, 5000)):
            t0 = time.time(); r = subprocess.run([sys.executable, os.path.join(rl, "raw.py"), ck, "--games", str(games), "--vs", A_p(opp),
                                                  "--seed0", str(seed0)], cwd=rl, env=env, capture_output=True, text=True)
            line = (r.stdout.strip().splitlines() or [r.stderr[-300:]])[-1]; print(f"e82r {tag} vs {opp} s{seed0}: {line} [{time.time() - t0:.0f}s]", flush=True); rep.write(f"e82r {tag} vs {opp} s{seed0}: {line}\n"); rep.flush()
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
