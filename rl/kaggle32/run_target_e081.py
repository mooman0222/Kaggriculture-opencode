"""Kaggle GPU: E081 stage 4 — targeted recovery + retrain from scratch (2026-09-25).

第 3 段階 (dry_ep7: 対 v41 -15.1k / 対 E072 -11.5k) の残る裾は 3 型:
(A) 5014型: d7 t168に教師がSELLしない少数派分岐 (所持金≥1400) をcloneがSELLし、羊ウィンドウを逸して-136k。
    手元probeでt168はSELL 0.576/HIRE 0.412の接戦、dry poolの少数派は85/855局(10%)あるので蕪増しでは直らない。
(B) 終盤出血: d24まで追従→d26-28に給餌漏れで17→6頭 (5005s1・5008s0)。
(C) 相手富化 (5011): 追わない (相対タイミングのゼロサム、BCでは学べない)。
手元で標的recover 150局を生成済み (student=dry_ep7、問題世界5000-5029と近傍5030-5079、対v41/e072)。
手順: clean 1000 + recover 1000 + dry 1000 + 標的 150局で最初から学習 (bs 512、lr 7e-4、--epochs 10 --patience 2 --min-delta 0.005)
-> _best + 奇数epを新規band (seed 5200、汚染なし) で対v41・対E072各32戦 + 問題世界16局 + t168 probe。
予測: 5014型→-20k以内かつt168 probe P(SELL)<0.5、新規bandでdry_ep7以上 (-15k/-11kを維持以上)。
外れ: t168がSELLのまま (stage-2の決まった誤りと同類→打ち止め)、新規bandで悪化 (標的過適合→不採用)。
rl/コードは target dataset内の code_rl を優先する (panel-agentsの再uploadなしでtrain_raw.pyのearly stoppingを使うため)。
乾式実行: SKIP_BUILD=1 KAGGLE_ROOT=<偽ツリー> KAGGLE_DRY=1 .venv/bin/python rl/kaggle32/run_target_e081.py
"""
import glob, os, shutil, subprocess, sys, time
import numpy as np
ROOT = os.environ.get("KAGGLE_ROOT", "/kaggle"); INPUT = os.path.join(ROOT, "input"); output = os.path.join(ROOT, "working"); DRY = os.environ.get("KAGGLE_DRY")
EPOCHS = 1 if DRY else 10
FRESH_SEED, FRESH_GAMES = (5200, 2) if DRY else (5200, 32)
REF = [(5005, 2), (5011, 2)] if DRY else [(5005, 8), (5011, 8)]  # 問題世界 (学習に含むため参考値)


def first(suffix, prefer):
    m = sorted(glob.glob(os.path.join(INPUT, "**", suffix), recursive=True)); m = [x for x in m if prefer in x] or m
    assert m, suffix; return m[0]


# 更新版train_raw.py (early stopping) を含むcode_rlを優先、無ければpanel-agents
_tgt = [x for x in sorted(glob.glob(os.path.join(INPUT, "**", "code_rl", "train_raw.py"), recursive=True)) if "target-e081" in x]
rl = os.path.dirname(_tgt[0]) if _tgt else os.path.dirname(first("train_raw.py", "panel-agents"))
kagsim_src = os.path.dirname(os.path.dirname(first("sim/sim.hpp", "v41self")))
agents = os.path.dirname(os.path.dirname(first("e081/main.py", "panel-agents"))); A = lambda n: os.path.join(agents, n, "main.py")
print("rl:", rl, flush=True)

# shard pool: ファイル名のseed帯で分類 (中身のloadは各dir先頭1件だけ確認)
pool = {}
for f in glob.glob(os.path.join(INPUT, "**", "*_s*.npz"), recursive=True):
    seed = int(os.path.basename(f).split("_")[0])
    tag = "clean" if 199999 < seed < 210000 else "recover" if 229999 < seed < 240000 else "dry" if 239999 < seed < 250000 else "target" if 4999 < seed < 6000 else None
    if tag: pool.setdefault(tag, os.path.dirname(f))
print("pool:", {k: (v, len(glob.glob(os.path.join(v, "*.npz")))) for k, v in pool.items()}, flush=True)
assert set(pool) == {"clean", "recover", "dry", "target"}, pool
probe0 = sorted(glob.glob(os.path.join(pool["clean"], "*.npz")))[0]
assert "uop" in np.load(probe0).files, "old-format shard?"
datas = [os.path.join(pool[t], "*.npz") for t in ("clean", "recover", "dry", "target")]

os.makedirs(output, exist_ok=True)
basepp = os.environ.get("PYTHONPATH", "")
if os.environ.get("SKIP_BUILD"):
    env = dict(os.environ, PYTHONPATH=rl + (":" + basepp if basepp else ""))  # 手元試験用: ビルド済みkagsimを使う
else:
    sim = os.path.join(output, "kagsim_src"); shutil.copytree(kagsim_src, sim, dirs_exist_ok=True)
    r = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=sim, capture_output=True, text=True); assert r.returncode == 0, r.stderr[-300:]
    env = dict(os.environ, PYTHONPATH=sim + ":" + rl)

out = os.path.join(output, "raw_e081_tgt.pt"); t0 = time.time()
cmd = [sys.executable, os.path.join(rl, "train_raw.py"), "--data", *datas, "--out", out,
       "--epochs", str(EPOCHS), "--bs", "512", "--lr", "7e-4", "--max-games", "12" if DRY else "100000",
       "--patience", "0" if DRY else "2", "--min-delta", "0.005"]
r = subprocess.run(cmd, cwd=rl, env=env)
print(f"train rc {r.returncode} [{time.time() - t0:.0f}s]", flush=True); assert r.returncode == 0

ckpts = sorted(glob.glob(out.replace(".pt", "_ep*.pt")))
ckpts = [c for c in ckpts if int(c.split("_ep")[1].split(".")[0]) % 2 == 1]  # 奇数epのみ評価
if os.path.exists(out.replace(".pt", "_best.pt")): ckpts.append(out.replace(".pt", "_best.pt"))
print("eval ckpts:", [os.path.basename(c) for c in ckpts], flush=True)


def probe_board():
    """t168 probe用の教師盤面 (5014) を2局生成。速い (~10s)。"""
    pdir = os.path.join(output, "probe")
    r = subprocess.run([sys.executable, os.path.join(rl, "gen_selfplay.py"), "--raw", "--agent", A("e081"), "--opp", A("v41"),
                        "--seed0", "5014", "--games", "2", "--out", pdir], cwd=rl, env=dict(env, OMP_NUM_THREADS="1"),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-300:]
    return os.path.join(pdir, "5014_s0.npz")


pfile = probe_board()
psell = """
import sys, glob
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
        print(f"t168probe {ck.split('_ep')[-1]}: P(SELL)={ps:.3f} P(HIRE)={ph:.3f} label={KINDS[int(a['mk'][t,0])]}", flush=True)
""" % (rl, pfile, ckpts)
r = subprocess.run([sys.executable, "-c", psell], cwd=rl, env=env, capture_output=True, text=True)
print(r.stdout.strip() or r.stderr[-500:], flush=True)
with open(os.path.join(output, "eval.txt"), "a") as rep:
    rep.write(r.stdout); rep.flush()
    for ck in ckpts:
        tag = os.path.basename(ck)
        for opp in ("e072", "v41"):
            t0 = time.time(); r = subprocess.run([sys.executable, os.path.join(rl, "raw.py"), ck, "--games", str(FRESH_GAMES), "--vs", A(opp),
                                                  "--seed0", str(FRESH_SEED)], cwd=rl, env=env, capture_output=True, text=True)
            line = (r.stdout.strip().splitlines() or [r.stderr[-300:]])[-1]; print(f"tgt {tag} fresh vs {opp}: {line} [{time.time() - t0:.0f}s]", flush=True); rep.write(f"tgt {tag} fresh vs {opp}: {line}\n"); rep.flush()
        for s0, g in REF:
            t0 = time.time(); r = subprocess.run([sys.executable, os.path.join(rl, "raw.py"), ck, "--games", str(g), "--vs", A("e072"),
                                                  "--seed0", str(s0)], cwd=rl, env=env, capture_output=True, text=True)
            line = (r.stdout.strip().splitlines() or [r.stderr[-300:]])[-1]; print(f"tgt {tag} ref{s0} vs e072: {line} [{time.time() - t0:.0f}s]", flush=True); rep.write(f"tgt {tag} ref{s0} vs e072: {line}\n"); rep.flush()
if "sim" in dir(): shutil.rmtree(sim, ignore_errors=True)
print("ALL DONE", flush=True)
