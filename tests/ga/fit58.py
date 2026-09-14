"""E058 適応度: 候補テープを agents/e058 (v41 chassis + ライブ層) に載せ、(a) 実戦80席の記録ストリーム相手の席差し替え、
(b) 反応クローン (v41 / more-yield) 相手 seeds×両席 を L1 で回す。使い方: .venv/bin/python tests/ga/fit58.py tapes.json"""
import glob, importlib.util, json, os, shutil, sys, tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import kagsim

WRAP_DIR = Path("agents/e058")
POOL = "tmp/e058/pool_e057.json"
OPPS = {"v41": "tmp/e058/agents/ahmedberatozer_kaggriculture-v41-review-candidate",
        "my": "tmp/e058/agents/ahmedberatozer_more-yield-smarter-labor"}
PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def _load(d, name):
    p = str(Path(d) / "main.py"); sys.path.insert(0, str(d))
    s = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def _stage(tapes_path):
    tmp = Path(tempfile.mkdtemp(prefix="e058_"))
    for f in ("main.py", "base.py", "LICENSE.txt"): shutil.copy(WRAP_DIR / f, tmp / f)
    shutil.copy(tapes_path, tmp / "actions.json"); return tmp


def _pool_worker(args):
    tapes_path, entries = args
    tmp = _stage(tapes_path); cand = _load(tmp, "cand_" + tmp.name); out = []
    for p in entries:
        cand._LIVE = None; me = 1 - p["seat"]; g = kagsim.Game(p["seed"]); stream = p["stream"]
        while not g.done:
            t = g.step_count; a = cand.agent(g.observe(me)); b = stream[t] or PASS
            g.step(*((a, b) if me == 0 else (b, a)))
        out.append(g.reward(me) - g.reward(1 - me))
    shutil.rmtree(tmp, ignore_errors=True); return out


def _live_worker(args):
    tapes_path, opp_dir, seeds = args
    tmp = _stage(tapes_path); cand = _load(tmp, "cand_" + tmp.name); opp = _load(opp_dir, "opp_" + tmp.name); out = []
    for seed in seeds:
        for me in (0, 1):
            cand._LIVE = None; g = kagsim.Game(seed)
            while not g.done:
                a = cand.agent(g.observe(me)); b = opp.agent(g.observe(1 - me))
                g.step(*((a, b) if me == 0 else (b, a)))
            out.append(g.reward(me) - g.reward(1 - me))
    shutil.rmtree(tmp, ignore_errors=True); return out


_POOL = None


def evaluate(tapes, seeds=range(2000, 2012), live=True, workers=8, pool=None):
    global _POOL
    if _POOL is None: _POOL = json.load(open(POOL))
    entries = pool if pool is not None else _POOL
    tp = Path(tempfile.mkdtemp(prefix="tapes_")) / "actions.json"; json.dump(tapes, open(tp, "w"))
    res = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        chunks = [entries[i::workers] for i in range(workers)]
        futs = [ex.submit(_pool_worker, (str(tp), ch)) for ch in chunks if ch]
        lfuts = {}
        if live:
            seeds = list(seeds); sch = [seeds[i::4] for i in range(4)]
            for k, d in OPPS.items(): lfuts[k] = [ex.submit(_live_worker, (str(tp), d, ch)) for ch in sch if ch]
        ms = [m for f in futs for m in f.result()]
        res["pool"] = {"mean": sum(ms) / len(ms), "wr": sum(1 for m in ms if m > 0) / len(ms), "n": len(ms)}
        for k, fs in lfuts.items():
            ms = [m for f in fs for m in f.result()]
            res[k] = {"mean": sum(ms) / len(ms), "wr": sum(1 for m in ms if m > 0) / len(ms), "n": len(ms)}
    shutil.rmtree(tp.parent, ignore_errors=True)
    return res


def score(r):
    s = 0.5 * r["pool"]["mean"] + 3000 * 0.5 * r["pool"]["wr"]
    for k, w in (("v41", 0.3), ("my", 0.2)):
        if k in r: s += w * r[k]["mean"] + 3000 * w * r[k]["wr"]
    return s


def fmt(r):
    return " ".join(f"{k} {v['mean']:+.0f}/{v['wr']:.2f}" for k, v in r.items())


if __name__ == "__main__":
    import time
    t0 = time.time(); r = evaluate(json.load(open(sys.argv[1]))); print(fmt(r), f"score {score(r):+.0f}", f"[{time.time()-t0:.1f}s]")
