"""ライブ層 (sr0909_live の main.py) を載せた候補テープを L1 で評価する。反応クローン (0909 base / E055) 相手、seeds×両席。"""
import importlib.util, json, os, sys, tempfile, shutil
from pathlib import Path
import kagsim
from concurrent.futures import ProcessPoolExecutor

WRAP_DIR = Path("agents/sr0909_live")


def _load_agent_from_dir(d, name):
    p = str(Path(d) / "main.py"); sys.path.insert(0, str(d))
    s = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def _worker(args):
    tapes_path, opp_dir, seeds, seat_list = args
    tmp = Path(tempfile.mkdtemp(prefix="wrapped_"))
    for f in ("main.py", "router.py", "LICENSE.txt"): shutil.copy(WRAP_DIR / f, tmp / f)
    shutil.copy(tapes_path, tmp / "actions.json")
    cand = _load_agent_from_dir(tmp, "cand_" + tmp.name); opp = _load_agent_from_dir(opp_dir, "opp_" + Path(opp_dir).name + tmp.name)
    out = []
    for seed in seeds:
        for me in seat_list:
            for m in (cand, opp):
                for k in ("_LIVE", "_POLICY", "_ROUTER"):
                    if hasattr(m, k): setattr(m, k, None)
            g = kagsim.Game(seed)
            while not g.done:
                a = cand.agent(g.observe(me)); b = opp.agent(g.observe(1 - me))
                g.step(*((a, b) if me == 0 else (b, a)))
            out.append(g.reward(me) - g.reward(1 - me))
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def evaluate_wrapped(tapes, seeds=range(2000, 2024), opps=("agents/sr0909_base", "agents/sr0909_live"), workers=8):
    tp = Path(tempfile.mkdtemp(prefix="tapes_")) / "actions.json"; json.dump(tapes, open(tp, "w"))
    seeds = list(seeds); chunks = [seeds[i::workers] for i in range(workers)]
    res = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for opp in opps:
            futs = [ex.submit(_worker, (str(tp), opp, ch, [0, 1])) for ch in chunks if ch]
            ms = [m for f in futs for m in f.result()]
            res[Path(opp).name] = {"mean": sum(ms) / len(ms), "wr": sum(1 for m in ms if m > 0) / len(ms), "n": len(ms)}
    shutil.rmtree(tp.parent, ignore_errors=True)
    return res


if __name__ == "__main__":
    import time
    t0 = time.time(); r = evaluate_wrapped(json.load(open(sys.argv[1])))
    print(r, f"[{time.time()-t0:.1f}s]")
