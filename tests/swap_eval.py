"""実戦リプレイの席差し替え (ショップ列を記録どおりに固定)。相手は記録行動のテープ (データ) で、コードは動かさない。

  .venv/bin/python tests/swap_eval.py --replays 'tmp/own0924/*.json' --team MMN0222 --cands agents/e073/main.py agents/X/main.py
  --team を省くと両席それぞれを差し替える。先頭の候補を基準にしたペア差 (平均 ± SE, t) を出す。
  候補は自前の agents/ 配下に限る (ダウンロードした第三者コードは実行しない)。
"""
import argparse, glob, hashlib, importlib.util, json, math, os, sys, time
from multiprocessing import Pool
import kagsim

PASS = {"farmer": ["PASS"], "hands": [], "market": []}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_M = []


def own_agent(p):
    ap = os.path.abspath(p)
    if not ap.startswith(os.path.join(ROOT, "agents") + os.sep):
        raise SystemExit(f"refuse: {p} is not under agents/ (only our own agents are executed)")
    return ap


def load(p):
    sys.path.insert(0, os.path.dirname(p))
    s = importlib.util.spec_from_file_location("sw_" + hashlib.md5(p.encode()).hexdigest()[:8], p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def init(paths):
    global _M
    _M = [load(p) for p in paths]


def episode(f, team):
    r = json.load(open(f)); st = r["steps"]
    names = r["info"]["TeamNames"]
    if len(st) < 720 or (team and team not in names): return []
    shops = st[-1][0]["observation"]["town"]["unlocked_shops"]
    out = []
    for me in ([names.index(team)] if team else [0, 1]):
        tape = [st[t + 1][1 - me].get("action") or PASS for t in range(719)]
        rec = (r["rewards"][me] or 0) - (r["rewards"][1 - me] or 0)
        out.append(dict(eid=r["info"]["EpisodeId"], me=me, seed=r["info"]["seed"], shops=shops, tape=tape, rec=rec, opp=names[1 - me]))
    return out


def run(job):
    ep, ci = job
    m = _M[ci]
    for k in ("_LIVE", "_ROUTER", "_POLICY"):
        if hasattr(m, k): setattr(m, k, None)
    ag = getattr(m, "agent_entry", m.agent)
    g = kagsim.Game(ep["seed"], 720, ep["shops"]); me = ep["me"]
    while not g.done:
        t = g.step_count
        mine = ag(g.observe(me))
        g.step(*((mine, ep["tape"][t]) if me == 0 else (ep["tape"][t], mine)))
    return ep["eid"], me, ci, g.reward(me) - g.reward(1 - me), g.reward(me)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", required=True); ap.add_argument("--team"); ap.add_argument("--cands", nargs="+", required=True)
    ap.add_argument("--jobs", type=int, default=10); ap.add_argument("--n", type=int, default=100000); ap.add_argument("--out")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    cands = [own_agent(c) for c in a.cands]
    t0 = time.time(); seen = set(); eps = []
    for f in sorted(glob.glob(a.replays, recursive=True)):
        for ep in episode(f, a.team):
            if (ep["eid"], ep["me"]) not in seen:
                seen.add((ep["eid"], ep["me"])); eps.append(ep)
    eps = eps[:a.n]
    res = {}
    with Pool(a.jobs, initializer=init, initargs=(cands,)) as pool:
        for eid, me, ci, m, own in pool.imap_unordered(run, [(e, ci) for e in eps for ci in range(len(cands))], chunksize=1):
            res[(eid, me, ci)] = (m, own)
    rows = [dict(eid=e["eid"], me=e["me"], opp=e["opp"], rec=e["rec"],
                 m=[res[(e["eid"], e["me"], ci)][0] for ci in range(len(cands))],
                 own=[res[(e["eid"], e["me"], ci)][1] for ci in range(len(cands))]) for e in eps]
    if not a.quiet:
        for r in rows:
            print(f"{r['eid']} s{r['me']} {r['opp'][:18]:18s} rec {r['rec']:+8.0f} " + " ".join(f"{x:+8.0f}" for x in r["m"]))
    n = len(rows)
    print(f"== {n} seats, {time.time() - t0:.0f}s")
    for ci, c in enumerate(a.cands):
        ms = [r["m"][ci] for r in rows]; d = [r["m"][ci] - r["m"][0] for r in rows]; md = sum(d) / n
        se = math.sqrt(sum((x - md) ** 2 for x in d) / max(1, n - 1) / n)
        print(f"  [{ci}] {c:40s} W {sum(x > 0 for x in ms):3d}/{n} mean {sum(ms) / n:+7.0f}  own {sum(r['own'][ci] for r in rows) / n:8.0f}"
              f"  d vs[0] {md:+6.0f} ± {se:4.0f} (t={md / se if se else 0:+.1f}) +{sum(x > 0 for x in d)}/-{sum(x < 0 for x in d)}")
    if a.out:
        json.dump(dict(cands=a.cands, rows=rows), open(a.out, "w"))


if __name__ == "__main__":
    main()
