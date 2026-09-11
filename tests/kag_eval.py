"""kagsim (bit-exact C++ engine, ~0.1s/game) による高速評価。

  席差し替え: .venv/bin/python tests/kag_eval.py agents/X/main.py --replays 'tmp/top0911/**/episode-*.json' --team MMN0222 [--base agents/sr0909_base/main.py]
             --team を省くと両席それぞれを候補で差し替える (相手はテープ)。
  直接対決:   .venv/bin/python tests/kag_eval.py agents/X/main.py --vs agents/sr0909_base/main.py --games 24
"""
import argparse, glob, hashlib, importlib.util, json, os, sys, time
from collections import defaultdict
import kagsim

PASS = {"farmer": ["PASS"], "hands": [], "market": []}
KNOWN = {'267f5f1e': 'SR0908', 'b3105bcc': 'SR0909', '0b496ee6': 'ThomasOpen', '87b9b0d1': 'KaitoOpen', 'a17674d6': 'Griefer'}


def load(p):
    d = os.path.dirname(os.path.abspath(p)); sys.path.insert(0, d)
    s = importlib.util.spec_from_file_location("cand_" + hashlib.md5(p.encode()).hexdigest()[:8], p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return getattr(m, "agent_entry", m.agent), m


def fresh(mod):
    """新しい試合の前に、モジュール内のシングルトンをリセットする。"""
    for k in ("_LIVE", "_ROUTER", "_POLICY"):
        if hasattr(mod, k): setattr(mod, k, None)


def plan(a):
    a = a or PASS
    return (tuple(a.get("farmer") or ["PASS"]), tuple(tuple(h) for h in (a.get("hands") or [])))


def h72(steps, seat):
    return hashlib.sha256(json.dumps([plan(steps[t + 1][seat].get("action")) for t in range(72)]).encode()).hexdigest()[:8]


def play(seed, agents, mods, shops=None):
    """agents[i] は callable (live) か list (tape)。"""
    for m in mods:
        if m is not None: fresh(m)
    g = kagsim.Game(seed) if shops is None else kagsim.Game(seed, 720, shops)
    while not g.done:
        t = g.step_count
        acts = [a[t] if isinstance(a, list) else a(g.observe(i)) for i, a in enumerate(agents)]
        g.step(acts[0], acts[1])
    return g.reward(0), g.reward(1), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent"); ap.add_argument("--replays"); ap.add_argument("--team"); ap.add_argument("--base")
    ap.add_argument("--vs"); ap.add_argument("--games", type=int, default=24); ap.add_argument("--seed0", type=int, default=0); ap.add_argument("--n", type=int, default=10000)
    a = ap.parse_args()
    cand, cmod = load(a.agent)
    t0 = time.time()
    if a.vs:
        opp, omod = load(a.vs); tot = w = n = 0; rows = []
        for seed in range(a.seed0, a.seed0 + a.games // 2):
            r0, r1, _ = play(seed, [cand, opp], [cmod, omod]); m0 = r0 - r1
            r0, r1, _ = play(seed, [opp, cand], [omod, cmod]); m1 = r1 - r0
            rows.append((seed, m0, m1)); tot += m0 + m1; n += 2; w += (m0 > 0) + (m1 > 0)
        print("\n".join(f"seed {s:>5} as-p0 {m0:>+8,.0f} as-p1 {m1:>+8,.0f} pair {m0+m1:>+8,.0f}" for s, m0, m1 in rows))
        print(f"{a.agent} vs {a.vs}: {w}W{n-w}L avg {tot/n:+,.0f} min-pair {min(m0+m1 for _,m0,m1 in rows):+,.0f}  [{time.time()-t0:.1f}s]")
        return
    base, bmod = load(a.base) if a.base else (None, None)
    seen = set(); agg = defaultdict(list); k = 0
    for f in sorted(glob.glob(a.replays, recursive=True)):
        r = json.load(open(f)); eid = r["info"].get("EpisodeId", f)
        if eid in seen or len(r["steps"]) < 720: continue
        seen.add(eid); steps = r["steps"]; names = r["info"]["TeamNames"]; seed = r["info"]["seed"]
        seats = [names.index(a.team)] if a.team and a.team in names else ([] if a.team else [0, 1])
        for me in seats:
            if k >= a.n: break
            tape = [steps[t + 1][1 - me].get("action") or PASS for t in range(719)]
            lin = KNOWN.get(h72(steps, 1 - me), "other:" + names[1 - me])
            ag = [None, None]; ag[me] = cand; ag[1 - me] = tape
            r0, r1, g = play(seed, ag, [cmod if i == me else None for i in (0, 1)])
            new = (r0 - r1) if me == 0 else (r1 - r0)
            if base is not None:
                ag[me] = base; b0, b1, _ = play(seed, ag, [bmod if i == me else None for i in (0, 1)])
                ref = (b0 - b1) if me == 0 else (b1 - b0)
            else:
                ref = r["rewards"][me] - r["rewards"][1 - me]
            agg[lin].append((ref, new)); k += 1
            tel = {kk: v for kk, v in g.telemetry(me).items() if v and kk.startswith("refused")}
            print(f"{str(eid)[-9:]} seat{me} vs {lin:<24} {'base' if base else 'recorded'} {ref:+8.0f} -> {new:+8.0f} (delta {new-ref:+7.0f}) {tel}", flush=True)
    tot = [x for v in agg.values() for x in v]
    print("== by opponent lineage (ref W -> new W, mean delta)")
    for lin, v in sorted(agg.items(), key=lambda x: -len(x[1])):
        print(f"  {lin:<26} n={len(v):3d} W{sum(1 for r_,_ in v if r_>0)}->{sum(1 for _,n_ in v if n_>0)}  mean delta {sum(n_-r_ for r_,n_ in v)/len(v):+7.0f}  mean new {sum(n_ for _,n_ in v)/len(v):+7.0f}")
    if tot: print(f"  ALL n={len(tot)} W{sum(1 for r_,_ in tot if r_>0)}->{sum(1 for _,n_ in tot if n_>0)} mean delta {sum(n_-r_ for r_,n_ in tot)/len(tot):+.0f}  [{time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
