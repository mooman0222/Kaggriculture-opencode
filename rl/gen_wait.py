"""Reactive-WAIT labels: skip our window-head sale right after the rival dumped (bounded search, 2026-09-26).
Trigger: at a window-head step (t%4==1, 2<=t<710), town price of WOOL/MILK/STRAWBERRY fell >15% since t-1
(the rival's mod-0 dump cratered it) while our shed holds >=8 of it. Candidate: teacher's market minus that
item's SELLs (hold one window). Verification: paired rollout to game end (WAIT once, then teacher) vs teacher's
unmodified continuation, same seed/seat; adopt the batch only if paired t>2 over collected states.
Writes standard raw shards (labels = teacher's, except verified WAIT steps). If the gate fails, writes nothing.
usage: .venv/bin/python rl/gen_wait.py --student M1.pt --agent agents/e082/main.py --opp ... --seed0 260000 --games 200 --out DIR [--seeds 8 --treq 2.0]
DRY速写: --games 4 --seeds 2 (not used by kaggle33 DRY path, which skips Phase 2).
"""
import argparse, copy, glob, importlib.util, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import kagsim
from gen_selfplay import load
from raw import RawAgent, extract_raw

ITEMS = ("WOOL", "MILK", "STRAWBERRY")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--student", required=True); ap.add_argument("--agent", required=True)
    ap.add_argument("--opp", nargs="*", default=None); ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--games", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=8); ap.add_argument("--treq", type=float, default=2.0)
    ap.add_argument("--max-states", type=int, default=60)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    student = RawAgent(a.student); teacher = load(a.agent, "teacher"); opps = [load(p, f"opp{i}") for i, p in enumerate(a.opp or [a.agent])]
    for m in (teacher, *opps):
        for attr in ("_LIVE", "_POLICY", "_ROUTER"):
            if hasattr(m, attr): setattr(m, attr, None)
    tag = lambda m: getattr(m, "agent_entry", m.agent)
    states = []; started = time.time()
    for seed in range(a.seed0, a.seed0 + a.games):
        if len(states) >= a.max_states: break
        seat = seed % 2; opp = opps[(seed // 2) % len(opps)]
        g = kagsim.Game(seed); prev_px = {}
        while not g.done and len(states) < a.max_states:
            t = g.step_count; o = g.observe(seat)
            if 2 <= t < 710 and t % 4 == 1:
                for it in ITEMS:
                    px = o["market"]["prices"][it]; shed = o["private"]["shed"].get(it, 0)
                    if prev_px.get(it) and prev_px[it] > 0 and (prev_px[it] - px) / prev_px[it] > 0.15 and shed >= 8:
                        ta = tag(teacher)(o)
                        if any(x and x[0] == "SELL" and len(x) > 1 and x[1] == it for x in (ta.get("market") or [])):
                            states.append((seed, seat, t))
                            break
            for it in ITEMS: prev_px[it] = o["market"]["prices"][it]
            x = tag(teacher)(o); y = tag(opp)(g.observe(1 - seat)); g.step(*((x, y) if seat == 0 else (y, x)))
    print(f"triggers: {len(states)} [{time.time() - started:.0f}s]", flush=True)
    diffs = []; games = []
    for seed, seat, t0 in states:
        opp = opps[(seed // 2) % len(opps)]
        res = {}
        for mode in ("sell", "wait"):
            g = kagsim.Game(seed); hist = []
            while not g.done:
                t = g.step_count; o = g.observe(seat)
                x = tag(teacher)(o)
                if mode == "wait" and t == t0:
                    x = dict(x, market=[z for z in (x.get("market") or []) if not (z and z[0] == "SELL" and len(z) > 1 and z[1] in ITEMS)])
                    hist.append((t, copy.deepcopy(x)))
                y = tag(opp)(g.observe(1 - seat)); g.step(*((x, y) if seat == 0 else (y, x)))
            res[mode] = g.reward(seat) - g.reward(1 - seat)
        diffs.append(res["wait"] - res["sell"]); games.append((seed, seat, t0, res["wait"] - res["sell"]))
    n = len(diffs)
    if n == 0: print("no triggers; writing nothing", flush=True); return
    m = sum(diffs) / n; se = (sum((d - m) ** 2 for d in diffs) / max(1, n - 1)) ** 0.5 / (n ** 0.5)
    print(f"WAIT-sell paired {m:+.0f}±{se:.0f} (t={m / se if se else 0:+.1f}, n={n})", flush=True)
    if n < 10 or (m / se if se else 0) < a.treq: print(f"gate t>{a.treq} not met; writing nothing", flush=True); return
    # rewrite the positive games with WAIT labels at their trigger steps
    wrote = 0
    for seed, seat, t0, d in games:
        if d <= 0: continue
        opp = opps[(seed // 2) % len(opps)]
        g = kagsim.Game(seed); steps = [[{"observation": g.observe(s), "action": None} for s in (0, 1)]]
        while not g.done:
            t = g.step_count; o = g.observe(seat)
            x = tag(teacher)(o); label = copy.deepcopy(x)
            if t == t0:
                x = dict(x, market=[z for z in (x.get("market") or []) if not (z and z[0] == "SELL" and len(z) > 1 and z[1] in ITEMS)])
                label = copy.deepcopy(x)
            y = tag(opp)(g.observe(1 - seat)); acts = [None, None]; acts[seat] = x; acts[1 - seat] = y
            g.step(*acts)
            rec = [None, None]; rec[seat] = label; rec[1 - seat] = copy.deepcopy(y)
            steps.append([{"observation": g.observe(s), "action": rec[s]} for s in (0, 1)])
        np.savez_compressed(os.path.join(a.out, f"wait_{seed}_s{seat}.npz"), **extract_raw(steps, seat)); wrote += 1
    print(f"wrote {wrote} WAIT shards [{time.time() - started:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
