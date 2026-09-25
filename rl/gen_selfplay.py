"""Scripted-teacher games in kagsim -> BC shards (same format as extract.py): our own data, unlimited and on demand.
The teacher plays seat seed%2 against --opp (a panel, cycled every 2 seeds so each opponent gets both seats; default =
a second instance of the teacher, i.e. self-play). Only the teacher seat is written, as <out>/<seed>_s<seat>.npz.
usage: .venv/bin/python rl/gen_selfplay.py [--raw] --agent agents/e079/main.py --opp agents/e072/main.py third_party/public_agents/v46/main.py \
           --seed0 300000 --games 500 --out tmp/rl/e079panel"""
import argparse, copy, importlib.util, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import kagsim
from extract import extract_replay
from raw import extract_raw


def load(path, tag):
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec = importlib.util.spec_from_file_location(f"agent_{tag}", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--agent", required=True); ap.add_argument("--opp", nargs="*", default=None)
    ap.add_argument("--seed0", type=int, required=True); ap.add_argument("--games", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--raw", action="store_true", help="write rl/raw.py shards (engine action space) instead of Policy3 labels")
    ap.add_argument("--dry", default="", help="LO,HI: one random day in [LO,HI] per game on which the teacher's WATER is executed as PASS (crops die into weeds; the labels keep WATER and show the cleanup)")
    ap.add_argument("--noise", type=float, default=0.0, help="DART: per step, with this probability one teacher unit executes a random move; the label stays the teacher's action")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed0); teacher = load(a.agent, "teacher"); opps = [load(p, f"opp{i}") for i, p in enumerate(a.opp or [a.agent])]
    names = [os.path.basename(os.path.dirname(p)) for p in (a.opp or [a.agent])]; started = time.time()
    for seed in range(a.seed0, a.seed0 + a.games):
        seat = seed % 2; dst = os.path.join(a.out, f"{seed}_s{seat}.npz")
        if os.path.exists(dst): continue
        k = (seed // 2) % len(opps); players = [None, None]; players[seat] = teacher; players[1 - seat] = opps[k]
        for m in players:
            for attr in ("_LIVE", "_POLICY", "_ROUTER"):
                if hasattr(m, attr): setattr(m, attr, None)
        g = kagsim.Game(seed); dry_day = rng.randint(*map(int, a.dry.split(","))) if a.dry else -1
        steps = [[{"observation": g.observe(s), "action": None} for s in (0, 1)]]
        while not g.done:
            acts = [getattr(players[s], "agent_entry", players[s].agent)(g.observe(s)) for s in (0, 1)]
            labels = [copy.deepcopy(x) for x in acts]
            if a.noise and rng.random() < a.noise:   # perturb what the teacher executes, keep what it intended as the label
                units = [acts[seat].get("farmer") or ["PASS"], *(acts[seat].get("hands") or [])]; i = rng.randrange(len(units))
                units[i] = [rng.choice(["NORTH", "SOUTH", "EAST", "WEST"])]; acts[seat] = dict(acts[seat], farmer=units[0], hands=units[1:])
            if g.step_count // 24 == dry_day:
                acts[seat] = dict(acts[seat], farmer=["PASS"] if (acts[seat].get("farmer") or ["PASS"])[0] == "WATER" else acts[seat].get("farmer"),
                                  hands=[["PASS"] if h and h[0] == "WATER" else h for h in (acts[seat].get("hands") or [])])
            g.step(*acts)
            steps.append([{"observation": g.observe(s), "action": labels[s]} for s in (0, 1)])
        np.savez_compressed(dst, **(extract_raw(steps, seat) if a.raw else extract_replay({"steps": steps, "rewards": [g.reward(0), g.reward(1)]}, seat)))
        print(f"seed {seed} vs {names[k]}: teacher {g.reward(seat):.0f} opp {g.reward(1 - seat):.0f} [{time.time() - started:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
