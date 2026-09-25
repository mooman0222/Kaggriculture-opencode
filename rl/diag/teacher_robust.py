"""Can the teacher recover from its own mistakes? (precondition for DART-style noisy data generation)
Teacher vs teacher (mirror) in kagsim; at step T0 replace ONE unit's action of seat 0 by a random move, then let both
teachers play on. A robust teacher loses little; a tape that cannot recover collapses like the BC clone did.
usage (repo root): .venv/bin/python rl/diag/teacher_robust.py AGENT [n]"""
import sys, os, importlib.util, random
import numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), "rl"))
import kagsim


def load(p, tag):
    sys.path.insert(0, os.path.dirname(os.path.abspath(p)))
    s = importlib.util.spec_from_file_location(tag, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def main():
    path = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    A, B = load(path, "a"), load(path, "b"); rng = random.Random(0); rows = []
    for k in range(n):
        seed = 5000 + k; t0 = rng.randrange(24, 480); g = kagsim.Game(seed)
        while not g.done:
            a = getattr(A, "agent_entry", A.agent)(g.observe(0)); b = getattr(B, "agent_entry", B.agent)(g.observe(1))
            if g.step_count == t0:
                units = [a.get("farmer") or ["PASS"], *(a.get("hands") or [])]; i = rng.randrange(len(units))
                units[i] = [rng.choice(["NORTH", "SOUTH", "EAST", "WEST"])]; a = dict(a, farmer=units[0], hands=units[1:])
            g.step(a, b)
        rows.append(g.reward(0) - g.reward(1))
        print(f"seed {seed}: one random move at t={t0} (day {t0 // 24}) -> margin {rows[-1]:+8.0f}", flush=True)
    r = np.array(rows)
    print(f"mean {r.mean():+.0f} ± {r.std(ddof=1) / np.sqrt(len(r)):.0f}, median {np.median(r):+.0f}, worst {r.min():+.0f}")


if __name__ == "__main__":
    main()
