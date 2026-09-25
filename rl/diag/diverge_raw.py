"""How long does a raw-BC clone stay on its teacher's trajectory? For each (seed, seat) play clone-vs-OPP and TEACHER-vs-OPP
side by side (same seed, both games advance in lockstep) and report the first step at which our farm or the market differs.
usage (repo root): [HANDOFF=1] .venv/bin/python rl/diag/diverge_raw.py CKPT TEACHER OPP [games]
HANDOFF=1 bounds what DART can buy: the clone plays until it first leaves the teacher, then the teacher recovers."""
import sys, os, importlib.util
import numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), "rl"))
import kagsim
from raw import RawAgent


def load(p, tag):
    sys.path.insert(0, os.path.dirname(os.path.abspath(p)))
    s = importlib.util.spec_from_file_location(tag, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def state(o, seat):
    f = o["farms"][seat]
    return (f["money"], f["farmer"], f["hands"], f["tiles"], o["market"]["prices"], o["private"]["shed"], o["private"]["seeds"])


def main():
    ck, teacher_path, opp_path = sys.argv[1:4]; games = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    handoff = os.environ.get("HANDOFF") == "1"   # after the first divergence our seat is played by a shadow teacher that saw the same history
    clone = RawAgent(ck); teacher = load(teacher_path, "teacher"); o1, o2 = load(opp_path, "opp1"), load(opp_path, "opp2")
    shadow = load(teacher_path, "shadow")
    firsts = []; rows = []
    for gi in range(games):
        seed, seat = 5000 + gi // 2, gi % 2
        a, b = kagsim.Game(seed), kagsim.Game(seed); first = None
        while not a.done:
            if first is None and state(a.observe(seat), seat) != state(b.observe(seat), seat): first = a.step_count
            sx = getattr(shadow, "agent_entry", shadow.agent)(a.observe(seat))
            x = sx if (handoff and first is not None) else clone.act(a.observe(seat), seat); y = o1.agent(a.observe(1 - seat))
            a.step(*((x, y) if seat == 0 else (y, x)))
            x = getattr(teacher, "agent_entry", teacher.agent)(b.observe(seat)); y = o2.agent(b.observe(1 - seat))
            b.step(*((x, y) if seat == 0 else (y, x)))
        firsts.append(first if first is not None else 720)
        rows.append((a.reward(seat) - a.reward(1 - seat), b.reward(seat) - b.reward(1 - seat)))
        print(f"seed {seed} seat {seat}: first divergence {first}  margin clone {rows[-1][0]:+8.0f}  teacher {rows[-1][1]:+8.0f}", flush=True)
    r = np.array(rows)
    print(f"median first divergence {int(np.median(firsts))} (720 = never) | margin clone {r[:, 0].mean():+.0f}  teacher {r[:, 1].mean():+.0f}")


if __name__ == "__main__":
    main()
