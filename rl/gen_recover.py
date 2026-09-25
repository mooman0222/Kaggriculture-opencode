"""Recovery data on the student's own mistakes (DAgger-style, 2026-09-24).
The student (raw BC) plays; a shadow teacher sees the same observations and supplies the label at every step. When the
student's action differs from the teacher's, the student's action IS executed (the state leaves the teacher's trajectory
the way the student actually errs), then the teacher plays TAKEOVER steps to recover, then the student resumes.
Measured beforehand (rl/diag/diverge_raw.py HANDOFF=1): letting the teacher take over after the clone's first deviation
brought margin -61k back to -844, so the collapse is the student's failure to continue, not the deviation itself.
usage: .venv/bin/python rl/gen_recover.py --student tmp/rl/raw_v41_ep3.pt --agent third_party/public_agents/v41/main.py \
           --seed0 220000 --games 250 --out tmp/rl/recover"""
import argparse, copy, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import kagsim
from gen_selfplay import load
from raw import RawAgent, encode_raw, extract_raw


def same(a, b, n):
    """Equal as the engine sees them, ignoring trailing empty/absent market slots."""
    ea, eb = encode_raw(a, n), encode_raw(b, n)
    if not (np.array_equal(ea[0], eb[0]) and np.array_equal(ea[1], eb[1])): return False
    trim = lambda mk, mq: [(int(k), int(q)) for k, q in zip(mk, mq)]
    ma, mb = trim(ea[2], ea[3]), trim(eb[2], eb[3])
    while ma and ma[-1][0] <= 1: ma.pop()   # END / NOP at the tail do nothing
    while mb and mb[-1][0] <= 1: mb.pop()
    return ma == mb


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--student", required=True); ap.add_argument("--agent", required=True)
    ap.add_argument("--opp", nargs="*", default=None); ap.add_argument("--seed0", type=int, required=True); ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--takeover", type=int, default=24)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    student = RawAgent(a.student); teacher = load(a.agent, "teacher"); opps = [load(p, f"opp{i}") for i, p in enumerate(a.opp or [a.agent])]
    started = time.time()
    for seed in range(a.seed0, a.seed0 + a.games):
        seat = seed % 2; dst = os.path.join(a.out, f"{seed}_s{seat}.npz")
        if os.path.exists(dst): continue
        opp = opps[(seed // 2) % len(opps)]
        for m in (teacher, opp):
            for attr in ("_LIVE", "_POLICY", "_ROUTER"):
                if hasattr(m, attr): setattr(m, attr, None)
        g = kagsim.Game(seed); steps = [[{"observation": g.observe(s), "action": None} for s in (0, 1)]]
        teacher_until = 0; deviations = 0
        while not g.done:
            t = g.step_count; o = g.observe(seat); n = 1 + len(o["farms"][seat]["hands"])
            label = getattr(teacher, "agent_entry", teacher.agent)(g.observe(seat))
            executed = label
            if t >= teacher_until:
                mine = student.act(o, seat)
                if not same(mine, label, n):
                    executed = mine; teacher_until = t + 1 + a.takeover; deviations += 1
            other = getattr(opp, "agent_entry", opp.agent)(g.observe(1 - seat))
            acts = [None, None]; acts[seat] = executed; acts[1 - seat] = other
            g.step(*acts)
            rec = [None, None]; rec[seat] = copy.deepcopy(label); rec[1 - seat] = copy.deepcopy(other)
            steps.append([{"observation": g.observe(s), "action": rec[s]} for s in (0, 1)])
        np.savez_compressed(dst, **extract_raw(steps, seat))
        print(f"seed {seed}: student deviations {deviations}  final {g.reward(seat):.0f} / {g.reward(1 - seat):.0f} [{time.time() - started:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
