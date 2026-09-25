"""Closed-loop cost of the BC market representation, no learning involved: a teacher plays normally, but its market
orders are passed through actions.encode_action -> decode_action (per-item sums, MKT_BUCKETS rounding, fixed order,
10-order cap) before execution. The teacher keeps reacting to the real state. Opponent = unmodified copy of the teacher.
MODE=exact keeps exact quantities (only aggregation + reordering); MODE=bucket is the BC representation.
usage (repo root): MODE=bucket .venv/bin/python rl/diag/roundtrip.py AGENT_PATH [games]"""
import sys, os, importlib.util
import numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), "rl"))
import kagsim
from features import MAX_UNITS, PRODUCTS, CROPS, ANIMALS
from actions import encode_action, decode_action, BUY_PRODUCTS

MODE = os.environ.get("MODE", "bucket")


def load(p, tag):
    sys.path.insert(0, os.path.dirname(os.path.abspath(p)))
    s = importlib.util.spec_from_file_location(tag, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def exact(market, obs):
    """Aggregation + BC ordering (sells by value, buys, seeds, animals, hires, land), exact quantities, 10-order cap."""
    sums = {}; hires = 0; land = 0
    for o in market or []:
        if not o: continue
        if o[0] == "HIRE": hires += 1
        elif o[0] == "BUY_LAND": land = 1
        elif len(o) > 2: sums[(o[0], o[1])] = sums.get((o[0], o[1]), 0) + int(o[2])
    prices = obs["market"]["prices"]
    sells = sorted([(prices.get(p, 0) * q, ["SELL", p, q]) for (k, p), q in sums.items() if k == "SELL" and q > 0], key=lambda t: -t[0])
    out = [o for _, o in sells]
    for kind in ("BUY_PRODUCT", "BUY_SEED", "BUY_ANIMAL"):
        out += [[kind, p, q] for (k, p), q in sums.items() if k == kind and q > 0]
    out += [["HIRE"]] * hires + ([["BUY_LAND"]] if land else [])
    return out[:10]


def main():
    path = sys.argv[1]; games = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    me, opp = load(path, "me"), load(path, "opp"); rows = []
    for gi in range(games):
        seed, seat = 5000 + gi // 2, gi % 2
        g = kagsim.Game(seed)
        while not g.done:
            o = g.observe(seat); a = dict(me.agent(o))
            if MODE == "bucket":
                n = min(MAX_UNITS, 1 + len(o["farms"][seat]["hands"]))
                a["market"] = decode_action(np.zeros(MAX_UNITS, int), np.zeros(MAX_UNITS, int), encode_action(a, n)[2], o, seat)["market"]
            elif MODE == "exact":
                a["market"] = exact(a.get("market"), o)
            b = opp.agent(g.observe(1 - seat))
            g.step(*((a, b) if seat == 0 else (b, a)))
        rows.append((g.reward(seat), g.reward(seat) - g.reward(1 - seat)))
    r = np.array(rows)
    print(f"{MODE:6s} {os.path.basename(os.path.dirname(path))}: own {r[:, 0].mean():8.0f}  margin {r[:, 1].mean():+8.0f} ± {r[:, 1].std(ddof=1) / np.sqrt(len(r)):6.0f}  wins {(r[:, 1] > 0).sum()}/{len(r)}")


if __name__ == "__main__":
    main()
