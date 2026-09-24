"""提出の強さを相手の対戦時レートから最尤推定する (Elo 尺度 400、引分は除外)。

  .venv/bin/python tests/fit_strength.py tmp/traj0924/<sid>_traj.json ... [--min-opp 2000] [--since 2026-09-23T14:30]
"""
import argparse, json, math


def fit(rows):
    lo, hi = 0.0, 4000.0
    for _ in range(60):
        R = (lo + hi) / 2
        g = sum((1 if w else 0) - 1 / (1 + 10 ** ((o - R) / 400)) for o, w in rows)
        lo, hi = (R, hi) if g > 0 else (lo, R)
    R = (lo + hi) / 2
    info = sum((math.log(10) / 400) ** 2 * p * (1 - p) for o, _ in rows for p in [1 / (1 + 10 ** ((o - R) / 400))])
    return R, 1 / math.sqrt(info) if info > 0 else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trajs", nargs="+"); ap.add_argument("--min-opp", type=float, default=2000); ap.add_argument("--since", default="")
    ap.add_argument("--until", default="9999")
    a = ap.parse_args()
    for f in a.trajs:
        rows = [(r["oppr"], r["m"] > 0) for r in json.load(open(f))
                if r["oppr"] and r["oppr"] >= a.min_opp and r["m"] != 0 and a.since <= r["t"] < a.until]
        if not rows: print(f, "no rows"); continue
        R, se = fit(rows)
        print(f"{f.split('/')[-1][:8]}  n={len(rows):4d}  W={sum(w for _, w in rows):4d}  fit {R:7.1f} ± {se:4.0f}")


if __name__ == "__main__":
    main()
