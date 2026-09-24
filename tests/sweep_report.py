"""route_sweep の結果をペアごとに集計する。既定 (None) との差・勝ち数・上位ルートを出す。

  .venv/bin/python tests/sweep_report.py tmp/sweep_tune.json [--top 3] [--json tmp/cands.json]
"""
import argparse, json, math
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+"); ap.add_argument("--top", type=int, default=3); ap.add_argument("--json")
    a = ap.parse_args()
    res = [x for f in a.files for x in json.load(open(f))]
    g = defaultdict(lambda: defaultdict(list))
    for pair, route, seed, seat, m, own in res:
        g[tuple(pair)][route].append(m)
    out = {}
    tot_gain = 0
    for pair in sorted(g):
        rs = g[pair]
        base = rs.get(None, [])
        bm = sum(base) / len(base) if base else 0.0
        stats = []
        for r, v in rs.items():
            if r is None: continue
            n = len(v); mu = sum(v) / n; sd = math.sqrt(sum((x - mu) ** 2 for x in v) / max(1, n - 1))
            stats.append((mu, r, n, sum(x > 0 for x in v), sd / math.sqrt(n)))
        stats.sort(reverse=True)
        best = stats[:a.top]
        tot_gain += max(0.0, best[0][0] - bm) if best else 0
        out["+".join(pair)] = [dict(route=r, mean=mu, n=n, wins=w, se=se) for mu, r, n, w, se in best]
        print(f"{pair[0][:10]:10s}+{pair[1][:10]:10s} base {bm:+7.0f} | " + "  ".join(f"r{r}:{mu:+6.0f}±{se:4.0f} ({w}/{n})" for mu, r, n, w, se in best))
    print(f"sum of best-vs-base over pairs / 64 = {tot_gain / 64:+.0f} per game (optimistic, in-sample)")
    if a.json: json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
