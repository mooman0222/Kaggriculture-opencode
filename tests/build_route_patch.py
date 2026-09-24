"""検証済みの掃引結果からペア別ルート表を選び、agents/<dst>/main.py = <src>/main.py + 表の追記 を作る。

  .venv/bin/python tests/build_route_patch.py --val tmp/sweep_val.json --src agents/e076 --dst agents/e077 [--min-mean 300 --min-wins 0.67]
採用条件: 同じ seed・席の既定ルート (None) とのペア差の平均が min-mean 超、ペア差が正の割合が min-wins 以上。
"""
import argparse, json, os, shutil
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", nargs="+", required=True); ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--min-mean", type=float, default=300); ap.add_argument("--min-wins", type=float, default=0.67)
    ap.add_argument("--tag", default="E077")
    a = ap.parse_args()
    # 同じ seed・席の既定ルート (None) との差で比べる (席の非対称や世界の当たり外れを相殺する)
    g = defaultdict(lambda: defaultdict(dict))
    for f in a.val:
        for pair, route, seed, seat, m, own in json.load(open(f)):
            g[tuple(pair)][route][(seed, seat)] = m
    table = {}
    for pair, rs in sorted(g.items()):
        base = rs.get(None, {}); bm = sum(base.values()) / max(1, len(base))
        best = None
        for r, v in rs.items():
            if r is None: continue
            d = [v[k] - base[k] for k in v if k in base]
            if not d: continue
            mu = sum(d) / len(d); wr = sum(x > 0 for x in d) / len(d)
            if mu > a.min_mean and wr >= a.min_wins and (best is None or mu > best[0]):
                best = (mu, r, wr, len(d))
        if best:
            table[pair] = best[1]
            print(f"{pair[0]:15s}+{pair[1]:15s} base {bm:+7.0f} -> r{best[1]} paired {best[0]:+7.0f} wr {best[2]:.2f} n={best[3]}")
    print(len(table), "pairs patched")
    os.makedirs(a.dst, exist_ok=True)
    for f in ("LICENSE.txt", "NOTICE.txt"):
        shutil.copy(os.path.join(a.src, f), os.path.join(a.dst, f))
    src = open(os.path.join(a.src, "main.py")).read()
    block = f"""

# ---------------------------------------------------------------------------
# {a.tag} (MMN0222): anti-mirror route table. Two thirds of ladder opponents replay
# our exact farm with the same route table, so in those games both sides grow the
# same goods into the same saturated markets. For each first-two-shop pair, the
# route below beat the mirror (default table, same chassis) on held-out worlds.
_{a.tag}_PATCH = {json.dumps({f"{k[0]}+{k[1]}": v for k, v in table.items()})}
_E074_PATCH.update({{tuple(k.split('+')): v for k, v in _{a.tag}_PATCH.items()}})
globals().pop('agent', None)
agent = e076_agent
"""
    open(os.path.join(a.dst, "main.py"), "w").write(src + block)


if __name__ == "__main__":
    main()
