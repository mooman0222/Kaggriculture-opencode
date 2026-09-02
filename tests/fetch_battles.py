"""実戦リプレイを相手クラス別に集計する (E024/E029)。

使い方:
  kaggle competitions episodes <ref> | awk ... > ids.txt; xargs kaggle competitions replay -p DIR
  .venv/bin/python tests/fetch_battles.py --dir DIR --team MMN0222 [--lb leaderboard.csv]
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sig24(steps, pi):
    """motemen 方式: 最初24ステップの行動列の SHA-256 先頭10桁 = 系統署名。"""
    acts = [json.dumps(steps[i][pi].get("action"), sort_keys=True) for i in range(1, 25)]
    return hashlib.sha256("\n".join(acts).encode()).hexdigest()[:10]


def animals(farm):
    c = defaultdict(int)
    for row in farm["tiles"]:
        for t in row:
            if isinstance(t, dict) and "animal" in t:
                c[t["animal"][0]] += 1
    return "C%dS%dG%dK%d" % (c["C"], c["S"], c["G"], c["H"] + c["D"]) if c else "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--team", default="MMN0222")
    ap.add_argument("--lb")
    a = ap.parse_args()
    lb = {}
    if a.lb:
        for row in csv.reader(open(a.lb)):
            if len(row) > 4 and row[4].replace(".", "").isdigit():
                lb[row[2]] = float(row[4])

    rows = []
    for f in sorted(Path(a.dir).glob("*.json")):
        r = json.load(open(f))
        names = r["info"]["TeamNames"]
        if a.team not in names:
            continue
        me = names.index(a.team)
        op = 1 - me
        steps = r["steps"]
        last = steps[-1][0]["observation"]
        rew = r["rewards"]
        rows.append({
            "ep": r["info"]["EpisodeId"], "opp": names[op], "opp_lb": lb.get(names[op]),
            "seat": me, "me": rew[me], "op": rew[op], "margin": rew[me] - rew[op],
            "opp_sig": sig24(steps, op), "my_sig": sig24(steps, me),
            "opp_cls": animals(last["farms"][op]), "my_cls": animals(last["farms"][me]),
            "shop": (last.get("town") or {}).get("shops", [{}])[0].get("type") if isinstance(last.get("town"), dict) else None,
        })

    rows.sort(key=lambda x: x["margin"])
    w = sum(1 for x in rows if x["margin"] > 0)
    print(f"{len(rows)} games: {w}W {len(rows)-w}L, avg me ${sum(x['me'] for x in rows)/len(rows):,.0f} "
          f"op ${sum(x['op'] for x in rows)/len(rows):,.0f}")
    print("\n== losses ==")
    for x in rows:
        if x["margin"] < 0:
            print(f"  {x['ep']} seat{x['seat']} {x['margin']:>9,.0f}  me ${x['me']:>8,.0f} op ${x['op']:>8,.0f}  "
                  f"{x['opp']:<22} lb={x['opp_lb']}  sig={x['opp_sig']} cls={x['opp_cls']} (mine {x['my_cls']})")

    def group(key):
        g = defaultdict(list)
        for x in rows:
            g[x[key]].append(x)
        print(f"\n== by {key} ==")
        for k, xs in sorted(g.items(), key=lambda kv: -len(kv[1])):
            ww = sum(1 for x in xs if x["margin"] > 0)
            lbs = [x["opp_lb"] for x in xs if x["opp_lb"]]
            print(f"  {str(k):<12} n={len(xs):>3} {ww}W{len(xs)-ww}L  avg margin {sum(x['margin'] for x in xs)/len(xs):>9,.0f}"
                  f"  opp lb avg {sum(lbs)/len(lbs) if lbs else 0:.0f}  e.g. {', '.join(sorted({x['opp'] for x in xs})[:4])}")
    group("opp_sig")
    group("opp_cls")
    json.dump(rows, open(Path(a.dir) / "battles.json", "w"), indent=1)


if __name__ == "__main__":
    main()
