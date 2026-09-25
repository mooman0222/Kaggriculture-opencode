"""リプレイを記録行動のまま kagsim で再計算し、席ごとの品目別売上・支出・家畜・土地を集める (コードは実行しない)。

  .venv/bin/python tests/replay_stats.py --zip tmp/kds/kaggriculture-episodes-2026-09-22.zip --out tmp/stats0922.json
  .venv/bin/python tests/replay_stats.py --glob 'tmp/own0924/episode-*.json' --out tmp/stats_own.json
"""
import argparse, glob, json, zipfile
from multiprocessing import Pool
import kagsim

PASS = {"farmer": ["PASS"], "hands": [], "market": []}
ITEMS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
ZIP = None


def set_zip(z):
    global ZIP
    ZIP = z


def load(src):
    if ZIP:
        with zipfile.ZipFile(ZIP) as z:
            return json.loads(z.read(src))
    return json.load(open(src))


def work(src):
    try:
        r = load(src)
    except Exception:
        return None
    st = r["steps"]
    if len(st) < 720: return None
    names = r["info"]["TeamNames"]; seed = r["info"]["seed"]
    shops = st[-1][0]["observation"]["town"]["unlocked_shops"]
    g = kagsim.Game(seed, 720, shops)
    animals = [[0, 0, 0], [0, 0, 0]]; plants = [dict(), dict()]; hires = [0, 0]; care = [0, 0]; feed = [0, 0]
    for t in range(719):
        a = [st[t + 1][i].get("action") or PASS for i in (0, 1)]
        if t % 24 == 12:
            o = g.observe(0)
            for p in (0, 1):
                hires[p] += o["farms"][p]["hires_today"]
                if t == 24 * 20 + 12:
                    for row in o["farms"][p]["tiles"]:
                        for tile in row:
                            if isinstance(tile, dict) and tile.get("animal"):
                                animals[p][["GOOSE", "COW", "SHEEP"].index(tile["animal"])] += 1
        for p in (0, 1):
            for u in [a[p].get("farmer") or ["PASS"]] + list(a[p].get("hands") or []):
                if u and u[0] == "PLANT" and len(u) > 1: plants[p][u[1]] = plants[p].get(u[1], 0) + 1
                elif u and u[0] == "CARE": care[p] += 1
                elif u and u[0] == "FEED": feed[p] += 1
        g.step(a[0], a[1])
    ok = all(abs(g.reward(p) - (r["rewards"][p] or 0)) < 1e-6 for p in (0, 1))
    out = []
    for p in (0, 1):
        tel = g.telemetry(p)
        out.append(dict(eid=r["info"]["EpisodeId"], team=names[p], opp=names[1 - p], seat=p, money=g.reward(p), opp_money=g.reward(1 - p),
                        ok=ok, shops=shops, rev={k: tel["sell_revenue_items"][k] for k in ITEMS}, units={k: tel["sold_units_items"][k] for k in ITEMS},
                        spend=tel["total_spend"], spend_by=tel.get("spend_by", {}), hire_paid=tel["hire_paid"], discarded=tel["shed_discarded_units"], sell_dead=tel["sell_dead_units"],
                        animals_d20=animals[p], plants=plants[p], hires=hires[p], care=care[p], feed=feed[p]))
    return out


def main():
    global ZIP
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip"); ap.add_argument("--glob"); ap.add_argument("--out", required=True); ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--n", type=int, default=100000)
    a = ap.parse_args()
    if a.zip:
        ZIP = a.zip; srcs = [n for n in zipfile.ZipFile(a.zip).namelist() if n.endswith(".json")]
    else:
        srcs = sorted(glob.glob(a.glob))
    with Pool(a.jobs, initializer=set_zip, initargs=(ZIP,)) as pool:
        rows = [x for res in pool.imap_unordered(work, srcs[:a.n], chunksize=2) if res for x in res]
    json.dump(rows, open(a.out, "w"))
    print(len(rows), "seats;", sum(not r["ok"] for r in rows), "not reproduced")


if __name__ == "__main__":
    main()
