"""土台 (E087 など) の上に「窓頭 (step%4==1) の追加 SELL」を後知恵で入れたときの伸びしろを測る (実装前の上限測定)。

  対テープ (実戦の記録相手):
    .venv/bin/python tests/overlay_headroom.py agents/e087/main.py --replays 'tmp/e081_late0928_slim/episode-*.json' --team MMN0222 --jobs 4 --out tmp/headroom_tapes.json
  対反応する相手:
    .venv/bin/python tests/overlay_headroom.py agents/e087/main.py --vs agents/pub_me4/main.py --seeds 16 --seed0 5000 --jobs 4 --out tmp/headroom_me4.json

候補 (--family add) = (日 d, 品目) ごとに「d 日の各窓頭で品目の projected shed を全部追加売り (既存 SELL があれば数量を足す)」。
候補 (--family hold) = (日 d, 品目) ごとに「d 日の土台の SELL 注文からその品目を全部落とす (売らずに持ち越す)」。
日順・品目順に貪欲に採否 (最終 margin が改善したら残す)。参考に固定方策 (dump = E081 型の全窓全品ダンプ、cap6 = STRAWBERRY のみ 6 に上限) も出す。
09-28 の実測: me4 系は窓頭に価値ある在庫を残さない (残るのは $1 の牛乳) ので add 族の上限は 0。
後知恵の上限なので、学習方策の実効値はこれより小さい。agents/ 配下のみ実行する。
"""
import argparse, glob, hashlib, importlib.util, json, math, os, sys, time
from multiprocessing import Pool
import kagsim

PASS = {"farmer": ["PASS"], "hands": [], "market": []}
ITEMS = ("MILK", "WOOL", "STRAWBERRY")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = {}


def own(p):
    ap = os.path.abspath(p)
    if not ap.startswith(os.path.join(ROOT, "agents") + os.sep):
        raise SystemExit(f"refuse: {p} is not under agents/")
    return ap


def load(p):
    sys.path.insert(0, os.path.dirname(p))
    s = importlib.util.spec_from_file_location("hr_" + hashlib.md5(p.encode()).hexdigest()[:8], p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def qty(o):
    return max(0, int(o[2])) if len(o) >= 3 else 1


def make_agent(m, plan, mode, family="add"):
    """plan: set of (day, item)。family add → その日の窓頭で item を全部追加売り。family hold → その日の土台 SELL から item を落とす。mode: None / 'dump' / 'cap6'"""
    base = getattr(m, "agent_entry", m.agent)

    def ag(obs):
        action = base(obs)
        step = int(obs["step"])
        day = step // 24
        if family == "hold" and not mode:
            hold = {it for it in ITEMS if (day, it) in plan}
            if not hold:
                return action
            market = [list(o) for o in (action.get("market") or []) if not (o and o[0] == "SELL" and len(o) >= 2 and o[1] in hold)]
            return dict(action, market=market)
        if step < 2 or step >= 710 or step % 4 != 1:
            return action
        items = [it for it in ITEMS if mode or (day, it) in plan]
        if not items:
            return action
        market = [list(o) for o in (action.get("market") or [])]
        stock = m.projected_shed(action, m.FarmView(obs))
        for o in market:
            if o and o[0] == "SELL" and len(o) >= 2:
                stock[o[1]] = stock.get(o[1], 0) - qty(o)
        changed = False
        for it in items:
            n = stock.get(it, 0)
            if mode == "cap6" and it == "STRAWBERRY":
                n = min(n, 6)
            if n <= 0:
                continue
            same = next((o for o in market if o and o[0] == "SELL" and len(o) >= 3 and o[1] == it), None)
            if same is not None:
                same[2] = qty(same) + n
            elif len(market) < 10:
                market.append(["SELL", it, n])
            else:
                continue
            changed = True
        return dict(action, market=market) if changed else action
    return ag


def init(base_path, vs_path, family, opts=None):
    G.update(opts or {})
    G["m"] = load(base_path)
    G["vs"] = load(vs_path) if vs_path else None
    G["family"] = family


def play(ep, plan, mode):
    m = G["m"]
    for k in ("_LIVE", "_ROUTER", "_POLICY"):
        if hasattr(m, k): setattr(m, k, None)
    ag = make_agent(m, plan, mode, G["family"]); me = ep["me"]
    if ep.get("tape") is not None:
        g = kagsim.Game(ep["seed"], 720, ep["shops"])
        while not g.done:
            t = g.step_count; mine = ag(g.observe(me))
            g.step(*((mine, ep["tape"][t]) if me == 0 else (ep["tape"][t], mine)))
    else:
        vm = G["vs"]
        for k in ("_LIVE", "_ROUTER", "_POLICY"):
            if hasattr(vm, k): setattr(vm, k, None)
        opp = getattr(vm, "agent_entry", vm.agent)
        g = kagsim.Game(ep["seed"])
        while not g.done:
            mine = ag(g.observe(me)); theirs = opp(g.observe(1 - me))
            g.step(*((mine, theirs) if me == 0 else (theirs, mine)))
    return g.reward(me) - g.reward(1 - me)


ANIMALS = ("COW", "SHEEP", "GOOSE")


def features(ep, plan):
    """最終 plan で 1 局を再生し、各日の朝 (step 24d) の特徴量と hold ラベルを返す (学習用)。"""
    m = G["m"]
    for k in ("_LIVE", "_ROUTER", "_POLICY"):
        if hasattr(m, k): setattr(m, k, None)
    ag = make_agent(m, plan, None, G["family"]); me = ep["me"]
    vm = G["vs"]; opp = None
    if ep.get("tape") is None:
        for k in ("_LIVE", "_ROUTER", "_POLICY"):
            if hasattr(vm, k): setattr(vm, k, None)
        opp = getattr(vm, "agent_entry", vm.agent)
    g = kagsim.Game(ep["seed"], 720, ep["shops"]) if ep.get("tape") is not None else kagsim.Game(ep["seed"])
    px, inv, sold = {}, {}, {}
    rows = []
    while not g.done:
        t = g.step_count; obs = g.observe(me)
        prices = obs["market"]["prices"]; inventory = obs["market"]["inventory"]
        px[t] = {it: float(prices.get(it, 0) or 0) for it in ITEMS}; inv[t] = {it: float(inventory.get(it, 0) or 0) for it in ITEMS}
        if t % 24 == 0 and t >= 48:
            d = t // 24; farm = obs["farms"][me]; shed = obs["private"]["shed"]
            an = {a: 0 for a in ANIMALS}
            for row in farm["tiles"]:
                for tile in row:
                    if isinstance(tile, dict) and tile.get("animal") in an: an[tile["animal"]] += 1
            f = dict(day=d, money=float(farm["money"]), cows=an["COW"], sheep=an["SHEEP"], geese=an["GOOSE"])
            y0, y1 = t - 24, t - 1
            for it in ITEMS:
                hist = [px[u][it] for u in range(y0, t)]
                f[f"px_{it}"] = px[t][it]; f[f"pxmax_{it}"] = max(hist); f[f"pxmin_{it}"] = min(hist); f[f"px24_{it}"] = px[y0][it]
                f[f"inv_{it}"] = inv[t][it]; f[f"dinv_{it}"] = inv[t][it] - inv[y0][it]
                f[f"shed_{it}"] = float(shed.get(it, 0)); f[f"sold_{it}"] = float(sum(sold.get(u, {}).get(it, 0) for u in range(y0, t)))
                f[f"hold_{it}"] = int((d, it) in plan)
            rows.append(f)
        mine = ag(obs)
        sold[t] = {}
        for o in (mine.get("market") or []):
            if o and o[0] == "SELL" and len(o) >= 2 and o[1] in ITEMS: sold[t][o[1]] = sold[t].get(o[1], 0) + qty(o)
        if opp is None:
            g.step(*((mine, ep["tape"][t]) if me == 0 else (ep["tape"][t], mine)))
        else:
            theirs = opp(g.observe(1 - me)); g.step(*((mine, theirs) if me == 0 else (theirs, mine)))
    return rows


def run(ep):
    t0 = time.time()
    if "plan" in ep:  # --replan: 既存の plan で特徴量だけ記録する
        plan = {tuple(x) for x in ep["plan"]}
        return dict(eid=ep["eid"], me=ep["me"], opp=ep.get("opp", ""), seed=ep["seed"], base=ep["base"], dump=ep["base"], cap6=ep["base"],
                    greedy=ep["greedy"], plan=sorted(plan), secs=round(time.time() - t0), feat=features(ep, plan))
    base = play(ep, set(), None)
    dump = play(ep, set(), "dump") if G.get("static") else base
    cap6 = play(ep, set(), "cap6") if G.get("static") else base
    plan, best = set(), base
    for day in range(G.get("d0", 3), G.get("d1", 30)):
        for it in ITEMS:
            cand = plan | {(day, it)}
            v = play(ep, cand, None)
            if v > best + 1:
                plan, best = cand, v
    out = dict(eid=ep["eid"], me=ep["me"], opp=ep.get("opp", ""), seed=ep["seed"], base=base, dump=dump, cap6=cap6, greedy=best,
               plan=sorted(plan), secs=round(time.time() - t0))
    if G.get("feat"):
        out["feat"] = features(ep, plan)
    return out


def episodes_from_replays(pattern, team):
    out, seen = [], set()
    for f in sorted(glob.glob(pattern, recursive=True)):
        r = json.load(open(f)); st = r["steps"]; names = r["info"]["TeamNames"]
        if len(st) < 720 or (team and team not in names) or r["info"]["EpisodeId"] in seen:
            continue
        seen.add(r["info"]["EpisodeId"])
        shops = st[-1][0]["observation"]["town"]["unlocked_shops"]
        for me in ([names.index(team)] if team else [0, 1]):
            tape = [st[t + 1][1 - me].get("action") or PASS for t in range(719)]
            out.append(dict(eid=r["info"]["EpisodeId"], me=me, seed=r["info"]["seed"], shops=shops, tape=tape, opp=names[1 - me]))
    return out


def summarize(rows):
    n = len(rows)
    print(f"== {n} games, {sum(r['secs'] for r in rows) / n:.0f}s/game")
    for k in ("dump", "cap6", "greedy"):
        d = [r[k] - r["base"] for r in rows]; md = sum(d) / n
        se = math.sqrt(sum((x - md) ** 2 for x in d) / max(1, n - 1) / n)
        print(f"  {k:7s} vs base: {md:+7.0f} ± {se:4.0f} (t={md / se if se else 0:+.1f})  better/worse {sum(x > 0 for x in d)}/{sum(x < 0 for x in d)}"
              f"  W {sum(r['base'] > 0 for r in rows)} -> {sum(r[k] > 0 for r in rows)}")
    from collections import Counter
    c = Counter(it for r in rows for _, it in r["plan"]); cd = Counter(d for r in rows for d, _ in r["plan"])
    print("  greedy accepted (item):", dict(c), " accepted/game:", round(sum(len(r['plan']) for r in rows) / n, 1))
    print("  greedy accepted (day):", sorted(cd.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("--replays"); ap.add_argument("--team"); ap.add_argument("--vs")
    ap.add_argument("--seeds", type=int, default=16); ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--n", type=int, default=100000); ap.add_argument("--out")
    ap.add_argument("--family", choices=["add", "hold"], default="hold"); ap.add_argument("--static", action="store_true", help="dump/cap6 の固定方策も測る")
    ap.add_argument("--feat", action="store_true", help="最終 plan を再生して日ごとの特徴量 + hold ラベルを out に含める (学習用)")
    ap.add_argument("--days", default="3-30", help="候補にする日の範囲 (半開区間)")
    ap.add_argument("--replan", help="既存の out JSON の plan を使い、貪欲探索なしで特徴量だけ記録する")
    a = ap.parse_args()
    base = own(a.base); vs = own(a.vs) if a.vs else None
    if a.replays:
        eps = episodes_from_replays(a.replays, a.team)
    else:
        eps = [dict(eid=s, me=me, seed=s, opp=os.path.basename(os.path.dirname(vs))) for s in range(a.seed0, a.seed0 + a.seeds) for me in (0, 1)]
    eps = eps[:a.n]
    if a.replan:
        prev = {(r["eid"], r["me"]): r for r in json.load(open(a.replan))}
        eps = [dict(e, plan=prev[(e["eid"], e["me"])]["plan"], base=prev[(e["eid"], e["me"])]["base"], greedy=prev[(e["eid"], e["me"])]["greedy"]) for e in eps if (e["eid"], e["me"]) in prev]
        G["feat"] = True
    rows = []
    G["static"] = a.static; G["feat"] = a.feat; G["d0"], G["d1"] = map(int, a.days.split("-"))
    with Pool(a.jobs, initializer=init, initargs=(base, vs, a.family, dict(G)), ) as pool:
        for r in pool.imap_unordered(run, eps, chunksize=1):
            rows.append(r)
            print(f"{r['eid']} s{r['me']} {str(r['opp'])[:16]:16s} base {r['base']:+7.0f} dump {r['dump'] - r['base']:+6.0f} cap6 {r['cap6'] - r['base']:+6.0f} greedy {r['greedy'] - r['base']:+6.0f} plan {len(r['plan'])} [{r['secs']}s]", flush=True)
    summarize(rows)
    if a.out:
        json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
