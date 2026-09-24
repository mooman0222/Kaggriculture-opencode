"""記録された試合を記録行動どおりに進めながら、対象席で自前エージェントが「何を打つか」を並べて比較する (教師強制)。

  .venv/bin/python tests/diverge.py --agent agents/e073/main.py --replays 'tmp/top0924/*/episode-*.json' --team TEAM [--out f.json]
  --team を省くと、各リプレイでディレクトリ名のチーム (fetch_top の出力) を対象にする。
出力: 試合ごとに 農場行動の初回不一致 step / 日別不一致数 / 市場注文の不一致数 / 我々のルート / 相手テープとのルート一致。
"""
import argparse, glob, hashlib, importlib.util, json, os, sys, time
from multiprocessing import Pool
import kagsim

PASS = {"farmer": ["PASS"], "hands": [], "market": []}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_M = None


def norm_unit(u):
    u = list(u or ["PASS"])
    if len(u) == 3 and u[2] == 1 and u[0] in ("PICKUP", "DROP", "PLACE"):
        u = u[:2]
    return tuple(tuple(x) if isinstance(x, list) else x for x in u)


def farm(a):
    a = a or PASS
    return (norm_unit(a.get("farmer")),) + tuple(norm_unit(h) for h in (a.get("hands") or []))


def market(a):
    return tuple(tuple(o) for o in ((a or PASS).get("market") or []) if o and not (len(o) >= 3 and o[2] == 0))


def init(p):
    global _M
    ap = os.path.abspath(p)
    assert ap.startswith(os.path.join(ROOT, "agents") + os.sep), "only our own agents"
    sys.path.insert(0, os.path.dirname(ap))
    s = importlib.util.spec_from_file_location("dv_" + hashlib.md5(ap.encode()).hexdigest()[:8], ap)
    _M = importlib.util.module_from_spec(s); s.loader.exec_module(_M)


def work(job):
    f, team = job
    r = json.load(open(f)); st = r["steps"]; names = r["info"]["TeamNames"]
    if team.startswith("OPP:"):
        if team[4:] not in names: return None
        team = names[1 - names.index(team[4:])]
        if team == team[4:]: return None
    if len(st) < 720 or team not in names: return None
    me = names.index(team)
    shops = st[-1][0]["observation"]["town"]["unlocked_shops"]
    rec = [[st[t + 1][i].get("action") or PASS for i in (0, 1)] for t in range(719)]
    for k in ("_LIVE", "_ROUTER", "_POLICY"):
        if hasattr(_M, k): setattr(_M, k, None)
    ag = getattr(_M, "agent_entry", _M.agent)
    g = kagsim.Game(r["info"]["seed"], 720, shops)
    fm = []; mm = []; first = None; route = None; err = 0
    for t in range(719):
        try:
            mine = ag(g.observe(me))
        except Exception:
            mine = PASS; err += 1
        if t == 150:
            try: route = _M._IMPL.chassis.players[me]["route"]
            except Exception: route = None
        fe = farm(mine) == farm(rec[t][me]); me_ = market(mine) == market(rec[t][me])
        fm.append(0 if fe else 1); mm.append(0 if me_ else 1)
        if not fe and first is None: first = t
        g.step(rec[t][0], rec[t][1])
    ok = abs(g.reward(me) - (r["rewards"][me] or 0)) < 1e-6
    return dict(eid=r["info"]["EpisodeId"], team=team, me=me, opp=names[1 - me], shops=shops[:2], route=route,
                first_farm=first, farm_mis_by_day=[sum(fm[d * 24:(d + 1) * 24]) for d in range(30)],
                mkt_mis=sum(mm), mkt_mis_first=next((t for t, x in enumerate(mm) if x), None), replay_ok=ok, err=err,
                rec_margin=(r["rewards"][me] or 0) - (r["rewards"][1 - me] or 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True); ap.add_argument("--replays", required=True); ap.add_argument("--team")
    ap.add_argument("--jobs", type=int, default=10); ap.add_argument("--out")
    ap.add_argument("--opp-of", help="このチームの相手の席を対象にする")
    a = ap.parse_args()
    jobs = []
    for f in sorted(glob.glob(a.replays, recursive=True)):
        team = ("OPP:" + a.opp_of) if a.opp_of else a.team
        if not team:
            tj = os.path.join(os.path.dirname(os.path.dirname(f)), "teams.json")
            d = os.path.dirname(f)
            if os.path.exists(tj):
                team = next((t["name"] for t in json.load(open(tj)) if os.path.abspath(t["dir"]) == os.path.abspath(d)), None)
        if team: jobs.append((f, team))
    t0 = time.time()
    with Pool(a.jobs, initializer=init, initargs=(a.agent,)) as pool:
        rows = [x for x in pool.imap(work, jobs) if x]
    for x in rows:
        print(f"{x['eid']} {x['team'][:16]:16s} s{x['me']} {'/'.join(s[:4] for s in x['shops'])} r={x['route']} first_farm={x['first_farm']} "
              f"mkt_mis={x['mkt_mis']}@{x['mkt_mis_first']} farm_mis/day={''.join(min(9, v) and str(min(9, v)) or '.' for v in x['farm_mis_by_day'])} ok={x['replay_ok']} m={x['rec_margin']:+.0f}")
    print(f"== {len(rows)} games {time.time() - t0:.0f}s")
    if a.out: json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
