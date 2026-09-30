"""スリム化リプレイから指定チームの席のテープを集め、テープ切替エージェント用のライブラリ JSON を作る。
  python tests/tr/build_lib.py 'Fourth Quadrant' 'tmp/fq_slim/*.json' tmp/tr/lib_fq.json
"""
import glob, json, sys
team, pattern, out = sys.argv[1], sys.argv[2], sys.argv[3]
allow = set(int(x) for x in open(sys.argv[4]).read().split()) if len(sys.argv) > 4 else None
acts, index, tapes = [], {}, []
for f in sorted(glob.glob(pattern)):
    r = json.load(open(f)); st = r["steps"]; names = r["info"]["TeamNames"]
    if team not in names or len(st) < 720 or (allow is not None and r["info"]["EpisodeId"] not in allow):
        continue
    s = names.index(team)
    ids = []
    for t in range(719):
        a = st[t + 1][s].get("action") or {"farmer": ["PASS"], "hands": [], "market": []}
        k = json.dumps(a, separators=(",", ":"), sort_keys=True)
        if k not in index:
            index[k] = len(acts); acts.append(a)
        ids.append(index[k])
    shops = st[-1][0]["observation"]["town"]["unlocked_shops"]
    tapes.append(dict(eid=r["info"]["EpisodeId"], s=s, shops=shops, own=r["rewards"][s], opp=r["rewards"][1 - s],
                      seed=r["info"]["seed"], opp_team=names[1 - s], ids=ids))
json.dump(dict(actions=acts, tapes=tapes), open(out, "w"), separators=(",", ":"))
print(len(tapes), "tapes,", len(acts), "unique actions")
