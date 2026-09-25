"""Representation ceiling: replay a teacher game in kagsim, but drive the teacher seat through the BC action
abstraction fed with the *ground-truth labels* (a perfect model). unit/market = rec (recorded raw action) | oracle
(labels -> act3 decoding). rec/rec also checks replay-obs vs kagsim-obs feature parity.
usage: oracle.py 'glob' [n]"""
import sys, json, glob, os
import numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), "rl"))
import kagsim
from features import encode, MAX_UNITS, OPS, QTY_BUCKETS, PRODUCTS, unbucket
from actions import encode_action, decode_action
from labels import destination_labels
from act_common import step_toward, trim_plants

PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def unit_from_label(pos, dest, dop, dqty):
    x, y = dest % 10, dest // 10
    if (int(pos[0]), int(pos[1])) != (x, y):
        return step_toward((int(pos[0]), int(pos[1])), (x, y))
    name = OPS[dop]; q = unbucket(dqty, QTY_BUCKETS)
    if name.startswith("PLANT_"): return ["PLANT", name[6:]]
    if name.startswith("PICKUP_"): return ["PICKUP", name[7:], int(q)]
    if name.startswith("PLACE_"):
        it = name[6:]; return ["PLACE", it, int(q)] if it in PRODUCTS else ["PLACE", it]
    return [name]


def run(f, team=os.environ.get("TEAM", "Majkel1337"), modes=(("rec", "rec"), ("oracle", "rec"), ("rec", "oracle"), ("oracle", "oracle"))):
    r = json.load(open(f)); names = r["info"]["TeamNames"]
    if team not in names or len(r["steps"]) < 720: return None
    me = names.index(team); st = r["steps"]; shops = st[-1][0]["observation"]["town"]["unlocked_shops"]
    for t in range(720): st[t][me]["observation"]["step"] = t
    dest, dop, dqty = destination_labels(st, me, 719)
    rec = (r["rewards"][me] or 0, r["rewards"][1 - me] or 0)
    out = {"eid": r["info"]["EpisodeId"], "rec": rec}
    for mu, mm in modes:
        g = kagsim.Game(r["info"]["seed"], 720, shops)
        parity_bad = 0; unk = 0; first_div = None
        for t in range(719):
            obs = g.observe(me)
            ra = st[t + 1][me]["action"] or PASS; oa = st[t + 1][1 - me]["action"] or PASS
            if mu == "rec" and mm == "rec":
                a, b = encode(obs, me), encode(st[t][me]["observation"], me)
                if any(not np.array_equal(a[k], b[k]) for k in a): parity_bad += 1
            act = {"farmer": ra.get("farmer") or ["PASS"], "hands": list(ra.get("hands") or []), "market": list(ra.get("market") or [])}
            if mu == "oracle":
                units = [obs["farms"][me]["farmer"], *obs["farms"][me]["hands"]]
                ru = [ra.get("farmer") or ["PASS"], *(ra.get("hands") or [])]
                ua = []
                for i, u in enumerate(units):
                    d = int(dest[t, i]) if i < MAX_UNITS else -1
                    if d < 0:
                        unk += 1; ua.append(ru[i] if i < len(ru) else ["PASS"])
                    else:
                        ua.append(unit_from_label(u, d, int(dop[t, i]), int(dqty[t, i])))
                trim_plants(ua, obs["private"]["seeds"])
                act["farmer"], act["hands"] = ua[0], ua[1:]
            if mm == "oracle":
                n = min(MAX_UNITS, 1 + len(obs["farms"][me]["hands"]))
                mkt = encode_action(ra, n)[2]
                act["market"] = decode_action(np.zeros(MAX_UNITS, int), np.zeros(MAX_UNITS, int), mkt, obs, me)["market"]
            if first_div is None:
                ro = st[t][me]["observation"]["farms"][me]
                so = obs["farms"][me]
                if ro["money"] != so["money"] or ro["farmer"] != so["farmer"] or ro["hands"] != so["hands"]: first_div = t
            g.step(*((act, oa) if me == 0 else (oa, act)))
        res = (g.reward(me), g.reward(1 - me))
        out[f"{mu}/{mm}"] = dict(own=res[0], opp=res[1], parity_bad=parity_bad, unk=unk, first_div=first_div)
    return out


if __name__ == "__main__":
    files = sorted(glob.glob(sys.argv[1]))[: int(sys.argv[2]) if len(sys.argv) > 2 else 8]
    rows = []
    for f in files:
        o = run(f)
        if o is None: continue
        rows.append(o)
        print(o["eid"], "rec own %.0f opp %.0f" % o["rec"], " | ".join(
            f"{k}: own {v['own']:.0f} m {v['own'] - v['opp']:+.0f} div@{v['first_div']} par{v['parity_bad']} unk{v['unk']}"
            for k, v in o.items() if "/" in k), flush=True)
    for k in [k for k in rows[0] if "/" in k]:
        own = np.array([r[k]["own"] for r in rows]); m = np.array([r[k]["own"] - r[k]["opp"] for r in rows])
        ro = np.array([r["rec"][0] for r in rows]); rm = np.array([r["rec"][0] - r["rec"][1] for r in rows])
        d = own - ro; dm = m - rm
        print(f"{k:14s} n={len(rows)} d_own {d.mean():+8.0f} ± {d.std(ddof=1) / np.sqrt(len(d)):6.0f}  d_margin {dm.mean():+8.0f} ± {dm.std(ddof=1) / np.sqrt(len(d)):6.0f}")
