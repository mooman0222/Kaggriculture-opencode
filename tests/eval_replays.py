"""実戦リプレイの中で Shop Router 0908 系が座っていた席を候補に差し替え、相手をテープ再生して margin を比較する。

使い方: .venv/bin/python tests/eval_replays.py agents/X/main.py --glob 'tmp/top0909/replays*/*/episode-*.json' [--lineage 267f5f1e] [--n 40]
"""
import argparse, glob, hashlib, importlib.util, json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kaggle_environments import make

PASS = {"farmer": ["PASS"], "hands": [], "market": []}
LIN = {'267f5f1e': 'SR0908', 'b3105bcc': 'SR0908v', '0b496ee6': 'ThomasOpen', '87b9b0d1': 'KaitoOpen', '149b67ca': '7bf8'}


def plan(a):
    a = a or PASS
    return (tuple(a.get("farmer") or ["PASS"]), tuple(tuple(h) for h in (a.get("hands") or [])))


def h72(steps, seat):
    return hashlib.sha256(json.dumps([plan(steps[t + 1][seat].get("action")) for t in range(72)]).encode()).hexdigest()[:8]


def load(p):
    s = importlib.util.spec_from_file_location("cand_" + str(abs(hash(p))), p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return getattr(m, "agent_entry", m.agent)


def tape(steps, seat):
    def agent(obs, configuration=None):
        i = int(obs["step"]) + 1
        return steps[i][seat]["action"] if i < len(steps) else PASS
    return agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent"); ap.add_argument("--glob", required=True); ap.add_argument("--lineage", default="267f5f1e")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--skip-mirror", action="store_true")
    ap.add_argument("--team", help="席の選択を系統ハッシュではなくチーム名で行う")
    ap.add_argument("--base", help="対照エージェント。指定時は recorded ではなく同席で base を走らせた結果と比較する")
    a = ap.parse_args()
    cand = load(a.agent); base = load(a.base) if a.base else None
    seen = set(); rows = []
    for f in sorted(glob.glob(a.glob)):
        r = json.load(open(f)); eid = r["info"].get("EpisodeId", f)
        if eid in seen or len(r["steps"]) < 720: continue
        seen.add(eid)
        hs = [h72(r["steps"], s) for s in (0, 1)]
        for me in (0, 1):
            if (r["info"]["TeamNames"][me] == a.team) if a.team else (hs[me] == a.lineage):
                opp = LIN.get(hs[1 - me], "other:" + r["info"]["TeamNames"][1 - me])
                if a.skip_mirror and opp == "SR0908": continue
                rows.append((f, me, opp, r)); break
    rows = rows[: a.n]
    agg = defaultdict(list)
    for f, me, opp, r in rows:
        env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=False)
        env.info["seed"] = r["info"]["seed"]
        agents = [None, None]; agents[me] = cand; agents[1 - me] = tape(r["steps"], 1 - me)
        env.run(agents)
        last = env.steps[-1]; my, his = last[me].reward, last[1 - me].reward
        rec = r["rewards"][me] - r["rewards"][1 - me]; new = my - his
        opp_drift = his - r["rewards"][1 - me]
        if base is not None:
            envb = make("kaggriculture", configuration={"episodeSteps": 720}, debug=False)
            envb.info["seed"] = r["info"]["seed"]
            agents[me] = base; envb.run(agents)
            lb = envb.steps[-1]; rec = lb[me].reward - lb[1 - me].reward; opp_drift = his - lb[1 - me].reward
        agg[opp].append((rec, new))
        print(f"{Path(f).name[8:17]} seat{me} vs {opp:<26} recorded {rec:+8.0f} -> {new:+8.0f} (delta {new-rec:+7.0f}, opp drift {opp_drift:+6.0f}) {last[me].status}", flush=True)
    print("== summary by opponent lineage (recorded W-L -> new W-L, mean delta)")
    tot = []
    for k, v in sorted(agg.items(), key=lambda x: -len(x[1])):
        rw = sum(1 for rec, _ in v if rec > 0); nw = sum(1 for _, new in v if new > 0)
        d = sum(new - rec for rec, new in v) / len(v); tot += v
        print(f"  {k:<26} n={len(v):2d} W{rw}->{nw}  mean delta {d:+7.0f}")
    if tot:
        print(f"  ALL n={len(tot)} W{sum(1 for r_,_ in tot if r_>0)}->{sum(1 for _,n_ in tot if n_>0)} mean delta {sum(n_-r_ for r_,n_ in tot)/len(tot):+.0f}")


if __name__ == "__main__":
    main()
