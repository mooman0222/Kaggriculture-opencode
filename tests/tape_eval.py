"""実戦リプレイの相手行動をテープ再生し、候補エージェントを同シードで実走させて margin を測る。

使い方: .venv/bin/python tests/tape_eval.py agents/X.py --dir tmp/e039_since0906 --sig 2ca32a682f [--n 8] [--cls C9S8G0K0]
候補 = 記録時の自エージェントなら recorded margin と一致する (再現チェック)。
注意: 相手は固定行動 (適応反応なし)。相手の農場行動は市場に依存しないが、資金差で購入が落ちる可能性はある
(最終資金が記録から大きくずれたら無効)。
"""
import argparse, importlib.util, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kaggle_environments import make
from tests.fetch_battles import animals


def load(p):
    s = importlib.util.spec_from_file_location(Path(p).stem + str(abs(hash(p))), p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return getattr(m, "agent_entry", m.agent)


def tape(steps, seat):
    def agent(obs, configuration=None):
        i = int(obs["step"]) + 1
        return steps[i][seat]["action"] if i < len(steps) else {"farmer": ["PASS"], "hands": [], "market": []}
    return agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent"); ap.add_argument("--dir", required=True); ap.add_argument("--sig"); ap.add_argument("--ep", nargs="*", type=int, help="エピソード ID 指定 (sig の代わり)")
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--cls", help="相手の最終クラスで絞る"); ap.add_argument("--team", default="MMN0222")
    a = ap.parse_args()
    cand = load(a.agent)
    rows = json.load(open(Path(a.dir) / "battles.json"))
    sel = [r for r in rows if (r["ep"] in set(a.ep) if a.ep else r["opp_sig"] == a.sig) and (not a.cls or r["opp_cls"] == a.cls)]
    sel = sorted(sel, key=lambda r: r["ep"])[: a.n]
    tot = 0; rec_tot = 0; w = 0
    for r in sel:
        rp = json.load(open(Path(a.dir) / f"episode-{r['ep']}-replay.json"))
        me = rp["info"]["TeamNames"].index(a.team); op = 1 - me
        env = make("kaggriculture", configuration={"episodeSteps": rp["configuration"]["episodeSteps"]}, debug=False)
        env.info["seed"] = rp["info"]["seed"]
        agents = [None, None]; agents[me] = cand; agents[op] = tape(rp["steps"], op)
        env.run(agents)
        last = env.steps[-1]; my = last[me].reward; his = last[op].reward
        m = my - his; rec = rp["rewards"][me] - rp["rewards"][op]
        tot += m; rec_tot += rec; w += m > 0
        cls = animals(last[0].observation["farms"][me])
        print(f"ep {r['ep']} seat{me} opp_cls {r['opp_cls']} recorded {rec:+8.0f} -> {m:+8.0f}  (me {my:8.0f} opp {his:8.0f} recorded_opp {rp['rewards'][op]:8.0f}) my_cls {cls} {'DONE' if last[me].status=='DONE' else last[me].status}", flush=True)
    n = len(sel)
    print(f"{a.agent} vs tape {a.sig or a.ep}: {w}W{n-w}L, mean margin {tot/n:+.0f} (recorded {rec_tot/n:+.0f}, delta {(tot-rec_tot)/n:+.0f})")


if __name__ == "__main__":
    main()
