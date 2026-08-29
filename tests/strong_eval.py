"""強敵相手の評価ハーネス (E018 用)。

リプレイから相手のルートを抽出し、adaptive_route の市場オーバーレイを被せて
「強敵エージェント」として再生する。LB 実戦 (topdata/lbdata) の相手を
ローカルで再現するのが目的。

使い方:
    .venv/bin/python tests/strong_eval.py --agent agents/adaptive_route.py \\
        --replays /tmp/opencode/lbdata --games 3
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kaggle_environments import make

from tests.route_gen import extract_route


def load_agent(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def replay(p0_agent, p1_agent, seed):
    env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=False)
    env.info["seed"] = seed
    env.run([p0_agent, p1_agent])
    return env.steps[-1][0].reward, env.steps[-1][1].reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, help="評価するエージェントの .py")
    parser.add_argument("--replays", required=True, help="リプレイ JSON のあるディレクトリ")
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--own-team", default="MMN0222", help="自分側のチーム名 (スキップする)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    mod = load_agent(Path(args.agent))
    adaptive = load_agent(root / "agents" / "adaptive_route.py")
    our_route = json.load(open(root / ".opencode/data/route_itmoni_101730370_p0.json"))

    results = []
    for f in sorted(Path(args.replays).glob("*.json")):
        r = json.load(open(f))
        teams = r.get("info", {}).get("TeamNames", ["?", "?"])
        rewards = r.get("rewards", [0, 0])
        for pi in (0, 1):
            name = teams[pi] if pi < len(teams) else "?"
            if args.own_team and args.own_team in (name or ""):
                continue
            route = extract_route(str(f), player=pi)
            if not route:
                continue
            opp = adaptive.make_adaptive_agent(route)
            total = 0.0
            total_opp = 0.0
            wins = 0
            for seed in range(args.seed0, args.seed0 + args.games):
                r0, r1 = replay(mod.make_adaptive_agent(our_route), opp, seed)
                total += r0
                total_opp += r1
                wins += r0 > r1
            results.append((f.name, pi, name, rewards[pi], total / args.games, total_opp / args.games, wins, args.games))
            print(f"{f.name} p{pi} [{name}] orig_opp_reward=${rewards[pi]:,.0f} | "
                  f"our avg=${total/args.games:,.0f} vs opp avg=${total_opp/args.games:,.0f} "
                  f"({wins}/{args.games} wins)", flush=True)

    print("\n=== summary ===")
    if results:
        avg_our = sum(r[4] for r in results) / len(results)
        avg_opp = sum(r[5] for r in results) / len(results)
        avg_winrate = sum(r[6] for r in results) / sum(r[7] for r in results)
        print(f"overall: our avg=${avg_our:,.0f} opp avg=${avg_opp:,.0f} winrate={avg_winrate:.2f} "
              f"({len(results)} opponents)")


if __name__ == "__main__":
    main()
