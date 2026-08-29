"""上位リプレイのルートを全品目リアクティブ市場エージェントで評価する (E018 M1)。

使い方:
    .venv/bin/python tests/eval_routes.py [--games 5] [--opponent random|base] [--top N]
"""
import argparse
import hashlib
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


def load_adaptive():
    root = Path(__file__).resolve().parent.parent
    return load_agent(root / "agents" / "adaptive_route.py")


def replay(route_agent, opponent, seed):
    env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=False)
    env.info["seed"] = seed
    env.run([route_agent, opponent])
    return env.steps[-1][0].reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="/tmp/opencode/topdata")
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--opponent", default="random", choices=["random", "starter", "base"])
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    if args.opponent == "base":
        opponent = load_agent(root / "agents" / "base.py").agent
    else:
        opponent = args.opponent
    adaptive = load_adaptive()

    files = sorted(Path(args.dir).glob("*.json"))
    results = []
    seen = set()
    for f in files:
        r = json.load(open(f))
        teams = r.get("info", {}).get("TeamNames", ["?", "?"])
        rewards = r.get("rewards", [0, 0])
        for pi in (0, 1):
            route = extract_route(str(f), player=pi)
            h = hashlib.md5(json.dumps(route, sort_keys=True).encode()).hexdigest()[:10]
            dup = h in seen
            seen.add(h)
            agent = adaptive.make_adaptive_agent(route)
            total = 0.0
            for seed in range(100, 100 + args.games):
                total += replay(agent, opponent, seed)
            avg = total / args.games
            results.append({
                "file": f.name, "player": pi, "team": teams[pi] if pi < len(teams) else "?",
                "orig_reward": rewards[pi], "avg": avg, "dup": dup, "hash": h,
            })
            print(f"{f.name} p{pi} [{teams[pi]}] orig=${rewards[pi]:,.0f} -> avg=${avg:,.0f} (dup={dup})", flush=True)

    print("\n=== ranking ===")
    results.sort(key=lambda x: -x["avg"])
    for i, res in enumerate(results[:args.top]):
        print(f"{i+1:2d}. {res['file']} p{res['player']} {res['team']}: "
              f"avg=${res['avg']:,.0f} (orig ${res['orig_reward']:,.0f}) dup={res['dup']}")


if __name__ == "__main__":
    main()