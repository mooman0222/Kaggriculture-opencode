"""1試合を実行して結果を表示する。

使い方:
    .venv/bin/python tests/run_match.py [--seed 42] [--opponent random|starter|base] [--replay replay.json]

--opponent:
    random : 組み込みのランダムエージェント
    starter: 組み込みの carrot ループ (ベースライン)
    base   : agents/base.py の melon_maxxer 原版
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kaggle_environments import make

OPPONENTS = {"random": "random", "starter": "starter"}


def load_agent(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--opponent", default="random", choices=["random", "starter", "base"])
    parser.add_argument("--replay", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    opponent = OPPONENTS.get(args.opponent, load_agent(root / "agents" / "base.py"))
    my_agent = load_agent(root / "main.py")

    env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=False)
    env.info["seed"] = args.seed
    env.run([my_agent, opponent])

    final = env.steps[-1]
    obs = final[0].observation
    print(f"seed={args.seed} opponent={args.opponent}")
    for i, s in enumerate(final):
        print(f"  Player {i}: money=${s.reward:.0f} status={s.status}")
    for i, s in enumerate(final):
        farm = obs.farms[i]
        priv = s.observation.private
        print(
            f"  Player {i}: shed={dict(priv['shed'])} seeds={dict(priv['seeds'])} "
            f"hands={len(farm['hands'])} unlocked={farm['unlocked_quadrants']}"
        )
    print(f"  Prices: {dict(obs.market.prices)}")
    print(f"  Town shops: {obs.town.unlocked_shops}")

    if args.replay:
        with open(args.replay, "w") as f:
            json.dump(env.toJSON(), f)
        print(f"  Replay saved to {args.replay}")


if __name__ == "__main__":
    main()