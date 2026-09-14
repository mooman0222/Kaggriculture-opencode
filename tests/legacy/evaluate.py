"""エージェントを複数試合評価する (勝率・平均所持金)。

使い方:
    .venv/bin/python tests/evaluate.py --games 20 --opponent random
    .venv/bin/python tests/evaluate.py --games 20 --opponent base
"""
import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kaggle_environments import make

OPPONENTS = {"random": "random", "starter": "starter"}


def load_agent(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # 提出時 kaggle が呼ぶのは最後の callable (= agent_entry)。ローカルもそれに合わせる
    return getattr(mod, "agent_entry", mod.agent)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--opponent", default="random", choices=["random", "starter", "base"])
    parser.add_argument("--seed0", type=int, default=0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    opponent = OPPONENTS.get(args.opponent, load_agent(root / "agents" / "base.py"))
    my_agent = load_agent(root / "agents" / "kaito_v56_orak16.py")

    wins = losses = ties = 0
    my_money, opp_money = [], []
    for seed in range(args.seed0, args.seed0 + args.games):
        env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=False)
        env.info["seed"] = seed
        env.run([my_agent, opponent])
        r0, r1 = env.steps[-1][0].reward, env.steps[-1][1].reward
        my_money.append(r0)
        opp_money.append(r1)
        if r0 > r1:
            wins += 1
        elif r0 < r1:
            losses += 1
        else:
            ties += 1

    n = args.games
    print(f"vs {args.opponent} ({n} games)")
    print(f"  win  : {wins} ({wins / n:.1%})")
    print(f"  loss : {losses} ({losses / n:.1%})")
    print(f"  tie  : {ties} ({ties / n:.1%})")
    print(f"  money: mine avg=${sum(my_money) / n:.0f} (min={min(my_money):.0f}, max={max(my_money):.0f})")
    print(f"         opp  avg=${sum(opp_money) / n:.0f} (min={min(opp_money):.0f}, max={max(opp_money):.0f})")


if __name__ == "__main__":
    main()