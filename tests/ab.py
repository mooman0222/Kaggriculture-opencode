"""2つのエージェントを同じシード・同じ相手で A/B 比較する。

記事 736219 (前回1位) の推奨に従い、勝率と平均所持金を「同じシード」で比較する。

使い方:
    .venv/bin/python tests/ab.py --variant agents/experimental.py --opponent base --games 20
    .venv/bin/python tests/ab.py --variant agents/experimental.py --opponent agents/kaito_v56_orak16.py --games 20
    # --opponent agents/kaito_v56_orak16.py = 現最強を相手にした勝率テスト (強い相手での評価)

    # シードを固定して再現性を保証:
    .venv/bin/python tests/ab.py --variant agents/experimental.py --opponent base --seed0 42
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


def play(agent_fn, opponent, seed):
    env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=False)
    env.info["seed"] = seed
    env.run([agent_fn, opponent])
    return env.steps[-1][0].reward, env.steps[-1][1].reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, help="実験対象のエージェントファイル (A)")
    parser.add_argument("--base", default="agents/kaito_v56_orak16.py", help="比較基準のエージェントファイル (B)")
    parser.add_argument("--opponent", default="base", help="共通の対戦相手: random/starter/base/ファイルパス")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed0", type=int, default=0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    opponent = OPPONENTS.get(args.opponent)
    if opponent is None:
        if args.opponent == "base":
            opponent = load_agent(root / "agents" / "base.py")
        elif args.opponent.endswith(".py"):
            opponent = load_agent(args.opponent)
        else:
            parser.error(f"unknown opponent: {args.opponent}")

    variant = load_agent(args.variant)
    base = load_agent(args.base)

    res = {args.base: [], args.variant: []}
    for seed in range(args.seed0, args.seed0 + args.games):
        rv, ro = play(variant, opponent, seed)
        rb, ro2 = play(base, opponent, seed)
        res[args.variant].append((rv, ro))
        res[args.base].append((rb, ro2))

    print(f"A/B: {args.variant} vs {args.base}  (opponent={args.opponent}, {args.games} games, seeds {args.seed0}..{args.seed0 + args.games - 1})")
    for label, rows in res.items():
        wins = sum(1 for m, o in rows if m > o)
        losses = sum(1 for m, o in rows if m < o)
        ties = args.games - wins - losses
        money = [m for m, _ in rows]
        opp_money = [o for _, o in rows]
        print(f"  {label}")
        print(f"    win {wins} ({wins / args.games:.1%}) / loss {losses} / tie {ties}")
        print(f"    money: avg=${sum(money) / args.games:.0f} (min={min(money):.0f}, max={max(money):.0f})")
        print(f"    opponent avg=${sum(opp_money) / args.games:.0f}")

    per_seed = [(seed, res[args.variant][i][0], res[args.base][i][0]) for i, seed in
                enumerate(range(args.seed0, args.seed0 + args.games))]
    better = sum(1 for _, v, b in per_seed if v > b)
    print(f"  variant wins vs base on same seeds: {better}/{args.games}")


if __name__ == "__main__":
    main()