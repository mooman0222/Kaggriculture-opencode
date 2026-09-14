"""上位チームのリプレイから戦略シグネチャを抽出・集計する。

使い方: .venv/bin/python tests/analyze_top_replays.py [--dir tmp/topdata] [--extra FILE ...]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def extract(path):
    r = json.load(open(path))
    info = r.get("info", {})
    agents = info.get("Agents", [])
    team_names = info.get("TeamNames", ["?", "?"])
    rewards = r.get("rewards", [0, 0])
    seed = info.get("seed")
    out = {"file": Path(path).name, "seed": seed, "teams": team_names, "rewards": rewards}
    steps = r["steps"]
    for pi in (0, 1):
        ag = {"team": team_names[pi] if pi < len(team_names) else "?"}
        buys = Counter()
        sells = Counter()
        first_sell = {}
        hires = 0
        land = 0
        for si, step in enumerate(steps):
            a = step[pi].get("action") or {}
            for order in a.get("market", []):
                if not order:
                    continue
                op = order[0]
                if op == "HIRE":
                    hires += 1
                elif op == "BUY_LAND":
                    land += 1
                elif op in ("BUY_ANIMAL", "BUY_SEED", "BUY_PRODUCT", "SELL"):
                    item = order[1]
                    n = order[2] if len(order) > 2 else 1
                    if op == "SELL":
                        sells[item] += n
                        if item not in first_sell:
                            first_sell[item] = si
                    else:
                        buys[(op, item)] += n
        ag["buys"] = dict(buys)
        ag["sells"] = dict(sells)
        ag["first_sell_step"] = first_sell
        ag["hires"] = hires
        ag["land"] = land
        # 最終状態
        obs = steps[-1][pi]["observation"]
        farm = obs["farms"][pi]
        ag["unlocked"] = farm.get("unlocked_quadrants", [])
        tiles = farm["tiles"]
        cnt = Counter()
        for row in tiles:
            for t in row:
                if t is None:
                    cnt["EMPTY"] += 1
                elif t == "LOCKED":
                    cnt["LOCKED"] += 1
                elif isinstance(t, dict):
                    k = t.get("kind")
                    if k == "PLANT":
                        cnt["PLANT:" + t.get("crop", "?")] += 1
                    elif "animal" in t:
                        cnt["ANIMAL:" + t["animal"]] += 1
                    else:
                        cnt[k] += 1
        ag["final_tiles"] = dict(cnt)
        out[f"p{pi}"] = ag
    # 価格推移
    prices = {}
    for day in (0, 10, 20, 29):
        obs = steps[day * 24][0]["observation"]
        p = obs["market"]["prices"]
        prices[f"d{day}"] = {k: p[k] for k in ("WHEAT", "MELON", "STRAWBERRY", "MILK", "WOOL", "EGG", "FERTILIZER")}
    out["prices"] = prices
    return out


def fmt_row(g):
    parts = []
    for pi in (0, 1):
        ag = g[f"p{pi}"]
        buys = ag["buys"]
        animals = {k[1]: v for k, v in buys.items() if k[0] == "BUY_ANIMAL"}
        seeds = {k[1]: v for k, v in buys.items() if k[0] == "BUY_SEED"}
        buyprod = {k[1]: v for k, v in buys.items() if k[0] == "BUY_PRODUCT"}
        a = ",".join(f"{k}:{v}" for k, v in sorted(animals.items()))
        s = ",".join(f"{k}:{v}" for k, v in sorted(seeds.items()))
        bp = ",".join(f"{k}:{v}" for k, v in sorted(buyprod.items()))
        sl = ",".join(f"{k}:{v}" for k, v in sorted(ag["sells"].items()))
        parts.append(
            f"${ag['team']} ${g['rewards'][pi]:,.0f} | animals[{a}] seeds[{s}] buyP[{bp}] sells[{sl}] "
            f"hire{ag['hires']} land{ag['land']} tiles[{','.join(f'{k}:{v}' for k,v in sorted(ag['final_tiles'].items()) if k != 'EMPTY' and k != 'LOCKED')}]"
        )
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="tmp/topdata")
    parser.add_argument("--extra", action="append", default=[])
    args = parser.parse_args()

    files = sorted(Path(args.dir).glob("*.json")) + [Path(p) for p in args.extra]
    games = [extract(str(f)) for f in files if f.exists()]
    print(f"games: {len(games)}")
    for g in games:
        print("=" * 100)
        print(g["file"], "seed=", g["seed"])
        print(fmt_row(g))
        p = g["prices"]
        print("prices: " + " | ".join(f"d{d}: W={p[f'd{d}']['WHEAT']} M={p[f'd{d}']['MELON']} S={p[f'd{d}']['STRAWBERRY']} F={p[f'd{d}']['FERTILIZER']}" for d in (0, 10, 20, 29)))


if __name__ == "__main__":
    main()