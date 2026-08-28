"""自前ルート生成のスキャフォールド (E018 プロジェクト用)。

次セッションがこのツールを土台にルート型エージェントを開発する。
主な機能:
  - extract_route: プレイ済みゲームから行動ルート (720ステップ) を抽出
  - replay_route: ルートを任意の相手・シードで再生して報酬を評価
  - evaluate_route: 複数シードの平均報酬
  - mutate_route: ルートの変異 (市場オーダーの変更)
  - optimize: 単純な山登り/GA による最適化
  - compress_route: base85+zlib 圧縮 (提出用埋め込み)

使い方:
  .venv/bin/python tests/route_gen.py --source /tmp/opencode/topdata/episode-XXX-replay.json
  .venv/bin/python tests/route_gen.py --extract-only ...
"""
import argparse
import base64
import json
import random
import sys
import zlib
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kaggle_environments import make


# ---------------------------------------------------------------------------
# ルート抽出
# ---------------------------------------------------------------------------

def extract_route(replay_path, player=0):
    """リプレイ JSON からプレイヤーの行動ルートを抽出する。"""
    r = json.load(open(replay_path))
    route = []
    for step in r["steps"]:
        a = step[player].get("action") or {}
        route.append({
            "farmer": a.get("farmer", ["PASS"]),
            "hands": a.get("hands", []),
            "market": a.get("market", []),
        })
    return route


# ---------------------------------------------------------------------------
# ルート再生 (雑草修復 + ハンド調整のオーバーレイ付き)
# ---------------------------------------------------------------------------

def _is_weed(tile):
    return isinstance(tile, dict) and tile.get("kind") == "WEED"


def _repair(action, pos, tiles):
    """PLANT/BUILD が雑草タイルで実行される場合 DIG に差し替える。"""
    if not action:
        return ["PASS"]
    op = action[0]
    fx, fy = pos
    tile = tiles[fy][fx]
    if op in ("PLANT", "BUILD_COOP", "BUILD_PASTURE") and _is_weed(tile):
        return ["DIG"]
    return action


def make_replay_agent(route):
    """ルートを再生するエージェント (雑草修復・ハンド調整付き)。"""
    def agent(obs):
        try:
            farms = obs.get("farms", [])
            player = obs.get("player", 0)
            if not farms or player >= len(farms):
                return {"farmer": ["PASS"], "hands": [], "market": []}
            farm = farms[player]
            step = obs.get("step", obs.get("day", 0) * 24 + obs.get("hour", 0))
            if step >= len(route):
                return {"farmer": ["PASS"], "hands": [], "market": []}
            r = route[step]
            farmer = _repair(r.get("farmer", ["PASS"]), farm["farmer"], farm["tiles"])
            actual_hands = farm.get("hands", [])
            hands = []
            for i, ha in enumerate(r.get("hands", [])):
                if i >= len(actual_hands):
                    break
                hands.append(_repair(ha, actual_hands[i], farm["tiles"]))
            while len(hands) < len(actual_hands):
                hands.append(["PASS"])
            return {"farmer": farmer, "hands": hands, "market": r.get("market", [])}
        except Exception:
            return {"farmer": ["PASS"], "hands": [], "market": []}
    return agent


def replay_route(route, opponent, seed):
    """ルートを1ゲーム再生し、報酬を返す。"""
    env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=False)
    env.info["seed"] = seed
    env.run([make_replay_agent(route), opponent])
    return env.steps[-1][0].reward


def evaluate_route(route, opponent, n=5, seed0=0):
    """複数シードの平均報酬。"""
    total = 0
    for seed in range(seed0, seed0 + n):
        total += replay_route(route, opponent, seed)
    return total / n


# ---------------------------------------------------------------------------
# 変異・最適化
# ---------------------------------------------------------------------------

def mutate_route(route, rng, rate=0.05):
    """市場オーダーを中心にランダム変異する (農場行動は破壊しにくい)。

    変異の種類:
      - 市場オーダーの数量変更 (±1-3)
      - 市場オーダーの削除/追加 (低確率)
      - ハンドの PASS 化 (低確率)
    """
    new = []
    for s, r in enumerate(route):
        entry = {"farmer": list(r["farmer"]), "hands": [list(h) for h in r["hands"]], "market": [list(o) for o in r["market"]]}
        # 市場オーダーの数量変更
        for i, o in enumerate(entry["market"]):
            if o and len(o) >= 3 and rng.random() < rate * 4:
                o[2] = max(1, o[2] + rng.choice([-2, -1, 1, 2]))
        # ハンドの PASS 化 (労働力調整)
        for i in range(len(entry["hands"])):
            if rng.random() < rate * 0.5:
                entry["hands"][i] = ["PASS"]
        new.append(entry)
    return new


def optimize(seed_route, opponent, generations=20, pop_size=8, n_eval=3, seed0=100):
    """単純な山登り: 現ベストの変異を評価して置換。"""
    rng = random.Random(42)
    best = seed_route
    best_score = evaluate_route(best, opponent, n_eval, seed0)
    print(f"gen0: best=${best_score:.0f}")
    for g in range(1, generations + 1):
        for _ in range(pop_size):
            cand = mutate_route(best, rng)
            score = evaluate_route(cand, opponent, n_eval, seed0)
            if score > best_score:
                best, best_score = cand, score
                print(f"gen{g}: best=${best_score:.0f} (improved)")
    return best, best_score


# ---------------------------------------------------------------------------
# 圧縮 (提出用)
# ---------------------------------------------------------------------------

def compress_route(route):
    return base64.b85encode(zlib.compress(json.dumps(route).encode(), 9)).decode()


def decompress_route(b85):
    return json.loads(zlib.decompress(base64.b85decode(b85)))


# ---------------------------------------------------------------------------
# ルートの要約 (何をしているルートか)
# ---------------------------------------------------------------------------

def summarize_route(route):
    buys = Counter()
    sells = Counter()
    hires = 0
    land = 0
    for r in route:
        for o in r["market"]:
            if not o:
                continue
            op = o[0]
            if op == "HIRE":
                hires += 1
            elif op == "BUY_LAND":
                land += 1
            elif op in ("BUY_ANIMAL", "BUY_SEED", "BUY_PRODUCT", "SELL"):
                item = o[1]
                n = o[2] if len(o) > 2 else 1
                if op == "SELL":
                    sells[item] += n
                else:
                    buys[(op, item)] += n
    return {
        "hires": hires, "land": land,
        "buys": dict(buys), "sells": dict(sells),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="ルート抽出元のリプレイ JSON")
    parser.add_argument("--extract-only", action="store_true", help="抽出のみ行い保存する")
    parser.add_argument("--out", default="/tmp/opencode/route_gen_out.json")
    parser.add_argument("--opponent", default="random")
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--optimize", action="store_true", help="ルートを最適化する")
    parser.add_argument("--generations", type=int, default=20)
    args = parser.parse_args()

    if args.source:
        route = extract_route(args.source)
        print("=== ルート要約 ===")
        print(summarize_route(route))
        if args.extract_only:
            with open(args.out, "w") as f:
                json.dump(route, f)
            print(f"saved to {args.out}")
            return
        print(f"=== 評価 (vs {args.opponent}, {args.games} games) ===")
        score = evaluate_route(route, args.opponent, args.games)
        print(f"avg reward: ${score:.0f}")
        if args.optimize:
            print("=== 最適化 ===")
            best, best_score = optimize(route, args.opponent, args.generations)
            print(f"optimized: ${best_score:.0f}")
            with open(args.out, "w") as f:
                json.dump(best, f)
            print(f"saved to {args.out}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()