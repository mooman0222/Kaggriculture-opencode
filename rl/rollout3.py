"""Closed-loop correction shards (DAgger-style) for Policy3.

Play a Policy3 checkpoint against v41 in kagsim and write shards in the demo format (train_bc2.KEYS). Labels are the
policy's own choices, except a unit that idles (PASS at its tile before IDLE_UNTIL) while an exact-legal perishable task
exists gets the nearest such task: WATER / FEED first, then CARE / HARVEST / COLLECT_FERTILIZER / FERTILIZE. These are the
states the demos never show (own farm layout, own mistakes) where the model currently learns nothing but PASS.

使い方: .venv/bin/python rl/rollout3.py tmp/kaggle_out_bc9/bc9k_ep3.pt --out tmp/rl/dagger3 --games 20 --seed0 7000
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kagsim
import numpy as np

from act3 import act_policy3
from actions import encode_action
from features import MAX_UNITS, N_OPS, OP_INDEX, encode, legal_ops, legal_ops_at
from play3 import BCAgent3, load_agent
from sp.legal_all import legal_all

IDLE_UNTIL = 20
PRIORITY = [np.array([OP_INDEX["WATER"], OP_INDEX["FEED"]]), np.array([OP_INDEX[n] for n in ("CARE", "HARVEST", "COLLECT_FERTILIZER", "FERTILIZE")])]


def relabel_idle(dest, dop, features, hour):
    """Replace PASS-at-current labels with the nearest exact-legal perishable task while it is still daytime."""
    if hour >= IDLE_UNTIL:
        return 0
    exact = legal_all(features)
    units = features["units"]
    tiles = np.arange(100)
    changed = 0
    for i in range(MAX_UNITS):
        if units[i, 0] == 0 or dop[i] != OP_INDEX["PASS"]:
            continue
        distance = np.abs(tiles % 10 - units[i, 2]) + np.abs(tiles // 10 - units[i, 3])
        for group in PRIORITY:
            ok = exact[i][:, group].any(1)
            if ok.any():
                tile = int(np.where(ok, distance, 999).argmin())
                dest[i] = tile
                dop[i] = int(group[np.flatnonzero(exact[i][tile, group])[0]])
                changed += 1
                break
    return changed


def rollout(agent, opponent, seed, seat, relabel=True):
    for name in ("_LIVE", "_POLICY"):
        if hasattr(opponent, name):
            setattr(opponent, name, None)
    game = kagsim.Game(seed)
    agent.state = {}
    out = {key: [] for key in ("tiles", "units", "items", "glob", "mkt", "dest", "dop", "dqty", "dmask")}
    changed = 0
    while not game.done:
        obs = game.observe(seat)
        features = encode(obs, seat)
        action, info = act_policy3(agent.model, obs, seat, agent.device, state=agent.state)
        opponent_action = opponent.agent(game.observe(1 - seat))
        unit_count = len(info["dest"])
        _, qty, mkt = encode_action(action, unit_count)
        dest = np.full(MAX_UNITS, -1, dtype=np.int8)
        dop = np.full(MAX_UNITS, -1, dtype=np.int8)
        dest[:unit_count] = info["dest"]
        dop[:unit_count] = info["op"]
        if relabel:
            changed += relabel_idle(dest, dop, features, int(obs["step"]) % 24)
        dqty = np.where((dop >= 16) & (dop <= 40), qty, 0).astype(np.int8)
        dmask = np.zeros((MAX_UNITS, N_OPS), dtype=bool)
        for i in range(unit_count):
            dmask[i] = legal_ops_at(obs, seat, i, int(dest[i]) % 10, int(dest[i]) // 10)
        for key in ("tiles", "units", "items", "glob"):
            out[key].append(features[key])
        out["mkt"].append(mkt)
        out["dest"].append(dest)
        out["dop"].append(dop)
        out["dqty"].append(dqty)
        out["dmask"].append(dmask)
        game.step(*((action, opponent_action) if seat == 0 else (opponent_action, action)))
    shard = {key: np.stack(value) for key, value in out.items()}
    shard["reward"] = np.array([game.reward(seat), game.reward(1 - seat)], dtype=np.float32)
    return shard, changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--out", required=True)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed0", type=int, default=7000)
    parser.add_argument("--vs", default="third_party/public_agents/v41/main.py")
    parser.add_argument("--no-relabel", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    agent = BCAgent3(args.checkpoint)
    opponent = load_agent(args.vs)
    for index in range(args.games):
        seed, seat = args.seed0 + index // 2, index % 2
        shard, changed = rollout(agent, opponent, seed, seat, relabel=not args.no_relabel)
        path = os.path.join(args.out, f"dg_{seed}_{seat}.npz")
        np.savez_compressed(path, **shard)
        labels = int((shard["dest"] >= 0).sum())
        print(f"{path}: own {shard['reward'][0]:.0f} opp {shard['reward'][1]:.0f} labels {labels} relabeled {changed} ({changed / labels:.3f})", flush=True)


if __name__ == "__main__":
    main()
