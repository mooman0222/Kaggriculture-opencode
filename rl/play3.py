"""Evaluate a Policy3 checkpoint against v41 in kagsim, alternating seats."""
import argparse
import hashlib
import importlib.util
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kagsim
import numpy as np
import torch

from act3 import act_policy3
from model3 import Policy3


class BCAgent3:
    def __init__(self, path, d=128, layers=3, device="cpu", temperature=0.0):
        self.device = torch.device(device)
        self.model = Policy3(d=d, layers=layers).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        self.temperature = temperature
        self.state = {}

    def act(self, obs, seat):
        if int(obs["step"]) == 0:
            self.state = {}
        return act_policy3(self.model, obs, seat, self.device, self.temperature, state=self.state)[0]


def load_agent(path):
    directory = os.path.dirname(os.path.abspath(path))
    sys.path.insert(0, directory)
    spec = importlib.util.spec_from_file_location("opponent_" + hashlib.md5(path.encode()).hexdigest()[:8], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--vs", default="third_party/public_agents/v41/main.py")
    parser.add_argument("--seed0", type=int, default=5000)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--temp", type=float, default=0.0)
    args = parser.parse_args()

    agent = BCAgent3(args.checkpoint, args.d, args.layers, temperature=args.temp)
    opponent = load_agent(args.vs)
    results = []
    started = time.time()
    max_step_time = 0.0
    for game_index in range(args.games):
        seed = args.seed0 + game_index // 2
        seat = game_index % 2
        for name in ("_LIVE", "_POLICY"):
            if hasattr(opponent, name):
                setattr(opponent, name, None)
        game = kagsim.Game(seed)
        while not game.done:
            observation = game.observe(seat)
            step_started = time.perf_counter()
            own_action = agent.act(observation, seat)
            max_step_time = max(max_step_time, time.perf_counter() - step_started)
            opponent_action = opponent.agent(game.observe(1 - seat))
            game.step(*((own_action, opponent_action) if seat == 0 else (opponent_action, own_action)))
        results.append((game.reward(seat), game.reward(1 - seat)))
        print(f"seed {seed} seat {seat}: own {game.reward(seat):8.0f} opp {game.reward(1 - seat):8.0f} margin {game.reward(seat) - game.reward(1 - seat):+8.0f}", flush=True)
    print(
        f"mean own {np.mean([row[0] for row in results]):.0f} opp {np.mean([row[1] for row in results]):.0f} "
        f"margin {np.mean([row[0] - row[1] for row in results]):+.0f} wins {sum(row[0] > row[1] for row in results)}/{len(results)} "
        f"max step {max_step_time * 1000:.0f} ms [{time.time() - started:.0f}s]"
    )


if __name__ == "__main__":
    main()