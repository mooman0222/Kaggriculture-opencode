"""Learned Policy3 agent (numpy inference). Weights in weights.npz next to this file;
features/actions/np_policy/np_policy2/np_policy3/act_common/np_act3 copied from rl/.
Market layer: sell rule C (evening bulk dump) + fertilizer dump, baked in.
Export: .venv/bin/python rl/export_agent.py tmp/rl/<ckpt>.pt [--d 128 --layers 3]"""
from __future__ import annotations
import os, sys, importlib.util
import numpy as np
from pathlib import Path


def _load(folder, name):
    spec = importlib.util.spec_from_file_location(name, str(folder / (name + ".py"))); mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


class RLAgent:
    def __init__(self, folder):
        folder = Path(folder); sys.path.insert(0, str(folder))
        _load(folder, "features"); _load(folder, "actions"); _load(folder, "np_policy"); _load(folder, "np_policy2")
        self.NP3 = _load(folder, "np_policy3"); self.ACT = _load(folder, "np_act3")
        W = dict(np.load(folder / "weights.npz")); meta = W.pop("__meta__", None)
        d, layers, heads = (int(meta[0]), int(meta[1]), int(meta[2])) if meta is not None else (128, 3, 4)
        self.pol = self.NP3.NpPolicy3(W, d, layers, heads)
        self.state = {}

    def act(self, obs):
        seat = int(obs["player"])
        if int(obs["step"]) == 0:
            self.state = {}
        return self.ACT.act_np_policy3(self.pol, obs, seat, state=self.state,
                                       sell_rule_c=True, dump_fert=True)[0]


_AGENT = None


def agent(observation, configuration=None):
    global _AGENT
    try:
        if _AGENT is None: _AGENT = RLAgent(Path(agent.__code__.co_filename).resolve().parent)
        return _AGENT.act(observation)
    except Exception:
        try:
            farm = observation["farms"][int(observation["player"])]
            return {"farmer": ["PASS"], "hands": [["PASS"] for _ in farm["hands"]], "market": []}
        except Exception:
            return {"farmer": ["PASS"], "hands": [], "market": []}


try:
    _AGENT = RLAgent(Path(agent.__code__.co_filename).resolve().parent)
except Exception:
    _AGENT = None
