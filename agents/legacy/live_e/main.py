"""Live policy E: E058 (v41 chassis + live extras) for the opening, then a demand-aware per-turn planner from SWITCH_STEP."""
from __future__ import annotations
import importlib.util, os
from pathlib import Path

SWITCH_STEP = int(os.environ.get("LE_SWITCH", "144"))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


class Live:
    def __init__(self, folder):
        self.e058 = _load(folder / "e058.py", "live_e_e058").Live(folder)
        self.pl = _load(folder / "planner.py", "live_e_planner")
        self.planner = None

    def act(self, obs):
        step = int(obs["step"])
        if step == 0: self.planner = None
        if step < SWITCH_STEP: return self.e058.act(obs)
        if self.planner is None: self.planner = self.pl.Planner()
        try:
            return self.planner.act(obs)
        except Exception:
            if os.environ.get("LE_DEBUG"): raise
            farm = obs["farms"][int(obs["player"])]
            return {"farmer": ["PASS"], "hands": [["PASS"] for _ in farm["hands"]], "market": []}


_LIVE = None


def agent(observation, configuration=None):
    global _LIVE
    if _LIVE is None: _LIVE = Live(Path(agent.__code__.co_filename).resolve().parent)
    return _LIVE.act(observation)
