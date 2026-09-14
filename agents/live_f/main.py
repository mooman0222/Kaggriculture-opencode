"""Live policy F: E058 (v41 chassis + live extras) with tile-live decisions layered on the tape's unit movements."""
from __future__ import annotations
import importlib.util, os
from pathlib import Path

START_STEP = int(os.environ.get("LF_START", "48"))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


class Live:
    def __init__(self, folder):
        self.e058 = _load(folder / "e058.py", "live_f_e058").Live(folder)
        self.tl = _load(folder / "tilelive.py", "live_f_tilelive").TileLive(self.e058.b)

    def act(self, obs):
        step = int(obs["step"])
        if step == 0: self.tl.day = -1
        act = self.e058.act(obs)
        if step < START_STEP or step >= 718: return act
        try:
            return self.tl.act(act, obs, step)
        except Exception:
            if os.environ.get("LE_DEBUG"): raise
            return act


_LIVE = None


def agent(observation, configuration=None):
    global _LIVE
    if _LIVE is None: _LIVE = Live(Path(agent.__code__.co_filename).resolve().parent)
    return _LIVE.act(observation)
