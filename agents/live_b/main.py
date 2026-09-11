"""Live policy B: Shop Router 0909 tapes for the opening (days 0-11), live planner from SWITCH_STEP."""
from __future__ import annotations
import importlib.util, os
from pathlib import Path

SWITCH_STEP = int(os.environ.get("LB_SWITCH", "288"))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


class Live:
    def __init__(self, folder):
        self.r = _load(folder / "router.py", "live_b_router"); self.policy = self.r.Policy(folder)
        self.pl = _load(folder / ("planner2.py" if os.environ.get("LB_PLANNER", "patrol") == "patrol" else "planner.py"), "live_b_planner")
        self.planner = self.pl.Planner()

    def act(self, obs):
        step = int(obs["step"])
        if step == 0: self.planner = self.pl.Planner()
        if step < SWITCH_STEP:
            act = self.policy.act(obs)
            if step <= 1:  # opening guard
                m = act["market"]
                if step == 0: act["market"] = [o for o in m if not (o and o[0] in ("BUY_PRODUCT", "SELL") and o[1] == "WHEAT")]
                else:
                    m = [o for o in m if not (o and o[0] == "SELL" and o[1] == "WHEAT")]
                    act["market"] = [o for o in m if o and o[0] == "BUY_ANIMAL"] + [o for o in m if not (o and o[0] == "BUY_ANIMAL")]
            return act
        return self.planner.act(obs)


_LIVE = None


def agent(observation, configuration=None):
    global _LIVE
    if _LIVE is None: _LIVE = Live(Path(agent.__code__.co_filename).resolve().parent)
    return _LIVE.act(observation)
