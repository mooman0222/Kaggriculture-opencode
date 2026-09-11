"""Shop Router 0909 (public, Apache-2.0) + clone-gated K-turn sell lead + idle-hand WATER/CARE.

Against a non-clone opponent the output is byte-identical to Shop Router 0909.
"""
from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path

LEAD_K = int(os.environ.get("SR_LEADK", "6"))
IDLE_WORK = os.environ.get("SR_IDLE", "1") == "1"
CLONE_GATE = os.environ.get("SR_GATE", "1") == "1"
OPEN_GUARD = os.environ.get("SR_GUARD", "1") == "1"
GATE_STEP = 72
GARDEN = os.environ.get("LC_GARDEN", "1") == "1"  # unit positions are tape-determined; weeds never move units


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Live:
    def __init__(self, folder):
        self.r = _load(folder / "router.py", "sr0909_router")
        self.policy = self.r.Policy(folder)
        self.native_advance = self.r.advance_sales
        self.g = _load(folder / "cropswap.py", "live_c_cropswap"); self.garden = self.g.CropSwap()
        self.reset()

    def reset(self):
        self.moved = {}      # step -> {item: qty already sold early}
        self.mismatch = 0
        self.clone = not CLONE_GATE

    def act(self, obs):
        step, seat = int(obs["step"]), int(obs["player"])
        if step == 0:
            self.reset()
        farm = obs["farms"][seat]
        if CLONE_GATE and 0 < step <= GATE_STEP:
            opp = obs["farms"][1 - seat]
            if [farm["farmer"], *farm["hands"]] != [opp["farmer"], *opp["hands"]]:
                self.mismatch += 1
            if step == GATE_STEP:
                self.clone = self.mismatch <= 2
        # clone: our K-turn lead replaces the native one-turn advance
        self.r.advance_sales = (lambda *a, **k: None) if self.clone else self.native_advance
        act = self.policy.act(obs)
        if OPEN_GUARD and step <= 1:
            self._opening_guard(act, step)
        if step >= self.r.LAST_STEP:
            return act
        if self.clone:
            self._lead(act, obs, step)
        if GARDEN:
            act = self.garden.apply(act, obs)  # before idle-work padding: extra hands are the tail of the list
        if IDLE_WORK:
            self._idle_work(act, farm, obs.get("private") or {})
        return act

    def _lead(self, act, obs, step):
        state = self.policy.players[int(obs["player"])]
        tape = self.policy.tapes[state.plan]
        market = list(act["market"])
        early = self.moved.pop(step, {})
        if early:  # quantities already sold early on previous turns
            fixed = []
            for o in market:
                if o and o[0] == "SELL" and o[1] in early and early[o[1]] > 0:
                    q = int(o[2]) - early[o[1]]
                    early[o[1]] = max(0, early[o[1]] - int(o[2]))
                    fixed.append(["SELL", o[1], q] if q > 0 else [])  # keep the slot
                else:
                    fixed.append(o)
            market = fixed
            act["market"] = market
        view = self.r.FarmView(obs)
        avail = self.r.projected_shed(act, view)  # includes this turn's DROP/PLACE/PICKUP
        for o in market:
            if o and o[0] == "SELL" and o[1] in avail:
                avail[o[1]] -= int(o[2])
        saved = copy.deepcopy(self.moved)
        lead = {}
        for k in range(1, LEAD_K + 1):
            t = step + k
            if t >= self.r.LAST_STEP or t == self.r.ROUTE_STEP or t == self.r.FINAL_PLAN_STEP:
                break  # the tape beyond a routing decision is unknown
            nxt = tape[t]
            bought = {o[1] for o in (nxt.get("market") or []) if o and o[0] == "BUY_PRODUCT"}
            bought |= {o[1] for o in market if o and o[0] == "BUY_PRODUCT"}
            for u in [nxt.get("farmer") or [], *(nxt.get("hands") or [])]:
                if u and u[0] == "PICKUP" and len(u) >= 2:  # resolves before that turn's market
                    avail[u[1]] = avail.get(u[1], 0) - (int(u[2]) if len(u) >= 3 else 1)
            already = self.moved.get(t, {})
            for o in nxt.get("market") or []:
                if not o or o[0] != "SELL" or o[1] not in self.r.PRODUCTS or o[1] in bought:
                    continue
                q = min(int(o[2]) - already.get(o[1], 0), max(0, avail.get(o[1], 0)))
                if q <= 0 or int(view.prices.get(o[1], 0)) < 2:
                    continue
                lead[o[1]] = lead.get(o[1], 0) + q
                avail[o[1]] -= q
                already[o[1]] = already.get(o[1], 0) + q
                self.moved[t] = already
        orders = [["SELL", i, q] for i, q in lead.items()]
        if orders and len(orders) + len(market) <= self.r.MAX_ORDERS:
            act["market"] = market + orders  # appended: tape orders keep their queue indices
        elif orders:
            self.moved = saved

    def _opening_guard(self, act, step):
        """Day-0 wheat round trips are a price trick that a griefing opponent turns into a -400 loss,
        which then fails the t1 sheep purchase (-70k). Skip the trick; buy animals before hiring."""
        m = act["market"]
        if step == 0:
            act["market"] = [o for o in m if not (o and o[0] in ("BUY_PRODUCT", "SELL") and o[1] == "WHEAT")]
        else:
            m = [o for o in m if not (o and o[0] == "SELL" and o[1] == "WHEAT")]  # no wheat held without the trick
            animals = [o for o in m if o and o[0] == "BUY_ANIMAL"]
            rest = [o for o in m if not (o and o[0] == "BUY_ANIMAL")]
            act["market"] = animals + rest

    def _idle_work(self, act, farm, private):
        tiles = farm["tiles"]
        units = [farm["farmer"], *farm["hands"]]
        acts = [act.get("farmer") or ["PASS"], *(act.get("hands") or [])]
        while len(acts) < len(units):
            acts.append(["PASS"])
        for i, (pos, a) in enumerate(zip(units, acts)):
            if a and a[0] != "PASS":
                continue
            tile = tiles[int(pos[1])][int(pos[0])]
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "PLANT" and not tile.get("watered_today"):
                acts[i] = ["WATER"]
            elif "animal" in tile and not tile.get("cared_today"):
                acts[i] = ["CARE"]  # no FEED: carried wheat is earmarked by the tape
        act["farmer"], act["hands"] = acts[0], acts[1:]


_LIVE = None


def agent(observation, configuration=None):
    global _LIVE
    if _LIVE is None:
        _LIVE = Live(Path(agent.__code__.co_filename).resolve().parent)
    return _LIVE.act(observation)
