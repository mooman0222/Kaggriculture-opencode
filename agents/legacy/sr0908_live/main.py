"""Shop Router 0908 (public tape router) + live layers: idle-hand work and one-turn sell lead."""
from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path



PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
PASS_ACT = {"farmer": ["PASS"], "hands": [], "market": []}

# layer switches (env overrides for A/B)
IDLE_WORK = os.environ.get("SR_IDLE", "1") == "1"
SELL_LEAD = os.environ.get("SR_LEAD", "1") == "1"
LEAD_K = int(os.environ.get("SR_LEADK", "6"))
CLONE_GATE = os.environ.get("SR_GATE", "1") == "1"
GATE_STEP = 72  # decide by the end of day 2; positions are tape-determined, weeds never move units


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Live:
    def __init__(self, folder):
        router_mod = _load(folder / "router.py", "sr0908_router")
        self.router = router_mod.Router(folder)
        self.reset()

    def reset(self):
        self.moved = {}  # step -> {item: qty already sold one turn early}
        self.mismatch = 0
        self.clone = not CLONE_GATE  # opponent runs the same tape (same unit positions every step)

    def act(self, obs):
        step = int(obs["step"])
        if step == 0:
            self.reset()
        base = self.router.act(obs)
        act = copy.deepcopy(base)
        seat = int(obs["player"])
        farm = obs["farms"][seat]
        private = obs.get("private") or {}
        if CLONE_GATE and 0 < step <= GATE_STEP:
            opp = obs["farms"][1 - seat]
            if [farm["farmer"], *farm["hands"]] != [opp["farmer"], *opp["hands"]]:
                self.mismatch += 1
            if step == GATE_STEP:
                self.clone = self.mismatch <= 2
        if IDLE_WORK:
            self._idle_work(act, farm, private)
        if SELL_LEAD and self.clone and step < 718:
            self._sell_lead(act, step, private)
        return act

    # --- layer A: idle units water/care/feed where they stand -------------
    def _idle_work(self, act, farm, private):
        tiles = farm["tiles"]
        units = [farm["farmer"], *farm["hands"]]
        invs = private.get("inventories") or []
        acts = [act.get("farmer") or ["PASS"], *(act.get("hands") or [])]
        # pad hands actions to the real hand count so idle hands get work too
        while len(acts) < len(units):
            acts.append(["PASS"])
        for i, (pos, a) in enumerate(zip(units, acts)):
            if a and a[0] != "PASS":
                continue
            x, y = int(pos[0]), int(pos[1])
            tile = tiles[y][x]
            if not isinstance(tile, dict):
                continue
            inv = invs[i] if i < len(invs) else {}
            if tile.get("kind") == "PLANT":
                if not tile.get("watered_today"):
                    acts[i] = ["WATER"]
            elif "animal" in tile:
                # no FEED here: carried wheat is earmarked by the tape for other animals
                if not tile.get("cared_today"):
                    acts[i] = ["CARE"]
        act["farmer"] = acts[0]
        act["hands"] = acts[1:]

    # --- layer B: sell next turn's tape sells now if the shed already has them
    def _sell_lead(self, act, step, private):
        shed = private.get("shed") or {}
        market = [o for o in (act.get("market") or [])]
        # drop quantities already sold early on the previous turn
        early = self.moved.pop(step, {})
        if early:
            fixed = []
            for o in market:
                if o and o[0] == "SELL" and o[1] in early and early[o[1]] > 0:
                    q = int(o[2]) - early[o[1]]
                    early[o[1]] = max(0, early[o[1]] - int(o[2]))
                    if q > 0:
                        fixed.append(["SELL", o[1], q])
                    else:
                        fixed.append([])  # keep the slot so later indices stay aligned
                else:
                    fixed.append(o)
            market = fixed
        avail = dict(shed)
        for o in market:  # stock already committed by this turn's own sells
            if o and o[0] == "SELL" and o[1] in avail:
                avail[o[1]] = avail.get(o[1], 0) - int(o[2])
        lead = []
        saved = copy.deepcopy(self.moved)
        tapes = self.router.tapes[self.router.active]
        for k in range(1, LEAD_K + 1):
            t = step + k
            if t >= 719 or t in self.router.stages:
                break  # tape beyond a routing decision is unknown
            nxt = tapes[t]
            bought = {o[1] for o in (nxt.get("market") or []) if o and o[0] == "BUY_PRODUCT"}
            bought |= {o[1] for o in market if o and o[0] == "BUY_PRODUCT"}
            # unit PICKUPs at t resolve before t's market: that stock must stay in the shed
            for u in [nxt.get("farmer") or [], *(nxt.get("hands") or [])]:
                if u and u[0] == "PICKUP" and len(u) >= 2:
                    avail[u[1]] = avail.get(u[1], 0) - (int(u[2]) if len(u) >= 3 else 1)
            already = self.moved.get(t, {})
            for o in nxt.get("market") or []:
                if not o or o[0] != "SELL" or o[1] not in PRODUCTS or o[1] in bought:
                    continue  # buy/sell round trips are price tricks, not liquidation
                want = int(o[2]) - already.get(o[1], 0)
                q = min(want, max(0, avail.get(o[1], 0)))
                if q <= 0:
                    continue
                lead.append(["SELL", o[1], q])
                avail[o[1]] -= q
                already[o[1]] = already.get(o[1], 0) + q
                self.moved[t] = already
        if lead:
            merged = {}
            for _, item, q in lead:
                merged[item] = merged.get(item, 0) + q
            lead = [["SELL", i, q] for i, q in merged.items()]
            if len(lead) + len(market) <= 10:
                # appended: the tape's own orders keep their queue indices
                market = market + lead
            else:  # over the order cap: skip leading this turn
                self.moved = saved
        act["market"] = market


_LIVE = None


def agent(observation, configuration=None):
    global _LIVE
    if _LIVE is None:
        _LIVE = Live(Path(agent.__code__.co_filename).resolve().parent)
    return _LIVE.act(observation)
