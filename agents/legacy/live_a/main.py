"""Live policy A: Shop Router 0909 farm plan + a live, price-aware seller.

The tape's SELL orders (except WHEAT, which the tape manages for feed) are removed and replaced by a
seller that reads current prices and shed pressure every turn: sell in small lots while the price holds,
hold while it recovers through town demand, force sales only under shed pressure or at the end.
"""
from __future__ import annotations

import copy
import importlib.util
import math
import os
from pathlib import Path

P = {  # tunables (env overrides for A/B)
    "hold": float(os.environ.get("LA_HOLD", "0.85")),      # sell only if price >= hold * base
    "burst": int(os.environ.get("LA_BURST", "2")),          # max units per product per turn while holding
    "shed_hi": int(os.environ.get("LA_SHED", "80")),        # force sales above this shed fill
    "endgame": int(os.environ.get("LA_END", "672")),        # from this step, liquidate progressively
    "lead": int(os.environ.get("LA_LEAD", "6")),            # clone pre-dump lookahead (0 = off)
}
BASE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}
IDLE_WORK = os.environ.get("SR_IDLE", "1") == "1"
FERTILIZE = os.environ.get("LA_FERT", "1") == "1"
GATE_STEP = 72


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


class Live:
    def __init__(self, folder):
        self.r = _load(folder / "router.py", "live_a_router")
        self.policy = self.r.Policy(folder)
        self.r.advance_sales = lambda *a, **k: None  # the tape's one-turn advance is replaced by the seller
        self.reset()

    def reset(self):
        self.mismatch = 0; self.clone = False

    def act(self, obs):
        step, seat = int(obs["step"]), int(obs["player"])
        if step == 0: self.reset()
        farm = obs["farms"][seat]
        if 0 < step <= GATE_STEP:
            opp = obs["farms"][1 - seat]
            if [farm["farmer"], *farm["hands"]] != [opp["farmer"], *opp["hands"]]: self.mismatch += 1
            if step == GATE_STEP: self.clone = self.mismatch <= 2
        act = self.policy.act(obs)
        if step <= 1: self._opening_guard(act, step)
        if step >= self.r.LAST_STEP: return act  # tape's own liquidation
        self._sell(act, obs, step)
        if IDLE_WORK: self._idle_work(act, farm, obs.get("private"), step // 24)
        return act

    def _opening_guard(self, act, step):
        m = act["market"]
        if step == 0:
            act["market"] = [o for o in m if not (o and o[0] in ("BUY_PRODUCT", "SELL") and o[1] == "WHEAT")]
        else:
            m = [o for o in m if not (o and o[0] == "SELL" and o[1] == "WHEAT")]
            act["market"] = [o for o in m if o and o[0] == "BUY_ANIMAL"] + [o for o in m if not (o and o[0] == "BUY_ANIMAL")]

    def _sell(self, act, obs, step):
        """Keep the tape's tuned sells; add guards: cash for upcoming purchases, shed pressure,
        fertilizer ASAP (no demand ever restores it), and clone pre-dump."""
        r = self.r
        state = self.policy.players[int(obs["player"])]; tape = self.policy.tapes[state.plan]
        market = list(act["market"])
        view = r.FarmView(obs); prices = view.prices
        stock = r.projected_shed(act, view)
        for o in market:
            if o and o[0] == "SELL" and o[1] in stock: stock[o[1]] -= int(o[2])
        reserve = {}; cost = 0
        for k in range(1, 9):
            t = step + k
            if t >= r.LAST_STEP: break
            for u in [tape[t].get("farmer") or [], *(tape[t].get("hands") or [])]:
                if u and u[0] == "PICKUP" and len(u) >= 2: reserve[u[1]] = reserve.get(u[1], 0) + (int(u[2]) if len(u) >= 3 else 1)
            for o in tape[t].get("market") or []:
                if not o: continue
                if o[0] == "BUY_ANIMAL": cost += {"GOOSE": 300, "COW": 400, "SHEEP": 500}[o[1]] * int(o[2])
                elif o[0] == "BUY_LAND": cost += 2000
                elif o[0] == "BUY_SEED": cost += {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}[o[1]] * int(o[2])
                elif o[0] == "BUY_PRODUCT": cost += prices.get(o[1], 30) * int(o[2])
        money = float(obs["farms"][int(obs["player"])]["money"])
        for o in market:  # this turn's own orders
            if o and o[0] == "SELL": money += prices.get(o[1], 0) * min(int(o[2]), max(0, stock.get(o[1], 0) + int(o[2])))
            elif o and o[0] == "BUY_ANIMAL": money -= {"GOOSE": 300, "COW": 400, "SHEEP": 500}[o[1]] * int(o[2])
        need = max(0.0, cost - money)
        dump_soon = {}
        if self.clone and P["lead"]:
            for k in range(1, P["lead"] + 1):
                t = step + k
                if t >= r.LAST_STEP or t == r.ROUTE_STEP or t == r.FINAL_PLAN_STEP: break
                for o in tape[t].get("market") or []:
                    if o and o[0] == "SELL" and len(o) >= 3: dump_soon[o[1]] = dump_soon.get(o[1], 0) + int(o[2])
        total = sum(max(0, v) for v in stock.values())
        extra = {}
        items = [i for i in r.PRODUCTS if i != "WHEAT" and stock.get(i, 0) - reserve.get(i, 0) > 0]
        items.sort(key=lambda i: -prices.get(i, 0) / BASE[i])
        for item in items:
            avail = stock[item] - reserve.get(item, 0); price = prices.get(item, 0)
            if price < 2: continue
            q = 0
            if item == "FERTILIZER": q = avail
            if item in dump_soon: q = max(q, min(avail, dump_soon[item]))
            if need > 0: q = max(q, min(avail, math.ceil(need / price))); 
            if total > P["shed_hi"]: q = max(q, min(avail, 4))
            q = min(q, avail)
            if q > 0:
                extra[item] = q; total -= q; need -= q * price
        orders = [["SELL", i, int(q)] for i, q in extra.items()]
        if orders and len(market) + len(orders) <= r.MAX_ORDERS:
            act["market"] = market + orders
        elif orders:  # over the cap: keep only the cash guard
            act["market"] = (market + orders)[: r.MAX_ORDERS]

    def _idle_work(self, act, farm, private=None, day=0):
        tiles = farm["tiles"]; units = [farm["farmer"], *farm["hands"]]
        invs = (private or {}).get("inventories") or []
        acts = [act.get("farmer") or ["PASS"], *(act.get("hands") or [])]
        while len(acts) < len(units): acts.append(["PASS"])
        for i, (pos, a) in enumerate(zip(units, acts)):
            if a and a[0] != "PASS": continue
            tile = tiles[int(pos[1])][int(pos[0])]
            if not isinstance(tile, dict): continue
            inv = invs[i] if i < len(invs) else {}
            if (FERTILIZE and tile.get("kind") == "PLANT" and tile.get("crop") in ("STRAWBERRY", "TOMATO", "MELON")
                    and tile.get("fertilized_until_day", -1) < day and inv.get("FERTILIZER", 0) > 0):
                acts[i] = ["FERTILIZE"]
            elif tile.get("kind") == "PLANT" and not tile.get("watered_today"): acts[i] = ["WATER"]
            elif "animal" in tile and not tile.get("cared_today"): acts[i] = ["CARE"]
        act["farmer"], act["hands"] = acts[0], acts[1:]


_LIVE = None


def agent(observation, configuration=None):
    global _LIVE
    if _LIVE is None: _LIVE = Live(Path(agent.__code__.co_filename).resolve().parent)
    return _LIVE.act(observation)
