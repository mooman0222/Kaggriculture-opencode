"""E057a: GA-evolved 0909 tapes (E056 base) + general adaptive seller (live_a).

Farm = GA tapes (13x719, shop-routed). Market = tape sells kept, plus live
extras: cash guard for upcoming purchases, clone pre-dump (K=6), shed-pressure
relief, fertilizer ASAP. Additive-only: never removes tape sells (except WHEAT
feed management stays with tape). Non-clone output differs from GA base only
by the extras; clone games get front-run.
"""
from __future__ import annotations

import copy
import importlib.util
import math
import os
from pathlib import Path

P = {
    "shed_hi": int(os.environ.get("LA_SHED", "80")),
    "lead": int(os.environ.get("LA_LEAD", "6")),
}
BASE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}
IDLE_WORK = os.environ.get("SR_IDLE", "1") == "1"
FERTILIZE = os.environ.get("LA_FERT", "1") == "1"
OHFR = os.environ.get("LD_OHFR", "1") == "1"
OHFR_MIN = int(os.environ.get("LD_OHFR_MIN", "4"))
GATE_STEP = 72
ANIMAL_PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


class Live:
    def __init__(self, folder):
        self.r = _load(folder / "router.py", "live_d_router")
        self.policy = self.r.Policy(folder)
        self.r.advance_sales = lambda *a, **k: None
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
        if step >= self.r.LAST_STEP: return act
        self._sell(act, obs, step)
        if OHFR: self._opp_front_run(act, obs, step)
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
        for o in market:
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
            if need > 0: q = max(q, min(avail, math.ceil(need / price)))
            if total > P["shed_hi"]: q = max(q, min(avail, 4))
            q = min(q, avail)
            if q > 0:
                extra[item] = q; total -= q; need -= q * price
        orders = [["SELL", i, int(q)] for i, q in extra.items()]
        if orders and len(market) + len(orders) <= r.MAX_ORDERS:
            act["market"] = market + orders
        elif orders:
            act["market"] = (market + orders)[: r.MAX_ORDERS]

    def _opp_front_run(self, act, obs, step):
        """Opponent-adaptive sell lead: if the opponent's public farm holds ripe
        harvest (animals with yield / ripe crops), they will dump product X in
        the next turns and crash its price. Sell our X now, ahead of them.
        Additive-only, capped by real stock, price floor, and order cap."""
        r = self.r
        seat = int(obs["player"])
        try:
            opp_tiles = obs["farms"][1 - seat]["tiles"]
        except Exception:
            return
        ready = {}
        for row in opp_tiles:
            for tl in row:
                if not isinstance(tl, dict):
                    continue
                if "animal" in tl:
                    yu = int(tl.get("yield_units", 0) or 0)
                    if yu > 0:
                        prod = ANIMAL_PRODUCT.get(tl["animal"][0] if isinstance(tl.get("animal"), list) else tl.get("animal"))
                        if prod:
                            ready[prod] = ready.get(prod, 0) + min(yu, 4)
                elif tl.get("kind") == "PLANT" and int(tl.get("yield_units", 0) or 0) > 0:
                    crop = tl.get("crop")
                    if crop in ("STRAWBERRY", "MELON", "WHEAT", "CARROT"):
                        ready[crop] = ready.get(crop, 0) + min(int(tl["yield_units"]), 4)
        if not ready:
            return
        view = r.FarmView(obs)
        stock = r.projected_shed(act, view)
        for o in act["market"]:
            if o and o[0] == "SELL" and len(o) >= 3 and o[1] in stock:
                stock[o[1]] -= int(o[2])
        for item in ("MILK", "WOOL", "EGG", "STRAWBERRY", "MELON"):
            op = ready.get(item, 0)
            if op < OHFR_MIN:
                continue
            avail = max(0, int(stock.get(item, 0)))
            px = int(view.prices.get(item, 0))
            if avail <= 0 or px < 2:
                continue
            if px < BASE.get(item, 50) * 0.4 and step < r.LAST_STEP - 24:
                continue  # crashed already: too late to front-run, let town demand recover
            if step % 4 == 0 and op < OHFR_MIN * 2 and step < r.LAST_STEP - 24:
                continue  # pre-tick trough: wait one turn for demand to lift price unless dump is imminent
            if any(o and o[0] == "SELL" and len(o) > 1 and o[1] == item for o in act["market"]):
                continue  # already selling X this turn (tape or seller extras)
            if len(act["market"]) >= r.MAX_ORDERS:
                break
            q = min(avail, op, 4)  # small lots: front-run without crashing our own market in one dump
            if q > 0:
                act["market"] = list(act["market"]) + [["SELL", item, int(q)]]
                stock[item] -= q

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
