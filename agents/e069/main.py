"""E069: E068 の夜明けガードを**シャシー非依存**に書き直したもの。 E068 版は base.py の私有名 (`_View` / `chassis._projected_shed`) に依存しており、 公開シャシーが新版になると壊れうる。こちらは生の observation だけから棚を射影するので、 土台を差し替えても main.py の夜明けガードはそのまま載る。中身の規則は E068 と同じ。"""
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

P = {"shed_hi": int(os.environ.get("LA_SHED", "80")), "lead": int(os.environ.get("LA_LEAD", "6")), "lead_boost": int(os.environ.get("LA_LEAD_BOOST", "12"))}
BASE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}
IDLE_WORK = os.environ.get("SR_IDLE", "1") == "1"
FERTILIZE = os.environ.get("LA_FERT", "1") == "1"
SELLER = os.environ.get("LD_SELL", "1") == "1"
OHFR = os.environ.get("LD_OHFR", "0") == "1"
PHASE_D0 = int(os.environ.get("LA_PHASE_D0", "99"))  # town consumes at h%4==0 after orders fill: a SELL placed then is quoted pre-consumption. Hold it one hour. 99 = off
OHFR_MIN = int(os.environ.get("LD_OHFR_MIN", "4"))
SELL_FIRST = os.environ.get("LA_SELLFIRST", "1") == "1"  # market slots resolve in order with a price refresh between them: a SELL behind a BUY/HIRE lets the rival's slot-0 SELL take the higher quote (E041)
GATE_STEP = 72
LAST_STEP, ROUTE_STEP, FINAL_PLAN_STEP, MAX_ORDERS = 718, 144, 648, 10
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
ANIMAL_PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


DAWN = int(os.environ.get("LA_DAWN", "22"))   # 夜明けガードの発動時刻 (0 = off)。22 が最良 (20/23 は劣る)
SHED_CAP = 100



class Live:
    def __init__(self, folder):
        self.b = _load(folder / "base.py", "e069_base")
        self.reset()

    def reset(self):
        self.prev_inv = None; self.prev_sells = {}; self.boost = False
        self.held = {}
        self.mismatch = 0; self.clone = False

    # -- accessors into the base chassis --
    def _route(self, seat):
        st = self.b._IMPL.chassis.players.get(seat) or {}
        r = st.get("route"); return r if r in self.b._IMPL.chassis.routes else 0

    def _tape(self, seat):
        return self.b._IMPL.chassis.routes[self._route(seat)]

    def _view(self, obs, seat):
        return self.b._View(obs, seat, self.b._IMPL.chassis.cfg)

    def _stock(self, act, view):
        return self.b._IMPL.chassis._projected_shed(act, view)

    def act(self, obs):
        step, seat = int(obs["step"]), int(obs["player"])
        if step == 0: self.reset()
        farm = obs["farms"][seat]
        if 0 < step <= GATE_STEP:
            opp = obs["farms"][1 - seat]
            if [farm["farmer"], *farm["hands"]] != [opp["farmer"], *opp["hands"]]: self.mismatch += 1
            if step == GATE_STEP: self.clone = self.mismatch <= 2
        act = self.b.agent(obs)
        if step >= LAST_STEP or step <= 1: return act
        try:
            if SELLER: self._sell(act, obs, step)
            if DAWN: self._dawn_guard(act, obs, step)
            if OHFR: self._opp_front_run(act, obs, step)
            if IDLE_WORK: self._idle_work(act, farm, obs.get("private"), step // 24)
            self._phase(act, step)
            if SELL_FIRST: act["market"] = sorted(list(act.get("market") or []), key=lambda o: 0 if (o and o[0] == "SELL") else 1)[:MAX_ORDERS]
        except Exception:
            pass
        try:
            self.prev_sells = {}
            for o in act.get("market") or []:
                if o and o[0] == "SELL" and len(o) >= 3: self.prev_sells[o[1]] = self.prev_sells.get(o[1], 0) + int(o[2])
        except Exception:
            pass
        return act

    def _phase(self, act, step):
        """Release sells deferred from the previous hour (SELL-first), then defer this hour's sells if it is a tick hour in the endgame."""
        market = list(act.get("market") or [])
        held = getattr(self, "held", None) or {}
        if held:  # merge by item, SELL first, but never drop the base's non-SELL orders (purchases feed the tape's cash chain)
            sells = dict(held); others = []
            for o in market:
                if o and o[0] == "SELL" and len(o) >= 3: sells[o[1]] = sells.get(o[1], 0) + int(o[2])
                else: others.append(o)
            sell_orders = [["SELL", i, int(q)] for i, q in sells.items() if q > 0][:max(0, MAX_ORDERS - len(others))]
            market = sell_orders + others; self.held = {}
        if step // 24 >= PHASE_D0 and step % 4 == 0 and step < LAST_STEP - 1:
            keep = []; held = {}
            for o in market:
                if o and o[0] == "SELL" and len(o) >= 3: held[o[1]] = held.get(o[1], 0) + int(o[2])
                else: keep.append(o)
            market = keep; self.held = held
        act["market"] = market[:MAX_ORDERS]

    @staticmethod
    def _projected_shed_raw(act, obs, seat):
        # base.py の _projected_shed と同じ射影を、生の観測だけから作る (シャシー非依存)。
        # この手のユニット行動を織り込んだ棚: PICKUP は減らし、小屋隣接での DROP / PLACE(非動物) は容量まで足す。
        private = obs.get("private") or {}
        proj = {k: int(v or 0) for k, v in (private.get("shed") or {}).items()}
        invs = private.get("inventories") or []
        farm = obs["farms"][seat]
        units = [farm["farmer"], *(farm.get("hands") or [])]
        board = len(farm["tiles"])
        half = board // 2
        total = sum(max(0, v) for v in proj.values())
        acts = [act.get("farmer") or ["PASS"], *(act.get("hands") or [])]
        for i, pos in enumerate(units):
            if not (int(pos[0]) in (half - 1, half) and int(pos[1]) in (half - 1, half)):
                continue
            a = acts[i] if i < len(acts) else ["PASS"]
            op = a[0] if a else "PASS"
            inv = invs[i] if i < len(invs) else {}
            if op == "PICKUP" and len(a) >= 2 and a[1] in proj:
                q = min(proj[a[1]], max(0, int(a[2]) if len(a) >= 3 else 1))
                proj[a[1]] -= q; total -= q
            elif op == "DROP":
                for item, held in (inv or {}).items():
                    take = min(max(0, int(held or 0)), max(0, SHED_CAP - total))
                    if take > 0: proj[item] = proj.get(item, 0) + take; total += take
            elif op == "PLACE" and len(a) >= 2 and a[1] not in ANIMAL_PRODUCT:
                take = min(max(0, int(a[2]) if len(a) >= 3 else 1),
                           max(0, int((inv or {}).get(a[1], 0) or 0)), max(0, SHED_CAP - total))
                if take > 0: proj[a[1]] = proj.get(a[1], 0) + take; total += take
        return proj

    def _dawn_guard(self, act, obs, step):
        # 日跨ぎでユニットの手持ちは shed に移されるが、容量 100 を超えた分は破棄される
        # (sim.hpp drop_inventories)。実測で 6.5 個/局・約 $527 が消えていた。
        # 破棄される見込みの分だけ、shed から高値の品を先に売って場所を空ける。市場側だけの追加。
        hour = step % 24
        if not DAWN or hour < DAWN or hour > 23:
            return
        seat = int(obs["player"])
        private = obs.get("private") or {}
        prices = (obs.get("market") or {}).get("prices") or {}
        shed = self._projected_shed_raw(act, obs, seat)
        held = {}
        for inv in (private.get("inventories") or []):
            for k, v in (inv or {}).items():
                n = int(v or 0)
                if n > 0: held[k] = held.get(k, 0) + n
        market = list(act.get("market") or [])
        for o in market:
            if o and o[0] == "SELL" and len(o) >= 3 and o[1] in shed: shed[o[1]] -= int(o[2])
        total = sum(max(0, v) for v in shed.values())
        overflow = sum(held.values()) - max(0, SHED_CAP - total)
        if overflow <= 0:
            return
        items = sorted((i for i in PRODUCTS if shed.get(i, 0) > 0),
                       key=lambda i: (i == "WHEAT", -prices.get(i, 0) / BASE[i]))
        added = []
        for item in items:
            if overflow <= 0 or len(market) + len(added) >= MAX_ORDERS: break
            if prices.get(item, 0) < 2: continue
            q = min(shed[item], overflow)
            if q > 0: added.append(["SELL", item, int(q)]); overflow -= q
        if added:
            act["market"] = (added + market)[:MAX_ORDERS]   # SELL は前スロット (E041)

    def _sell(self, act, obs, step):
        seat = int(obs["player"]); tape = self._tape(seat)
        market = list(act["market"])
        view = self._view(obs, seat); prices = view.prices
        stock = self._stock(act, view)
        for o in market:
            if o and o[0] == "SELL" and len(o) >= 3 and o[1] in stock: stock[o[1]] -= int(o[2])
        reserve = {}; cost = 0
        for k in range(1, 9):
            t = step + k
            if t >= LAST_STEP: break
            tt = self.b._IMPL.chassis.routes[2][t] if t >= FINAL_PLAN_STEP else tape[t]
            for u in [tt.get("farmer") or [], *(tt.get("hands") or [])]:
                if u and u[0] == "PICKUP" and len(u) >= 2: reserve[u[1]] = reserve.get(u[1], 0) + (int(u[2]) if len(u) >= 3 else 1)
            for o in tt.get("market") or []:
                if not o: continue
                if o[0] == "BUY_ANIMAL": cost += ANIMAL_COST[o[1]] * int(o[2])
                elif o[0] == "BUY_LAND": cost += 2000
                elif o[0] == "BUY_SEED": cost += SEED_COST[o[1]] * int(o[2])
                elif o[0] == "BUY_PRODUCT": cost += prices.get(o[1], 30) * int(o[2])
        money = float(obs["farms"][seat]["money"])
        for o in market:
            if not o: continue
            if o[0] == "SELL" and len(o) >= 3: money += prices.get(o[1], 0) * min(int(o[2]), max(0, stock.get(o[1], 0) + int(o[2])))
            elif o[0] == "BUY_ANIMAL": money -= ANIMAL_COST[o[1]] * int(o[2])
        need = max(0.0, cost - money)
        dump_soon = {}
        inv = dict(obs["market"].get("inventory") or {})
        if self.clone and self.prev_inv is not None and not self.boost:
            # identical farms: the rival is ahead of us if the market took in product p last step while we sold none of it and still hold it
            for p_, v in inv.items():
                if self.prev_sells.get(p_, 0) == 0 and v - self.prev_inv.get(p_, v) > 0 and stock.get(p_, 0) > 0: self.boost = True; break
        self.prev_inv = inv
        lead = P["lead_boost"] if self.boost else P["lead"]
        if self.clone and lead:
            for k in range(1, lead + 1):
                t = step + k
                if t >= LAST_STEP or t == ROUTE_STEP or t == FINAL_PLAN_STEP: break
                for o in tape[t].get("market") or []:
                    if o and o[0] == "SELL" and len(o) >= 3: dump_soon[o[1]] = dump_soon.get(o[1], 0) + int(o[2])
        total = sum(max(0, v) for v in stock.values())
        extra = {}
        items = [i for i in PRODUCTS if i != "WHEAT" and stock.get(i, 0) - reserve.get(i, 0) > 0]
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
        if orders:
            act["market"] = (market + orders)[:MAX_ORDERS]

    def _opp_front_run(self, act, obs, step):
        seat = int(obs["player"])
        opp_tiles = obs["farms"][1 - seat]["tiles"]
        ready = {}
        for row in opp_tiles:
            for tl in row:
                if not isinstance(tl, dict): continue
                if "animal" in tl:
                    yu = int(tl.get("yield_units", 0) or 0)
                    if yu > 0:
                        a = tl["animal"]; prod = ANIMAL_PRODUCT.get(a[0] if isinstance(a, list) else a)
                        if prod: ready[prod] = ready.get(prod, 0) + min(yu, 4)
                elif tl.get("kind") == "PLANT" and int(tl.get("yield_units", 0) or 0) > 0:
                    crop = tl.get("crop")
                    if crop in ("STRAWBERRY", "MELON", "WHEAT", "CARROT"): ready[crop] = ready.get(crop, 0) + min(int(tl["yield_units"]), 4)
        if not ready: return
        view = self._view(obs, seat); stock = self._stock(act, view)
        for o in act["market"]:
            if o and o[0] == "SELL" and len(o) >= 3 and o[1] in stock: stock[o[1]] -= int(o[2])
        for item in ("MILK", "WOOL", "EGG", "STRAWBERRY", "MELON"):
            op = ready.get(item, 0)
            if op < OHFR_MIN: continue
            avail = max(0, int(stock.get(item, 0))); px = int(view.prices.get(item, 0))
            if avail <= 0 or px < 2: continue
            if px < BASE.get(item, 50) * 0.4 and step < LAST_STEP - 24: continue
            if step % 4 == 0 and op < OHFR_MIN * 2 and step < LAST_STEP - 24: continue
            if any(o and o[0] == "SELL" and len(o) > 1 and o[1] == item for o in act["market"]): continue
            if len(act["market"]) >= MAX_ORDERS: break
            q = min(avail, op, 4)
            if q > 0:
                act["market"] = list(act["market"]) + [["SELL", item, int(q)]]; stock[item] -= q

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
