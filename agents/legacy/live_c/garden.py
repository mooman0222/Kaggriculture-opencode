"""Garden layer: in worlds with tomato demand (PIZZA_SHOP / FARMERS_MARKET among the first shops), buy the
fourth quadrant at day 12, hire extra hands after the tape's own hires, and run a small live planner on the
new tiles only: plant tomatoes, water, harvest. Tomato has a steep scarcity price (60 -> 300 as the town
drains 400 units) and the tape never grows it. Sales are metered so the scarcity premium is not dumped away."""
from __future__ import annotations
import os

SE = [(x, y) for y in range(5, 10) for x in range(5, 10)]
SHED = [(4, 4), (5, 4), (4, 5), (5, 5)]
P = {"day": int(os.environ.get("LC_DAY", "15")), "hands": int(os.environ.get("LC_HANDS", "4")), "hire_hour": int(os.environ.get("LC_HIRE_H", "3")),
     "seeds": int(os.environ.get("LC_SEEDS", "16")), "min_money": int(os.environ.get("LC_MONEY", "5500")),
     "sell_from_day": int(os.environ.get("LC_SELL_DAY", "22")), "sell_px": int(os.environ.get("LC_SELL_PX", "120")), "sell_per_turn": int(os.environ.get("LC_SELL_N", "6")),
     "triggers": tuple(os.environ.get("LC_TRIG", "PIZZA_SHOP,FARMERS_MARKET").split(",")), "trigger_shops": int(os.environ.get("LC_TRIG_N", "4"))}


def dist(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step_toward(pos, tgt):
    x, y = pos
    if tgt[0] != x: return ["EAST" if tgt[0] > x else "WEST"]
    if tgt[1] != y: return ["SOUTH" if tgt[1] > y else "NORTH"]
    return None


class Garden:
    def __init__(self): self.on = None; self.target = {}

    def apply(self, act, obs):
        step, seat = int(obs["step"]), int(obs["player"]); day, hour = step // 24, step % 24
        farm = obs["farms"][seat]; tiles = farm["tiles"]; shops = obs["town"]["unlocked_shops"]
        if step == 0: self.on = None; self.target = {}
        if day < P["day"]: return act
        if self.on is None:
            self.on = any(s in P["triggers"] for s in shops[: P["trigger_shops"]]) and farm["money"] >= P["min_money"] and len(farm["unlocked_quadrants"]) == 3
        if not self.on: return act
        market = list(act["market"])
        if day == P["day"] and hour == 1 and len(market) <= 8:
            market += [["BUY_LAND"], ["BUY_SEED", "TOMATO", P["seeds"]]]
        if hour == P["hire_hour"] and day <= 27 and len(market) + P["hands"] <= 10:
            market += [["HIRE"] for _ in range(P["hands"])]
        # --- planner for the extra hands (indices beyond the tape's hand actions)
        tape_hands = len(act.get("hands") or []); real = farm["hands"]
        extra = list(range(tape_hands, len(real)))
        acts = list(act.get("hands") or [])
        if extra and len(farm["unlocked_quadrants"]) == 4:
            seeds = obs["private"]["seeds"].get("TOMATO", 0)
            tasks = {}
            for (x, y) in SE:
                tl = tiles[y][x]
                if tl is None and seeds > 0 and day <= 20: tasks[(x, y, "PLANT")] = 1; seeds -= 1
                elif isinstance(tl, dict) and tl.get("kind") == "PLANT":
                    if tl.get("yield_units", 0) > 0: tasks[(x, y, "HARVEST")] = 0
                    if not tl.get("watered_today"): tasks[(x, y, "WATER")] = 0 if hour >= 14 else 1
                elif isinstance(tl, dict) and tl.get("kind") == "WEED": tasks[(x, y, "DIG")] = 2
            invs = obs["private"]["inventories"]
            for i in extra:
                inv = invs[i + 1] if i + 1 < len(invs) else {}
                if inv.get("TOMATO", 0) >= 3 or (inv.get("TOMATO", 0) > 0 and hour >= 21):
                    tasks[("DROP", i)] = -1  # per-unit task: bring the harvest home now
            if hour <= P["hire_hour"] + 1: self.target = {}
            assigned = set(); out = {}
            for i in list(self.target):
                k = self.target[i]
                if i not in extra or k not in tasks or k in assigned: self.target.pop(i, None); continue
                assigned.add(k)
            def tgt(k, i):
                return min(SHED, key=lambda sh: dist(tuple(real[i]), sh)) if k[0] == "DROP" else k[:2]
            pairs = sorted(((dist(tuple(real[i]), tgt(k, i)) + 4 * tier, i, str(k), k) for i in extra if i not in self.target
                            for k, tier in tasks.items() if k not in assigned and (k[0] != "DROP" or k[1] == i)), key=lambda x: x[:3])
            for _, i, _s, k in pairs:
                if i in self.target or k in assigned: continue
                self.target[i] = k; assigned.add(k)
            for i in extra:
                k = self.target.get(i); pos = tuple(real[i])
                if k is None:
                    # idle: carry produce home so it can sell today
                    inv = obs["private"]["inventories"][i + 1] if i + 1 < len(obs["private"]["inventories"]) else {}
                    if inv.get("TOMATO", 0) > 0:
                        mv = None if pos in SHED else step_toward(pos, min(SHED, key=lambda s: dist(pos, s)))
                        out[i] = mv or ["DROP"]
                    else: out[i] = ["PASS"]
                    continue
                if k[0] == "DROP":
                    mv = None if pos in SHED else step_toward(pos, min(SHED, key=lambda sh: dist(pos, sh)))
                    out[i] = mv or ["DROP"]
                else:
                    mv = step_toward(pos, k[:2])
                    out[i] = mv if mv else (["PLANT", "TOMATO"] if k[2] == "PLANT" else [k[2]])
                if not mv: self.target.pop(i, None)
            while len(acts) < len(real): acts.append(["PASS"])
            for i, a in out.items(): acts[i] = a
        # --- metered tomato sales
        shed_t = obs["private"]["shed"].get("TOMATO", 0); px = obs["market"]["prices"].get("TOMATO", 0)
        if shed_t > 0 and len(market) < 10:
            if day >= 28 or step >= 700: market.append(["SELL", "TOMATO", shed_t])
            elif day >= P["sell_from_day"] and px >= P["sell_px"]: market.append(["SELL", "TOMATO", min(shed_t, P["sell_per_turn"])])
        act["hands"] = acts; act["market"] = market[:10]
        return act
