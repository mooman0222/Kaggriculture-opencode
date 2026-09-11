"""Crop-swap layer: in worlds with tomato demand, the tape's mid-game PLANT WHEAT actions become PLANT TOMATO
on up to N tiles. The tape keeps watering and harvesting those tiles on its own schedule; tomato yields daily
for four days after day 8 and accumulates (max 4 held), so the tape's periodic HARVEST collects it. Tomato has a
steep scarcity price in such worlds (60 -> 150..300). Sales are metered from day 22."""
from __future__ import annotations
import os

P = {"d1": int(os.environ.get("CS_D1", "12")), "d2": int(os.environ.get("CS_D2", "17")), "tiles": int(os.environ.get("CS_TILES", "10")),
     "sell_day": int(os.environ.get("CS_SELL_DAY", "22")), "sell_px": int(os.environ.get("CS_SELL_PX", "120")), "sell_n": int(os.environ.get("CS_SELL_N", "6")),
     "triggers": tuple(os.environ.get("CS_TRIG", "PIZZA_SHOP,FARMERS_MARKET").split(",")), "trig_n": int(os.environ.get("CS_TRIG_N", "4"))}


class CropSwap:
    def __init__(self): self.on = None; self.swapped = 0; self.need_seed = 0

    def apply(self, act, obs):
        step, seat = int(obs["step"]), int(obs["player"]); day, hour = step // 24, step % 24
        if step == 0: self.on = None; self.swapped = 0; self.need_seed = 0
        if day < P["d1"]: return act
        if self.on is None:
            shops = obs["town"]["unlocked_shops"]
            self.on = sum(1 for s in shops[: P["trig_n"]] if s in P["triggers"]) >= 1
        if not self.on: return act
        market = list(act["market"])
        seeds = obs["private"]["seeds"]
        if P["d1"] <= day <= P["d2"] and self.swapped < P["tiles"]:
            # buy tomato seeds ahead: at dawn, enough for today's expected swaps
            if hour == 0 and seeds.get("TOMATO", 0) < 4 and len(market) < 10:
                market.append(["BUY_SEED", "TOMATO", min(6, P["tiles"] - self.swapped)])
            have = seeds.get("TOMATO", 0)
            units = [act.get("farmer") or ["PASS"], *(act.get("hands") or [])]
            for i, u in enumerate(units):
                if have <= 0 or self.swapped >= P["tiles"]: break
                if u and u[0] == "PLANT" and u[1] == "WHEAT":
                    units[i] = ["PLANT", "TOMATO"]; have -= 1; self.swapped += 1
            act["farmer"], act["hands"] = units[0], units[1:]
        shed_t = obs["private"]["shed"].get("TOMATO", 0); px = obs["market"]["prices"].get("TOMATO", 0)
        if shed_t > 0 and len(market) < 10 and not any(o and o[0] == "SELL" and o[1] == "TOMATO" for o in market):
            if day >= 28: market.append(["SELL", "TOMATO", shed_t])
            elif day >= P["sell_day"] and px >= P["sell_px"]: market.append(["SELL", "TOMATO", min(shed_t, P["sell_n"])])
        act["market"] = market[:10]
        return act
