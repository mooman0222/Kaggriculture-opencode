"""E058 (farm+market base) + Transformer market SELL top-up (additive-only) eval.
Usage: PYTHONPATH=.venv/lib/python3.14/site-packages /usr/bin/python3 rl/play_mkt.py tmp/rl/mkt1_ep2.pt --games 16 [--vs agents/e058/main.py] [--cap 6] [--no-wheat]
Compares E058+MKT vs --vs opponent in kagsim, both seats.
"""
import argparse, os, sys, importlib.util, hashlib, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, kagsim
from features import encode, PRODUCTS, MKT_BUCKETS, unbucket, BASE_PRICE
from model2 import Policy2 as Policy

PXGATE = float(os.environ.get("MKT_PXGATE", "0"))
OPPGATE = int(os.environ.get("MKT_OPPGATE", "0"))
WOOLONLY = os.environ.get("MKT_WOOLONLY", "0") == "1"
PULL = os.environ.get("MKT_PULL", "0") == "1"
PULL_READY = int(os.environ.get("MKT_PULL_READY", "8"))
PULL_PX = float(os.environ.get("MKT_PULL_PX", "0.5"))
OPP_ITEMS = ("MILK", "WOOL", "EGG", "STRAWBERRY", "MELON")

SELL_OK = ("CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")


def load_py(p, tag):
    s = importlib.util.spec_from_file_location(tag, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


class HybridAgent:
    def __init__(self, ckpt, d=128, layers=3, dev="cpu", cap=6, no_wheat=True):
        self.dev = torch.device(dev)
        self.model = Policy(d=d, layers=layers).to(self.dev)
        if ckpt != "none":
            self.model.load_state_dict(torch.load(ckpt, map_location=self.dev))
        self.model.eval()
        self.e058 = load_py("agents/e058/main.py", "e058_hyb")
        self.cap = cap; self.no_wheat = no_wheat; self.state = {}

    def act(self, obs, seat):
        step = int(obs["step"])
        if step == 0: self.state = {}; self.e058._LIVE = None
        act = self.e058.agent(obs)
        if step <= 1 or step >= 718: return act
        try:
            return self._topup(act, obs, seat, step)
        except Exception:
            return act

    def _topup(self, act, obs, seat, step):
        live = self.e058._LIVE
        market = list(act.get("market") or [])
        planned = {}
        for o in market:
            if o and o[0] == "SELL" and len(o) >= 3 and o[1] in PRODUCTS:
                planned[o[1]] = planned.get(o[1], 0) + int(o[2])
        f = encode(obs, seat)
        prev = self.state.get("prev")
        if prev is None:
            import numpy as _np
            from features import MAX_UNITS
            prev = _np.zeros((MAX_UNITS, 2), dtype=_np.int16); prev[:, 0] = 100; prev[:, 1] = 44
        with torch.no_grad():
            H = self.model.encode(*(torch.from_numpy(f[k]).unsqueeze(0).to(self.dev) for k in ("tiles", "units", "items", "glob")),
                                  prev=torch.from_numpy(prev).unsqueeze(0).to(self.dev))
            sell = self.model.market(H)["sell"][0].argmax(-1).cpu().numpy()
        view = live._view(obs, seat); stock = live._stock(act, view)
        for o in market:
            if o and o[0] == "SELL" and len(o) >= 3 and o[1] in stock:
                stock[o[1]] -= int(o[2])
        prices = view.prices
        ready = {}
        if OPPGATE or PULL:
            for row in obs["farms"][1 - seat]["tiles"]:
                for tl in row:
                    if not isinstance(tl, dict): continue
                    if "animal" in tl:
                        yu = int(tl.get("yield_units", 0) or 0)
                        if yu > 0:
                            a = tl["animal"]; p = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}.get(a[0] if isinstance(a, list) else a)
                            if p: ready[p] = ready.get(p, 0) + min(yu, 4)
                    elif tl.get("kind") == "PLANT" and int(tl.get("yield_units", 0) or 0) > 0 and tl.get("crop") in ("STRAWBERRY", "MELON", "WHEAT", "CARROT"):
                        ready[tl["crop"]] = ready.get(tl["crop"], 0) + min(int(tl["yield_units"]), 4)
        extra = []
        for j, item in enumerate(PRODUCTS):
            if self.no_wheat and item == "WHEAT": continue
            if item not in SELL_OK: continue
            if WOOLONLY and item != "WOOL": continue
            want = int(unbucket(int(sell[j]), MKT_BUCKETS))
            add = want - planned.get(item, 0)
            if add <= 0: continue
            if int(prices.get(item, 0)) < 2: continue
            if PXGATE and prices.get(item, 0) < BASE_PRICE.get(item, 50) * PXGATE: continue
            if OPPGATE and item in OPP_ITEMS and ready.get(item, 0) < OPPGATE: continue
            add = min(add, max(0, int(stock.get(item, 0))), self.cap)
            if add > 0: extra.append(["SELL", item, int(add)])
        if extra and len(market) + len(extra) <= 10:
            act["market"] = market + extra
            if os.environ.get("MKT_DEBUG"):
                print(f"  step {step} extra {[(e[1], e[2], round(prices.get(e[1], 0), 1)) for e in extra]}", flush=True)
        if PULL:
            self._pullfwd(act, obs, seat, step, market, ready if OPPGATE or PULL else {}, prices)
        return act

    def _pullfwd(self, act, obs, seat, step, market, ready, prices):
        """Volume-neutral-ish front-run: if opponent is about to dump X (ready>=PULL_READY)
        and X still has a decent price, pull X's tape-planned sells (next 6 steps) forward to now."""
        from features import BASE_PRICE
        live = self.e058._LIVE
        tape = live._tape(seat)
        market_now = list(act.get("market") or [])
        if len(market_now) >= 10: return
        view = live._view(obs, seat); stock = live._stock(act, view)
        for o in market_now:
            if o and o[0] == "SELL" and len(o) >= 3 and o[1] in stock:
                stock[o[1]] -= int(o[2])
        for item in OPP_ITEMS:
            if ready.get(item, 0) < PULL_READY: continue
            if prices.get(item, 0) < BASE_PRICE.get(item, 50) * PULL_PX: continue
            if any(o and o[0] == "SELL" and len(o) > 1 and o[1] == item for o in market_now): continue
            soon = 0
            for k in range(1, 7):
                t = step + k
                if t >= 718 or t == 144 or t == 648: break
                for o in tape[t].get("market") or []:
                    if o and o[0] == "SELL" and len(o) >= 3 and o[1] == item: soon += int(o[2])
            if soon <= 0: continue
            q = min(soon, max(0, int(stock.get(item, 0))), self.cap)
            if q > 0:
                act["market"] = list(act.get("market") or []) + [["SELL", item, int(q)]]
                stock[item] -= q; market_now = list(act["market"])
                if len(market_now) >= 10: return


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--vs", default="agents/e058/main.py"); ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--cap", type=int, default=6); ap.add_argument("--no-wheat", action="store_true", default=True)
    ap.add_argument("--d", type=int, default=128); ap.add_argument("--layers", type=int, default=3)
    a = ap.parse_args()
    agent = HybridAgent(a.ckpt, a.d, a.layers, cap=a.cap, no_wheat=a.no_wheat)
    opp = load_py(a.vs, "opp_mkt"); res = []; t0 = time.time(); tmax = 0
    for g_i in range(a.games):
        seed = a.seed0 + g_i // 2; seat = g_i % 2
        for k in ("_LIVE",):
            if hasattr(opp, k): setattr(opp, k, None)
        g = kagsim.Game(seed)
        while not g.done:
            o = g.observe(seat); t1 = time.perf_counter(); x = agent.act(o, seat); tmax = max(tmax, time.perf_counter() - t1)
            y = opp.agent(g.observe(1 - seat))
            g.step(*((x, y) if seat == 0 else (y, x)))
        tel = g.telemetry(seat); res.append((g.reward(seat), g.reward(1 - seat)))
        print(f"seed {seed} seat {seat}: hyb {g.reward(seat):8.0f} opp {g.reward(1-seat):8.0f} margin {g.reward(seat)-g.reward(1-seat):+8.0f} sold {tel.get('sold_units')} rev {tel.get('sell_revenue')}", flush=True)
    print(f"mean hyb {np.mean([r[0] for r in res]):.0f} opp {np.mean([r[1] for r in res]):.0f} margin {np.mean([r[0]-r[1] for r in res]):+.0f} wins {sum(r[0]>r[1] for r in res)}/{len(res)} max step {tmax*1000:.0f} ms [{time.time()-t0:.0f}s]")
