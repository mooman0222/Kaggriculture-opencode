# ---------------------------------------------------------------------------
# HOLD layer (MMN0222, 2026-09-28): learned "do not sell this item today" gate on top of the chassis.
# Trained to imitate a hindsight oracle (tests/overlay_headroom.py --family hold --feat): at each day
# start (step 24d) a small MLP scores MILK/WOOL/STRAWBERRY from public prices/inventory history, own shed
# and own sales; the best-scoring item above the threshold has its SELL orders dropped for the day.
# Guards mirror the oracle's plans (97 games): at most one item per day, never the same item three days in a
# row, and no hold when the shed is nearly full (holding then only feeds drop_inventories).
# Inference is numpy only. Set HOLD_THR / HOLD_OFF env vars for A/B (defaults frozen below).
import os as _hold_os
import numpy as _hold_np

_HOLD_ITEMS = ("MILK", "WOOL", "STRAWBERRY")
_HOLD_ANIMALS = ("COW", "SHEEP", "GOOSE")
_HOLD_MODEL = __HOLD_MODEL__
_HOLD_THR = float(_hold_os.environ.get("HOLD_THR", __HOLD_THR__))
_HOLD_OFF = bool(_hold_os.environ.get("HOLD_OFF"))
_HOLD_PARENT = [v for v in list(globals().values()) if callable(v)][-1]
_HOLD_STATE = {}
_HOLD_MAX_RUN = 2
_HOLD_SHED_GUARD = 80
_HOLD_REPORT = dict(days=0, holds=0, dropped=0, guarded=0, errors=0)


def _hold_features(st, obs, t, me):
    farm = obs["farms"][me]; shed = obs["private"]["shed"]; d = t // 24
    an = {a: 0 for a in _HOLD_ANIMALS}
    for row in farm["tiles"]:
        for tile in row:
            if isinstance(tile, dict) and tile.get("animal") in an: an[tile["animal"]] += 1
    f = [float(d), float(farm["money"]), float(an["COW"]), float(an["SHEEP"]), float(an["GOOSE"])]
    y0 = t - 24
    for it in _HOLD_ITEMS:
        hist = [st["px"][u][it] for u in range(y0, t)]
        f += [st["px"][t][it], max(hist), min(hist), st["px"][y0][it], st["inv"][t][it], st["inv"][t][it] - st["inv"][y0][it],
              float(shed.get(it, 0)), float(sum(st["sold"].get(u, {}).get(it, 0) for u in range(y0, t)))]
    f += [float(st["run"].get(it, 0) > 0) for it in _HOLD_ITEMS]
    return _hold_np.array(f, dtype=_hold_np.float64)


def _hold_scores(x):
    m = _HOLD_MODEL
    h = (x - _hold_np.array(m["mean"])) / _hold_np.array(m["std"])
    for W, b in m["layers"][:-1]:
        h = _hold_np.maximum(0.0, h @ _hold_np.array(W) + _hold_np.array(b))
    W, b = m["layers"][-1]
    z = h @ _hold_np.array(W) + _hold_np.array(b)
    return 1.0 / (1.0 + _hold_np.exp(-z))


def hold_agent(observation, configuration=None):
    action = _HOLD_PARENT(observation, configuration)
    if _HOLD_OFF:
        return action
    try:
        t = int(observation["step"]); me = int(observation["player"])
        st = _HOLD_STATE.get(me)
        if st is None or t <= st["last"]:
            st = _HOLD_STATE[me] = dict(last=-1, px={}, inv={}, sold={}, hold=set(), run={})
            for k in _HOLD_REPORT: _HOLD_REPORT[k] = 0
        st["last"] = t
        prices = observation["market"]["prices"]; inventory = observation["market"]["inventory"]
        st["px"][t] = {it: float(prices.get(it, 0) or 0) for it in _HOLD_ITEMS}
        st["inv"][t] = {it: float(inventory.get(it, 0) or 0) for it in _HOLD_ITEMS}
        if t % 24 == 0 and 48 <= t < 696:
            _HOLD_REPORT["days"] += 1
            p = _hold_scores(_hold_features(st, observation, t, me))
            shed_total = sum(int(v) for v in observation["private"]["shed"].values())
            cands = [(pi, it) for it, pi in zip(_HOLD_ITEMS, p) if pi > _HOLD_THR and st["run"].get(it, 0) < _HOLD_MAX_RUN]
            if cands and shed_total >= _HOLD_SHED_GUARD:
                cands = []; _HOLD_REPORT["guarded"] += 1
            st["hold"] = {max(cands)[1]} if cands else set()
            st["run"] = {it: (st["run"].get(it, 0) + 1 if it in st["hold"] else 0) for it in _HOLD_ITEMS}
            _HOLD_REPORT["holds"] += len(st["hold"])
        market = [list(o) for o in (action.get("market") or [])]
        st["sold"][t] = {}
        for o in market:
            if o and o[0] == "SELL" and len(o) >= 2 and o[1] in _HOLD_ITEMS:
                st["sold"][t][o[1]] = st["sold"][t].get(o[1], 0) + (max(0, int(o[2])) if len(o) >= 3 else 1)
        if st["hold"] and t < 696:
            kept = [o for o in market if not (o and o[0] == "SELL" and len(o) >= 2 and o[1] in st["hold"])]
            if len(kept) != len(market):
                _HOLD_REPORT["dropped"] += len(market) - len(kept)
                for it in st["hold"]: st["sold"][t].pop(it, None)
                action = dict(action, market=kept)
    except Exception:
        _HOLD_REPORT["errors"] += 1
    return action


hold_agent.telemetry = _HOLD_REPORT
globals().pop("agent", None)
agent = hold_agent
