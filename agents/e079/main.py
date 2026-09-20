"""E079: hcounter 素体 + buy-mirror 層。

Dude 局 (E076 で破棄 17) の機構: h23 の BUY 流入 (+27) が shed 空の夜に overflow
(+17) を作る。素体の storage guard は BUY 流入項を持たないため不発・棄権する。
本層は h22/h23 のみ、テープの BUY_PRODUCT と同量の SELL を同 step に追加する
wash (net≈0) で買いの取消し相当を行い、空棚型 overflow を塞ぐ。
- テープの SELL・順序は不変 (append のみ、10 件 cap で切られる側)
- WHEAT は飼料 reserve (2 日分) を残して mirror (給餌連鎖の保護)
- 単価 2 未満は対象外。発火は buy-overflow 夜のみ (jojo 全手で発火 0 を確認)
- テープ売却・開幕・thin・クランプ・スロット入替・h00 先行には触らない
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

MIRROR = os.environ.get("LA_BUYMIRROR", "1") == "1"
FEED_DAYS = int(os.environ.get("LA_BUYMIRROR_FEEDDAYS", "2"))  # 小麦 reserve 日数


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _live_fn(mod):
    # 土台の live は末尾 callable (hcounter は agent が末尾で同一のはずだが将来の罠に備える)
    fn = mod.agent
    for v in vars(mod).values():
        if callable(v):
            fn = v
    return fn


class Live:
    def __init__(self, folder):
        self.b = _load(folder / "base.py", "e079_base")
        self.live = _live_fn(self.b)

    def act(self, obs):
        act = self.live(obs)
        if not MIRROR:
            return act
        step = int(obs.get("step", -1))
        try:
            if step % 24 not in (22, 23) or step >= 717 or step < 0:
                return act
            private = obs.get("private") or {}
            prices = (obs.get("market") or {}).get("prices") or {}
            shed = private.get("shed") or {}
            invs = private.get("inventories") or []
            shed_tot = sum(max(0, int(v or 0)) for v in shed.values())
            carried = {}
            for inv in invs:
                for k, v in (inv or {}).items():
                    n = int(v or 0)
                    if n > 0:
                        carried[k] = carried.get(k, 0) + n
            carried_tot = sum(carried.values())
            market = [list(o) for o in (act.get("market") or [])]
            buys = {}
            for o in market:
                if o and o[0] == "BUY_PRODUCT" and len(o) >= 3:
                    buys[o[1]] = buys.get(o[1], 0) + max(0, int(o[2]))
            inflow = sum(buys.values())
            if inflow <= 0 or shed_tot + carried_tot - 99 + inflow <= 0:
                return act
            need = shed_tot + carried_tot - 99 + inflow  # >0: 塞ぐべき overflow 量
            # 飼料 reserve: 小麦は (動物数 × 日数) を残す
            farm = obs["farms"][int(obs["player"])]
            animals = 0
            for row in farm["tiles"]:
                for tl in row:
                    if isinstance(tl, dict) and "animal" in tl:
                        animals += 1
            feed_keep = animals * FEED_DAYS
            wheat_avail = int(shed.get("WHEAT", 0) or 0) + carried.get("WHEAT", 0)
            added = []
            for item, q in buys.items():
                if need <= 0 or len(market) + len(added) >= 10:
                    break
                if int(prices.get(item, 0) or 0) < 2:
                    continue
                qq = min(int(q), int(need))
                if item == "WHEAT":
                    qq = min(qq, max(0, wheat_avail - feed_keep))
                if qq > 0:
                    added.append(["SELL", item, int(qq)])
                    need -= qq
                    if item == "WHEAT":
                        wheat_avail -= qq
            if added:
                act = dict(act)
                act["market"] = market + added
        except Exception:
            pass
        return act


_LIVE = None


def agent(observation, configuration=None):
    global _LIVE
    if _LIVE is None:
        _LIVE = Live(Path(agent.__code__.co_filename).resolve().parent)
    return _LIVE.act(observation)
