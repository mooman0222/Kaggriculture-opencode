"""Bounded overlays for the inventory-timing/crop-value experiment.

The frozen Seven-Turn Rescue bundle remains byte-for-byte unchanged.  Every
entry point creates a fresh frozen parent and this wrapper changes only the
declared treatment.  Sale credits are settled on the original tape step so a
front-run never turns into an accidental duplicate sale.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any


def _submission_root():
    filename = globals().get("__file__") or _submission_root.__code__.co_filename
    return Path(filename).resolve().parent


class FrozenBaseFactory:
    """Submission-local replacement for the development hash registry."""

    @classmethod
    def from_registry(cls):
        return cls()

    def new_instance(self):
        import importlib.util
        from uuid import uuid4

        path = _submission_root() / "base_main.py"
        name = "_seven_turn_t4_base_" + uuid4().hex
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module.agent


MAX_ORDERS = 10
SALE_ITEMS = ("STRAWBERRY", "MILK", "WOOL", "MELON", "EGG", "TOMATO")
SALE_START = 336
SALE_RAW_END = 647
MIN_FRONT_RUN_UNITS = 4

# A crop substitution is deliberately much narrower than a general replanner.
# Tomato is ongoing like strawberry, so the parent's later WATER/HARVEST route
# stays meaningful.  Wheat is not substituted here: after its one harvest the
# replay route would leave the square empty, which would confound the test.
CROP_START = 264
CROP_END = 575
LOW_STRAWBERRY_PRICE = 8
LOW_PRICE_STREAK = 12
MAX_CROP_SWAPS = 2


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _step(observation: Any) -> int:
    return int(_get(observation, "step", 0) or 0)


def _player(observation: Any) -> int:
    return int(_get(observation, "player", 0) or 0)


def _farm(observation: Any, seat: int) -> Any:
    return list(_get(observation, "farms", []) or [])[seat]


def _tile_signature(tile: Any) -> tuple[Any, ...]:
    return (
        _get(tile, "kind"),
        _get(tile, "crop"),
        _get(tile, "animal"),
    )


def public_similarity(observation: Any) -> float:
    """Physical-route similarity without using private opponent information."""
    farms = list(_get(observation, "farms", []) or [])
    if len(farms) != 2:
        return 0.0
    left, right = farms
    if len(list(_get(left, "hands", []) or [])) != len(list(_get(right, "hands", []) or [])):
        return 0.0
    if len(list(_get(left, "unlocked_quadrants", []) or [])) != len(
        list(_get(right, "unlocked_quadrants", []) or [])
    ):
        return 0.0
    left_tiles = [tile for row in list(_get(left, "tiles", []) or []) for tile in row]
    right_tiles = [tile for row in list(_get(right, "tiles", []) or []) for tile in row]
    if not left_tiles or len(left_tiles) != len(right_tiles):
        return 0.0
    matches = sum(
        _tile_signature(a) == _tile_signature(b) for a, b in zip(left_tiles, right_tiles)
    )
    return matches / len(left_tiles)


def _market(observation: Any) -> Any:
    return _get(observation, "market", {}) or {}


def _market_inventory(observation: Any) -> dict[str, int]:
    raw = _get(_market(observation), "inventory", {}) or {}
    return {str(item): int(quantity or 0) for item, quantity in raw.items()}


def _prices(observation: Any) -> dict[str, int]:
    raw = _get(_market(observation), "prices", {}) or {}
    return {str(item): int(price or 0) for item, price in raw.items()}


def _private(observation: Any) -> Any:
    return _get(observation, "private", {}) or {}


def _shed(observation: Any) -> dict[str, int]:
    raw = _get(_private(observation), "shed", {}) or {}
    return {str(item): max(0, int(quantity or 0)) for item, quantity in raw.items()}


def _seeds(observation: Any) -> dict[str, int]:
    raw = _get(_private(observation), "seeds", {}) or {}
    return {str(item): max(0, int(quantity or 0)) for item, quantity in raw.items()}


def _commands(action: dict[str, Any]) -> list[list[Any]]:
    return [action.get("farmer") or ["PASS"], *(action.get("hands") or [])]


def _baseline_sale_step(raw_step: int) -> int:
    previous = raw_step - 1
    eligible = raw_step % 72 != 0 and previous % 4 != 0
    return previous if eligible else raw_step


def _planned_sales(tape: list[dict[str, Any]], raw_step: int) -> dict[str, int]:
    planned: dict[str, int] = defaultdict(int)
    if not (0 <= raw_step < len(tape)):
        return {}
    for order in tape[raw_step].get("market") or []:
        if (
            isinstance(order, list)
            and len(order) >= 3
            and order[0] == "SELL"
            and order[1] in SALE_ITEMS
        ):
            quantity = max(0, int(order[2]))
            # Values 100/1000 are stock-clearing sentinels, not an inventory
            # batch forecast.  They are outside this timing experiment.
            if 0 < quantity < 100:
                planned[str(order[1])] += quantity
    return dict(planned)


class ExperimentAgent:
    def __init__(self, *, sale_mode: str = "none", crop_guard: bool = False):
        if sale_mode not in {"none", "lead_one", "adaptive_lead_two"}:
            raise ValueError(f"unsupported sale mode: {sale_mode}")
        self.sale_mode = sale_mode
        self.crop_guard = bool(crop_guard)
        self.parent = FrozenBaseFactory.from_registry().new_instance()
        candidate = self.parent.__globals__.get("_candidate")
        self.router_policy = candidate.adapter.parent
        self.router_module = candidate.adapter.router
        self.states: dict[int, dict[str, Any]] = {}
        self.telemetry = {
            "sale_mode": sale_mode,
            "crop_guard": self.crop_guard,
            "front_run_events": [],
            "opponent_preemptions_detected": [],
            "crop_swap_events": [],
            "held_floor_sales": [],
        }

    def _new_state(self) -> dict[str, Any]:
        return {
            "last_step": -1,
            "similar_streak": 0,
            "credits": defaultdict(lambda: defaultdict(int)),
            "previous_moneys": None,
            "previous_own_sales": defaultdict(int),
            "previous_probe_value": 0,
            "opponent_preemptor": False,
            "low_strawberry_streak": 0,
            "pending_tomato_plants": 0,
            "crop_swaps": 0,
        }

    def _state(self, observation: Any) -> dict[str, Any]:
        seat, step = _player(observation), _step(observation)
        state = self.states.get(seat)
        if state is None or step <= int(state["last_step"]):
            state = self.states[seat] = self._new_state()
        state["last_step"] = step
        return state

    def _observe_opponent_sale(self, observation: Any, state: dict[str, Any]) -> None:
        seat = _player(observation)
        farms = list(_get(observation, "farms", []) or [])
        current = (
            float(_get(farms[seat], "money", 0) or 0),
            float(_get(farms[1 - seat], "money", 0) or 0),
        )
        previous = state["previous_moneys"]
        probe_value = int(state["previous_probe_value"])
        if previous is not None and probe_value >= 100 and not state["opponent_preemptor"]:
            own_delta = current[0] - previous[0]
            rival_delta = current[1] - previous[1]
            # With the public physical route still >=90% identical, non-market
            # spending is normally identical.  A matched money jump therefore
            # identifies that the rival joined our real T-1 sale.  This is much
            # stricter than market-inventory deltas, which are confounded by
            # nearby baseline sales and town demand.
            tolerance = max(5.0, 0.05 * probe_value)
            if abs(own_delta - rival_delta) <= tolerance:
                state["opponent_preemptor"] = True
                self.telemetry["opponent_preemptions_detected"].append(
                    {
                        "observed_step": _step(observation),
                        "own_money_delta": own_delta,
                        "rival_money_delta": rival_delta,
                        "probe_value_upper_bound": probe_value,
                        "tolerance": tolerance,
                    }
                )
        state["previous_moneys"] = current
        state["previous_own_sales"] = defaultdict(int)
        state["previous_probe_value"] = 0

    def _settle_credits(self, action: dict[str, Any], state: dict[str, Any], step: int) -> None:
        due = state["credits"].pop(step, {})
        if not due:
            return
        for order in action.get("market") or []:
            if not (isinstance(order, list) and len(order) >= 3 and order[0] == "SELL"):
                continue
            item = str(order[1])
            quantity = max(0, int(due.get(item, 0)))
            if quantity:
                removed = min(max(0, int(order[2])), quantity)
                order[2] = int(order[2]) - removed
                due[item] -= removed

    def _front_run(self, observation: Any, action: dict[str, Any], state: dict[str, Any]) -> None:
        step = _step(observation)
        if self.sale_mode == "none" or not (SALE_START <= step < SALE_RAW_END):
            return
        similarity = public_similarity(observation)
        state["similar_streak"] = state["similar_streak"] + 1 if similarity >= 0.90 else 0
        if state["similar_streak"] < 6:
            return

        extra_lead = 1
        if self.sale_mode == "adaptive_lead_two" and state["opponent_preemptor"]:
            extra_lead = 2

        route_state = self.router_policy.players.get(_player(observation))
        if route_state is None:
            return
        tape = self.router_policy.tapes[int(route_state.plan)]
        prices = _prices(observation)
        view = self.router_module.FarmView(observation)
        stock = self.router_module.projected_shed(action, view)
        already = {
            str(order[1])
            for order in action.get("market") or []
            if isinstance(order, list) and len(order) >= 3 and order[0] == "SELL"
        }

        candidates: list[tuple[int, str, int]] = []
        # At most three future tape rows are relevant.  The parent's own
        # one-turn advance has a four-step/day-boundary guard, so derive the
        # true baseline sale step for each row instead of assuming raw-1.
        for raw_step in range(step + 1, min(step + 4, SALE_RAW_END + 1)):
            if _baseline_sale_step(raw_step) - extra_lead != step:
                continue
            for item, planned in _planned_sales(tape, raw_step).items():
                candidates.append((raw_step, item, planned))

        probe_value = 0
        for raw_step, item, planned in candidates:
            if item in already or int(prices.get(item, 0)) < 2:
                continue
            quantity = min(max(0, int(stock.get(item, 0))), planned)
            if quantity < MIN_FRONT_RUN_UNITS:
                continue
            if len(action.get("market") or []) >= MAX_ORDERS:
                break
            action.setdefault("market", []).append(["SELL", item, quantity])
            already.add(item)
            stock[item] = max(0, int(stock.get(item, 0)) - quantity)
            state["credits"][raw_step][item] += quantity
            probe_value += quantity * int(prices.get(item, 0))
            self.telemetry["front_run_events"].append(
                {
                    "step": step,
                    "raw_sale_step": raw_step,
                    "item": item,
                    "quantity": quantity,
                    "extra_lead": extra_lead,
                    "similarity": round(similarity, 4),
                }
            )
        state["previous_probe_value"] = probe_value

    def _crop_value_guard(
        self, observation: Any, action: dict[str, Any], state: dict[str, Any]
    ) -> None:
        if not self.crop_guard:
            return
        step = _step(observation)
        prices = _prices(observation)
        strawberry_price = int(prices.get("STRAWBERRY", 0))
        if strawberry_price <= LOW_STRAWBERRY_PRICE:
            state["low_strawberry_streak"] += 1
        else:
            state["low_strawberry_streak"] = 0

        # At the absolute price floor, defer crop sales only while the shed
        # has room.  This avoids paying the irreversible $1 liquidation tax
        # without risking a nearly-full shed or touching terminal liquidation.
        market = action.setdefault("market", [])
        if CROP_START <= step < 648 and sum(_shed(observation).values()) <= 80:
            kept = []
            for order in market:
                if (
                    isinstance(order, list)
                    and len(order) >= 3
                    and order[0] == "SELL"
                    and order[1] in {"STRAWBERRY", "TOMATO", "MELON"}
                    and int(prices.get(str(order[1]), 0)) <= 1
                ):
                    self.telemetry["held_floor_sales"].append(
                        {"step": step, "item": str(order[1]), "quantity": int(order[2])}
                    )
                    continue
                kept.append(order)
            action["market"] = market = kept

        # First spend already-purchased tomato replacement seeds.  This is
        # done before changing today's market order because today's purchase
        # is unavailable to workers until the next callback.
        tomato_available = int(_seeds(observation).get("TOMATO", 0))
        replace_now = min(int(state["pending_tomato_plants"]), tomato_available)
        if replace_now > 0:
            commands = _commands(action)
            changed = 0
            for command in commands:
                if changed >= replace_now:
                    break
                if isinstance(command, list) and command[:2] == ["PLANT", "STRAWBERRY"]:
                    command[1] = "TOMATO"
                    changed += 1
            state["pending_tomato_plants"] -= changed

        if not (CROP_START <= step <= CROP_END):
            return
        if state["crop_swaps"] >= MAX_CROP_SWAPS:
            return
        if state["low_strawberry_streak"] < LOW_PRICE_STREAK:
            return
        # Tomato must be not merely better at this instant but sufficiently
        # better to pay for model error while preserving an ongoing crop.
        tomato_price = int(prices.get("TOMATO", 0))
        if tomato_price < 20 or tomato_price < strawberry_price + 15:
            return

        for index, order in enumerate(market):
            if not (
                isinstance(order, list)
                and len(order) >= 3
                and order[0] == "BUY_SEED"
                and order[1] == "STRAWBERRY"
                and int(order[2]) > 0
            ):
                continue
            quantity = int(order[2])
            if quantity == 1:
                market[index] = ["BUY_SEED", "TOMATO", 1]
            elif len(market) < MAX_ORDERS:
                order[2] = quantity - 1
                market.append(["BUY_SEED", "TOMATO", 1])
            else:
                return
            state["pending_tomato_plants"] += 1
            state["crop_swaps"] += 1
            self.telemetry["crop_swap_events"].append(
                {
                    "step": step,
                    "from": "STRAWBERRY",
                    "to": "TOMATO",
                    "units": 1,
                    "strawberry_price": strawberry_price,
                    "tomato_price": tomato_price,
                    "persistent_steps": state["low_strawberry_streak"],
                }
            )
            break

    def __call__(self, observation: Any, configuration: Any = None) -> dict[str, Any]:
        state = self._state(observation)
        self._observe_opponent_sale(observation, state)
        action = deepcopy(self.parent(observation, configuration))
        step = _step(observation)
        self._settle_credits(action, state, step)
        self._crop_value_guard(observation, action, state)
        self._front_run(observation, action, state)
        action["market"] = (action.get("market") or [])[:MAX_ORDERS]
        # Detection on the next observation must subtract every own sale, not
        # only the overlay's extra sale.  Otherwise a baseline sale at a probe
        # step would be misclassified as an opponent preemption.
        own_sales: dict[str, int] = defaultdict(int)
        for order in action["market"]:
            if isinstance(order, list) and len(order) >= 3 and order[0] == "SELL":
                own_sales[str(order[1])] += max(0, int(order[2]))
        state["previous_own_sales"] = own_sales
        return action


def build_agent(*, sale_mode: str = "none", crop_guard: bool = False):
    instance = ExperimentAgent(sale_mode=sale_mode, crop_guard=crop_guard)

    def agent(observation: Any, configuration: Any = None):
        return instance(observation, configuration)

    agent.experiment = instance
    agent.telemetry = instance.telemetry
    return agent
