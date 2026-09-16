"""Numpy-only Policy3 action (deployment). Bit-exact mirror of act3.act_policy3
with temperature=0, using NpPolicy3 + act_common + actions.decode_action.

No torch import: safe inside the Kaggle agent bundle.
"""
from __future__ import annotations

import numpy as np

from act_common import (
    apply_cow_cap,
    apply_sell_rules,
    legal_option_mask,
    step_toward,
    trim_plants,
)
from actions import decode_action
from features import MAX_UNITS, N_OPS, OP_INDEX, OPS, PRODUCTS, QTY_BUCKETS, SHED_TILES, encode, unbucket


def act_np_policy3(pol, obs, seat, state=None, sell_rule_c=True, dump_fert=True,
                   intra_thresh=0, cow_cap=0, melon_hold_until=0):
    features = encode(obs, seat)
    farm = obs["farms"][seat]
    units = [farm["farmer"], *farm["hands"]][:MAX_UNITS]
    unit_count = len(units)
    prior_options = np.full(MAX_UNITS, -1, dtype=np.int32)
    if state is not None and "options" in state:
        prior_options[:] = state["options"]

    hidden = pol.encode(features)
    option_logits = pol.option_logits(hidden)
    market = pol.market(hidden)

    claimed = set()
    selected_options = np.full(MAX_UNITS, -1, dtype=np.int32)
    destinations = np.zeros(MAX_UNITS, dtype=np.int64)
    operations = np.full(MAX_UNITS, OP_INDEX["PASS"], dtype=np.int64)
    for unit_index, unit in enumerate(units):
        position = (int(unit[0]), int(unit[1]))
        legal = legal_option_mask(obs, seat, unit_index, position)
        for destination in claimed:
            legal[destination] = False

        option = int(prior_options[unit_index])
        prior_destination = option // N_OPS if option >= 0 else -1
        prior_operation = option % N_OPS if option >= 0 else -1
        if option < 0 or not legal[prior_destination, prior_operation]:
            logits = option_logits[unit_index].copy()
            logits[~legal] = -1e9
            option = int(logits.reshape(-1).argmax())

        destination, operation = divmod(option, N_OPS)
        x, y = destination % 10, destination // 10
        destinations[unit_index] = destination
        operations[unit_index] = operation
        if (x, y) not in SHED_TILES:
            claimed.add(destination)
        if position != (x, y):
            selected_options[unit_index] = option

    quantity_logits = pol.qty_logits(hidden, destinations)

    unit_actions = []
    for unit_index, unit in enumerate(units):
        destination = int(destinations[unit_index])
        x, y = destination % 10, destination // 10
        position = (int(unit[0]), int(unit[1]))
        if position != (x, y):
            unit_actions.append(step_toward(position, (x, y)))
            continue
        operation = int(operations[unit_index])
        name = OPS[operation]
        quantity = unbucket(int(quantity_logits[unit_index].argmax()), QTY_BUCKETS)
        if name.startswith("PLANT_"):
            unit_actions.append(["PLANT", name[6:]])
        elif name.startswith("PICKUP_"):
            unit_actions.append(["PICKUP", name[7:], int(quantity)])
        elif name.startswith("PLACE_"):
            item = name[6:]
            unit_actions.append(["PLACE", item, int(quantity)] if item in PRODUCTS else ["PLACE", item])
        else:
            unit_actions.append([name])

    for unit_index in trim_plants(unit_actions, obs["private"]["seeds"]):
        selected_options[unit_index] = -1
    market_targets = np.concatenate([
        market["sell"].argmax(-1),
        market["buyp"].argmax(-1),
        market["seed"].argmax(-1),
        market["anim"].argmax(-1),
        [int(market["hire"].argmax())],
        [int(market["land"].argmax())],
    ])
    action = decode_action(np.zeros(MAX_UNITS, dtype=int), np.zeros(MAX_UNITS, dtype=int), market_targets, obs, seat)
    if sell_rule_c:
        action = apply_sell_rules(action, obs, seat, dump_fert=dump_fert, intra_thresh=intra_thresh,
                                  melon_hold_until=melon_hold_until)
    if cow_cap > 0:
        action = apply_cow_cap(action, state, cap=cow_cap)
    action["farmer"] = unit_actions[0] if unit_actions else ["PASS"]
    action["hands"] = unit_actions[1:]
    if state is not None:
        state["options"] = selected_options
    return action, {"dest": destinations[:unit_count], "op": operations[:unit_count]}
