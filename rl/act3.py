"""Inference for Policy3 joint destination-operation options."""
from __future__ import annotations

import numpy as np
import torch

from actions import decode_action
from act2 import step_toward, trim_plants
from features import (
    CROPS,
    MAX_UNITS,
    MKT_BUCKETS,
    N_OPS,
    OP_INDEX,
    OPS,
    PRODUCTS,
    QTY_BUCKETS,
    SHED_TILES,
    encode,
    legal_ops_at,
    unbucket,
)


def legal_option_mask(obs, seat, unit_index, position):
    """Return current legal options plus structurally valid options at remote tiles."""
    mask = np.zeros((100, N_OPS), dtype=bool)
    tiles = obs["farms"][seat]["tiles"]
    for destination in range(100):
        x, y = destination % 10, destination // 10
        if (x, y) == position:
            mask[destination] = legal_ops_at(obs, seat, unit_index, x, y)
            mask[destination, 1:5] = False
            continue
        tile = tiles[y][x]
        names = []
        if tile is None:
            names = ["PLANT_" + crop for crop in CROPS] + ["DIG", "BUILD_COOP", "BUILD_PASTURE"]
        elif isinstance(tile, dict):
            if "animal" in tile:
                names = ["FEED", "CARE", "HARVEST", "COLLECT_FERTILIZER"]
            elif tile.get("kind") == "WEED":
                names = ["DIG"]
            elif tile.get("kind") == "PLANT":
                names = ["WATER", "HARVEST", "FERTILIZE", "DIG"]
            elif tile.get("kind") == "COOP":
                names = ["DIG", "PLACE_GOOSE"]
            elif tile.get("kind") == "PASTURE":
                names = ["DIG", "PLACE_COW", "PLACE_SHEEP"]
        if (x, y) in SHED_TILES:
            names += ["PICKUP_" + item for item in (*PRODUCTS, "GOOSE", "COW", "SHEEP")]
            names += ["DROP"] + ["PLACE_" + item for item in PRODUCTS]
        for name in names:
            mask[destination, OP_INDEX[name]] = True
    return mask


def act_policy3(model, obs, seat, device, temperature=0.0, rng=None, state=None):
    features = encode(obs, seat)
    farm = obs["farms"][seat]
    units = [farm["farmer"], *farm["hands"]][:MAX_UNITS]
    unit_count = len(units)
    prior_options = np.full(MAX_UNITS, -1, dtype=np.int32)
    if state is not None and "options" in state:
        prior_options[:] = state["options"]

    with torch.no_grad():
        hidden = model.encode(*(torch.from_numpy(features[key]).unsqueeze(0).to(device) for key in ("tiles", "units", "items", "glob")))
        option_logits = model.option_logits(hidden)[0].cpu().numpy()
        market = model.market(hidden)

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
            flat = logits.reshape(-1)
            if temperature > 0:
                probabilities = np.exp((flat - flat.max()) / temperature)
                probabilities /= probabilities.sum()
                option = int((rng or np.random).choice(len(flat), p=probabilities))
            else:
                option = int(flat.argmax())

        destination, operation = divmod(option, N_OPS)
        x, y = destination % 10, destination // 10
        destinations[unit_index] = destination
        operations[unit_index] = operation
        if (x, y) not in SHED_TILES:
            claimed.add(destination)
        if position != (x, y):
            selected_options[unit_index] = option

    with torch.no_grad():
        quantity_logits = model.qty_logits(hidden, torch.from_numpy(destinations).unsqueeze(0).to(device))[0].cpu().numpy()

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
        market["sell"][0].argmax(-1).cpu().numpy(),
        market["buyp"][0].argmax(-1).cpu().numpy(),
        market["seed"][0].argmax(-1).cpu().numpy(),
        market["anim"][0].argmax(-1).cpu().numpy(),
        [int(market["hire"][0].argmax())],
        [int(market["land"][0].argmax())],
    ])
    action = decode_action(np.zeros(MAX_UNITS, dtype=int), np.zeros(MAX_UNITS, dtype=int), market_targets, obs, seat)
    action["farmer"] = unit_actions[0] if unit_actions else ["PASS"]
    action["hands"] = unit_actions[1:]
    if state is not None:
        state["options"] = selected_options
    return action, {"dest": destinations[:unit_count], "op": operations[:unit_count]}