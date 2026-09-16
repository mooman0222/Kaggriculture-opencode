"""Policy3: score each (destination tile, operation) as one joint option."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from features import (
    GLOB_F,
    ITEM_F,
    MAX_UNITS,
    MKT_BUCKETS,
    N_OPS,
    OP_INDEX,
    PRODUCTS,
    QTY_BUCKETS,
    TILE_F,
    UNIT_F,
)
from model import _mkt_ce, market_targets_split

N_TILES = 200
N_OWN = 100
N_ITEMS = len(PRODUCTS)
N_QTY = len(QTY_BUCKETS)
N_MB = len(MKT_BUCKETS)
OPTION_RANK = 16
OP_WEIGHT = None
PASS_IDLE_WEIGHT = 1.0  # set by the trainer (--pass-idle-weight): weight of a PASS label recorded while the farm still had an unwatered plant or an unfed animal (closed loop idles on these)


def option_mask_from_features(tiles, units):
    """Structural options available at each tile; transient legality is checked on arrival."""
    batch_size = tiles.shape[0]
    own_tiles = tiles.reshape(batch_size, N_TILES, TILE_F)[:, :N_OWN]
    kind = own_tiles[..., 0]
    present = units[..., 0] > 0
    mask = torch.zeros(batch_size, MAX_UNITS, N_OWN, N_OPS, dtype=torch.bool, device=tiles.device)

    current = (units[..., 3] * 10 + units[..., 2]).long().clamp(0, N_OWN - 1)
    mask[..., OP_INDEX["PASS"]].scatter_(2, current.unsqueeze(-1), present.unsqueeze(-1))

    def allow(tile_mask, operations):
        expanded = present.unsqueeze(-1) & tile_mask.unsqueeze(1)
        for operation in operations:
            mask[..., OP_INDEX[operation]] |= expanded

    allow(kind == 1, ["PLANT_WHEAT", "PLANT_CARROT", "PLANT_TOMATO", "PLANT_STRAWBERRY", "PLANT_MELON", "DIG", "BUILD_COOP", "BUILD_PASTURE"])
    allow(kind == 2, ["DIG"])
    allow(kind == 3, ["WATER", "HARVEST", "FERTILIZE", "DIG"])
    allow(kind == 4, ["DIG", "PLACE_GOOSE"])
    allow(kind == 5, ["DIG", "PLACE_COW", "PLACE_SHEEP"])
    allow(kind == 6, ["FEED", "CARE", "HARVEST", "COLLECT_FERTILIZER"])

    shed = torch.zeros(N_OWN, dtype=torch.bool, device=tiles.device)
    shed[[44, 45, 54, 55]] = True
    allow(shed.unsqueeze(0).expand(batch_size, -1), [
        "PICKUP_WHEAT", "PICKUP_CARROT", "PICKUP_TOMATO", "PICKUP_STRAWBERRY", "PICKUP_MELON",
        "PICKUP_EGG", "PICKUP_MILK", "PICKUP_WOOL", "PICKUP_FERTILIZER", "PICKUP_GOOSE", "PICKUP_COW", "PICKUP_SHEEP",
        "DROP", "PLACE_WHEAT", "PLACE_CARROT", "PLACE_TOMATO", "PLACE_STRAWBERRY", "PLACE_MELON",
        "PLACE_EGG", "PLACE_MILK", "PLACE_WOOL", "PLACE_FERTILIZER",
    ])
    return mask


class Policy3(nn.Module):
    def __init__(self, d=128, layers=3, heads=4, ff=256, drop=0.0):
        super().__init__()
        self.d = d
        self.kind_emb = nn.Embedding(8, 16)
        self.crop_emb = nn.Embedding(6, 8)
        self.anim_emb = nn.Embedding(4, 8)
        self.tile_in = nn.Linear(16 + 8 + 8 + (TILE_F - 3), d)
        self.tile_pos = nn.Parameter(torch.randn(N_TILES, d) * 0.02)
        self.unit_in = nn.Linear(UNIT_F, d)
        self.unit_pos = nn.Parameter(torch.randn(MAX_UNITS, d) * 0.02)
        self.item_in = nn.Linear(ITEM_F, d)
        self.item_pos = nn.Parameter(torch.randn(N_ITEMS, d) * 0.02)
        self.glob_in = nn.Linear(GLOB_F, d)
        self.type_emb = nn.Embedding(4, d)
        encoder = nn.TransformerEncoderLayer(d, heads, ff, dropout=drop, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(encoder, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

        self.option_q = nn.Linear(d, N_OPS * OPTION_RANK)
        self.option_k = nn.Linear(d, N_OPS * OPTION_RANK)
        self.option_bias = nn.Linear(d, N_OPS)
        self.qty_mlp = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, N_QTY))

        self.sell_head = nn.Linear(d, N_MB)
        self.buyp_head = nn.Linear(d, N_MB)
        self.seed_head = nn.Linear(d, N_MB)
        self.anim_head = nn.Linear(d, 3 * 5)
        self.hire_head = nn.Linear(d, 13)
        self.land_head = nn.Linear(d, 2)
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def encode(self, tiles, units, items, glob):
        batch_size = tiles.shape[0]
        tile_rows = tiles.reshape(batch_size, N_TILES, TILE_F)
        kind = tile_rows[..., 0].long().clamp(0, 7)
        crop = tile_rows[..., 1].long().clamp(0, 5)
        animal = tile_rows[..., 2].long().clamp(0, 3)
        scale = torch.tensor([30, 6, 1, 3, 3, 1, 1, 3, 1, 3, 8, 3, 1, 9, 9], device=tiles.device, dtype=torch.float32)
        numeric = tile_rows[..., 3:].float() / scale
        tile_tokens = self.tile_in(torch.cat([self.kind_emb(kind), self.crop_emb(crop), self.anim_emb(animal), numeric], -1))
        tile_tokens = tile_tokens + self.tile_pos + self.type_emb.weight[1]

        unit_scale = torch.tensor([1, 1, 9, 9] + [40] * 12 + [1, 16], device=units.device, dtype=torch.float32)
        unit_tokens = self.unit_in(units.float() / unit_scale) + self.unit_pos + self.type_emb.weight[2]
        item_tokens = self.item_in(items) + self.item_pos + self.type_emb.weight[3]
        global_token = (self.glob_in(glob) + self.type_emb.weight[0]).unsqueeze(1)
        tokens = torch.cat([global_token, tile_tokens, unit_tokens, item_tokens], 1)
        present = units[..., 0] > 0
        padding = torch.zeros(batch_size, tokens.shape[1], dtype=torch.bool, device=tokens.device)
        padding[:, 1 + N_TILES:1 + N_TILES + MAX_UNITS] = ~present
        hidden = self.norm(self.enc(tokens, src_key_padding_mask=padding))
        return {
            "g": hidden[:, 0],
            "own": hidden[:, 1:1 + N_OWN],
            "units": hidden[:, 1 + N_TILES:1 + N_TILES + MAX_UNITS],
            "items": hidden[:, 1 + N_TILES + MAX_UNITS:],
            "locked": tile_rows[:, :N_OWN, 0] == 0,
        }

    def option_logits(self, hidden):
        batch_size = hidden["units"].shape[0]
        queries = self.option_q(hidden["units"]).view(batch_size, MAX_UNITS, N_OPS, OPTION_RANK)
        keys = self.option_k(hidden["own"]).view(batch_size, N_OWN, N_OPS, OPTION_RANK)
        scores = torch.einsum("buor,btor->buto", queries, keys) / (OPTION_RANK ** 0.5)
        return scores + self.option_bias(hidden["units"]).unsqueeze(2)

    def qty_logits(self, hidden, dest):
        batch_size, units = dest.shape
        tile_token = torch.gather(hidden["own"], 1, dest.clamp(min=0).unsqueeze(-1).expand(batch_size, units, self.d))
        return self.qty_mlp(torch.cat([hidden["units"], tile_token], -1))

    def market(self, hidden):
        global_token, item_tokens = hidden["g"], hidden["items"]
        batch_size = global_token.shape[0]
        return {
            "sell": self.sell_head(item_tokens),
            "buyp": self.buyp_head(item_tokens[:, [0, 8]]),
            "seed": self.seed_head(item_tokens[:, :5]),
            "anim": self.anim_head(global_token).view(batch_size, 3, 5),
            "hire": self.hire_head(global_token),
            "land": self.land_head(global_token),
            "value": self.value_head(global_token).squeeze(-1),
        }

    def forward(self, tiles, units, items, glob, dest=None):
        hidden = self.encode(tiles, units, items, glob)
        out = self.market(hidden)
        out["option"] = self.option_logits(hidden)
        if dest is None:
            option = out["option"].masked_fill(hidden["locked"].unsqueeze(1).unsqueeze(-1), -1e9).flatten(2).argmax(-1)
            dest = option // N_OPS
        out["qty"] = self.qty_logits(hidden, dest)
        out["H"] = hidden
        return out


def bc_loss3(out, batch):
    dest = batch["dest"].long()
    operation = batch["dop"].long()
    present = dest >= 0
    target = dest.clamp(min=0) * N_OPS + operation.clamp(min=0)
    allowed_tiles = option_mask_from_features(batch["tiles"], batch["units"])
    target_mask = F.one_hot(target, N_OWN * N_OPS).view_as(out["option"]).bool() & present.unsqueeze(-1).unsqueeze(-1)
    logits = out["option"].masked_fill(~(allowed_tiles | target_mask), -1e9).flatten(2)

    current = (batch["units"][..., 3] * 10 + batch["units"][..., 2]).long()
    sample_weight = torch.where(dest == current, 1.0, 3.0)
    own = batch["tiles"].reshape(-1, N_TILES, TILE_F)[:, :N_OWN]
    work_left = (((own[..., 0] == 3) & (own[..., 5] == 0)) | ((own[..., 0] == 6) & (own[..., 8] == 0))).any(-1)
    idle_pass = (operation == OP_INDEX["PASS"]) & work_left.unsqueeze(-1)
    sample_weight = torch.where(idle_pass, sample_weight * PASS_IDLE_WEIGHT, sample_weight)
    if OP_WEIGHT is not None:
        sample_weight = sample_weight * OP_WEIGHT.to(logits.device)[operation.clamp(min=0)]
    option_loss = F.cross_entropy(logits[present], target[present], reduction="none")
    weights = sample_weight[present]
    option_loss = (option_loss * weights).sum() / weights.sum()

    quantity_op = ((operation >= 16) & (operation <= 27)) | ((operation >= 29) & (operation <= 40))
    quantity_mask = present & quantity_op
    quantity_loss = F.cross_entropy(out["qty"][quantity_mask], batch["dqty"].long()[quantity_mask]) if quantity_mask.any() else logits.new_zeros(())

    targets = market_targets_split(batch["mkt"].long())
    sell_loss = _mkt_ce(out["sell"], targets["sell"], N_MB)
    buy_product_loss = _mkt_ce(out["buyp"], targets["buyp"], N_MB)
    seed_loss = _mkt_ce(out["seed"], targets["seed"], N_MB)
    animal_loss = _mkt_ce(out["anim"], targets["anim"], 5, 20.0)
    hire_loss = _mkt_ce(out["hire"], targets["hire"], 13, 8.0)
    land_loss = _mkt_ce(out["land"], targets["land"], 2, 30.0)
    loss = option_loss + 0.3 * quantity_loss + sell_loss + 0.5 * (buy_product_loss + seed_loss + animal_loss) + 0.5 * hire_loss + 0.5 * land_loss

    with torch.no_grad():
        predicted = logits.argmax(-1)
        accuracy = {
            "option": (predicted == target)[present].float().mean().item(),
            "dest": (predicted // N_OPS == dest)[present].float().mean().item(),
            "op": (predicted % N_OPS == operation)[present].float().mean().item(),
            "sell": (out["sell"].argmax(-1) == targets["sell"]).float().mean().item(),
            "hire": (out["hire"].argmax(-1) == targets["hire"]).float().mean().item(),
        }
    return loss, accuracy