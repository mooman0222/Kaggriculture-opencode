"""Small Transformer policy: tokens = 1 global + 200 tiles + 14 units + 9 items. Heads: unit op/qty, per-item market, global market, value."""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from features import TILE_F, UNIT_F, ITEM_F, GLOB_F, MAX_UNITS, N_OPS, QTY_BUCKETS, MKT_BUCKETS, PRODUCTS
from actions import MKT_CLASSES, N_MKT

N_TILES = 200; N_ITEMS = len(PRODUCTS); N_QTY = len(QTY_BUCKETS); N_MB = len(MKT_BUCKETS)


class Policy(nn.Module):
    def __init__(self, d=128, layers=3, heads=4, ff=256, drop=0.0):
        super().__init__()
        self.kind_emb = nn.Embedding(8, 16); self.crop_emb = nn.Embedding(6, 8); self.anim_emb = nn.Embedding(4, 8)
        self.tile_in = nn.Linear(16 + 8 + 8 + (TILE_F - 3), d); self.tile_pos = nn.Parameter(torch.randn(N_TILES, d) * 0.02)
        self.unit_in = nn.Linear(UNIT_F, d); self.unit_pos = nn.Parameter(torch.randn(MAX_UNITS, d) * 0.02)
        self.item_in = nn.Linear(ITEM_F, d); self.item_pos = nn.Parameter(torch.randn(N_ITEMS, d) * 0.02)
        self.glob_in = nn.Linear(GLOB_F, d)
        self.type_emb = nn.Embedding(4, d)
        enc = nn.TransformerEncoderLayer(d, heads, ff, dropout=drop, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers); self.norm = nn.LayerNorm(d)
        self.op_head = nn.Linear(d, N_OPS); self.qty_head = nn.Linear(d, N_QTY)
        self.sell_head = nn.Linear(d, N_MB); self.buyp_head = nn.Linear(d, N_MB); self.seed_head = nn.Linear(d, N_MB)
        self.anim_head = nn.Linear(d, 3 * 5); self.hire_head = nn.Linear(d, 13); self.land_head = nn.Linear(d, 2); self.value_head = nn.Linear(d, 1)

    def forward(self, tiles, units, items, glob):
        """tiles [B,2,10,10,F] int, units [B,U,UF] int, items [B,9,IF] f32, glob [B,G] f32 -> dict of logits."""
        B = tiles.shape[0]
        t = tiles.reshape(B, N_TILES, TILE_F)
        kind = t[..., 0].long().clamp(0, 7); crop = t[..., 1].long().clamp(0, 5); anim = t[..., 2].long().clamp(0, 3)
        num = t[..., 3:].float() / torch.tensor([30, 6, 1, 3, 3, 1, 1, 3, 1, 3, 8, 3, 1, 9, 9], device=t.device, dtype=torch.float32)
        tt = self.tile_in(torch.cat([self.kind_emb(kind), self.crop_emb(crop), self.anim_emb(anim), num], -1)) + self.tile_pos + self.type_emb.weight[1]
        u = units.float(); u = u / torch.tensor([1, 1, 9, 9] + [40] * 12 + [1, 16], device=u.device, dtype=torch.float32)
        ut = self.unit_in(u) + self.unit_pos + self.type_emb.weight[2]
        it = self.item_in(items) + self.item_pos + self.type_emb.weight[3]
        gt = (self.glob_in(glob) + self.type_emb.weight[0]).unsqueeze(1)
        x = torch.cat([gt, tt, ut, it], 1)
        present = units[..., 0] > 0
        pad = torch.zeros(B, x.shape[1], dtype=torch.bool, device=x.device); pad[:, 1 + N_TILES:1 + N_TILES + MAX_UNITS] = ~present
        h = self.norm(self.enc(x, src_key_padding_mask=pad))
        g = h[:, 0]; hu = h[:, 1 + N_TILES:1 + N_TILES + MAX_UNITS]; hi = h[:, 1 + N_TILES + MAX_UNITS:]
        out = {"op": self.op_head(hu), "qty": self.qty_head(hu), "sell": self.sell_head(hi), "buyp": self.buyp_head(hi[:, [0, 8]]), "seed": self.seed_head(hi[:, :5]),
               "anim": self.anim_head(g).view(B, 3, 5), "hire": self.hire_head(g), "land": self.land_head(g), "value": self.value_head(g).squeeze(-1)}
        return out


def market_targets_split(mkt):
    """mkt [B,21] -> dict of targets matching head layout."""
    return {"sell": mkt[:, 0:9], "buyp": mkt[:, 9:11], "seed": mkt[:, 11:16], "anim": mkt[:, 16:19], "hire": mkt[:, 19], "land": mkt[:, 20]}


OP_WEIGHT = None  # set by trainer: tensor [N_OPS] of class weights (rare ops up-weighted)


def _mkt_ce(logits, target, n_classes, pos_w=6.0):
    """CE with non-zero targets up-weighted (market decisions are rare events)."""
    l = F.cross_entropy(logits.reshape(-1, n_classes), target.reshape(-1), reduction="none")
    w = torch.where(target.reshape(-1) > 0, torch.full_like(l, pos_w), torch.ones_like(l))
    return (l * w).sum() / w.sum()


def bc_loss(out, batch):
    """batch: op [B,U] (-1 absent), qty [B,U], mkt [B,21], mask [B,U,OPS]. Returns total loss and per-head accuracies."""
    op = batch["op"].long(); present = op >= 0
    mask = batch["mask"].clone()
    tgt = F.one_hot(op.clamp(min=0), mask.shape[-1]).bool() & present.unsqueeze(-1)
    mask |= tgt  # the recorded action is legal by definition (engine may still have refused it)
    logits = out["op"].masked_fill(~mask, -1e9)
    l_op = F.cross_entropy(logits[present], op[present], weight=OP_WEIGHT.to(logits.device) if OP_WEIGHT is not None else None)
    is_qty = present & ((op >= 16) & (op <= 27) | (op >= 29) & (op <= 40))
    l_qty = F.cross_entropy(out["qty"][is_qty], batch["qty"].long()[is_qty]) if is_qty.any() else logits.new_zeros(())
    tg = market_targets_split(batch["mkt"].long())
    l_sell = _mkt_ce(out["sell"], tg["sell"], N_MB); l_buyp = _mkt_ce(out["buyp"], tg["buyp"], N_MB); l_seed = _mkt_ce(out["seed"], tg["seed"], N_MB)
    l_anim = _mkt_ce(out["anim"], tg["anim"], 5, 20.0); l_hire = _mkt_ce(out["hire"], tg["hire"], 13, 8.0); l_land = _mkt_ce(out["land"], tg["land"], 2, 30.0)
    loss = l_op + 0.3 * l_qty + l_sell + 0.5 * (l_buyp + l_seed + l_anim) + 0.5 * l_hire + 0.5 * l_land
    with torch.no_grad():
        acc = {"op": (logits.argmax(-1) == op)[present].float().mean().item(), "sell": (out["sell"].argmax(-1) == tg["sell"]).float().mean().item(),
               "hire": (out["hire"].argmax(-1) == tg["hire"]).float().mean().item(), "seed": (out["seed"].argmax(-1) == tg["seed"]).float().mean().item(),
               "anim": (out["anim"].argmax(-1) == tg["anim"]).float().mean().item(), "land": (out["land"].argmax(-1) == tg["land"]).float().mean().item()}
    return loss, acc
