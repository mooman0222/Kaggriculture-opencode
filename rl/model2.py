"""Policy v2: units choose a destination tile (attention over own tiles) and the op to perform there (conditioned on the destination token)."""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from features import TILE_F, UNIT_F, ITEM_F, GLOB_F, MAX_UNITS, N_OPS, QTY_BUCKETS, MKT_BUCKETS, PRODUCTS
from model import market_targets_split, _mkt_ce

N_TILES = 200; N_OWN = 100; N_ITEMS = len(PRODUCTS); N_QTY = len(QTY_BUCKETS); N_MB = len(MKT_BUCKETS)
OP_WEIGHT = None


class Policy2(nn.Module):
    def __init__(self, d=128, layers=3, heads=4, ff=256, drop=0.0):
        super().__init__()
        self.d = d
        self.kind_emb = nn.Embedding(8, 16); self.crop_emb = nn.Embedding(6, 8); self.anim_emb = nn.Embedding(4, 8)
        self.tile_in = nn.Linear(16 + 8 + 8 + (TILE_F - 3), d); self.tile_pos = nn.Parameter(torch.randn(N_TILES, d) * 0.02)
        self.unit_in = nn.Linear(UNIT_F, d); self.unit_pos = nn.Parameter(torch.randn(MAX_UNITS, d) * 0.02)
        self.item_in = nn.Linear(ITEM_F, d); self.item_pos = nn.Parameter(torch.randn(N_ITEMS, d) * 0.02)
        self.glob_in = nn.Linear(GLOB_F, d); self.type_emb = nn.Embedding(4, d)
        enc = nn.TransformerEncoderLayer(d, heads, ff, dropout=drop, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False); self.norm = nn.LayerNorm(d)
        self.prev_dest_emb = nn.Embedding(101, d); self.prev_op_emb = nn.Embedding(45, d)
        self.dq = nn.Linear(d, d); self.dk = nn.Linear(d, d)
        self.op_mlp = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, N_OPS)); self.qty_mlp = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, N_QTY))
        self.sell_head = nn.Linear(d, N_MB); self.buyp_head = nn.Linear(d, N_MB); self.seed_head = nn.Linear(d, N_MB)
        self.anim_head = nn.Linear(d, 3 * 5); self.hire_head = nn.Linear(d, 13); self.land_head = nn.Linear(d, 2); self.value_head = nn.Linear(d, 1)

    def encode(self, tiles, units, items, glob, prev=None):
        B = tiles.shape[0]; t = tiles.reshape(B, N_TILES, TILE_F)
        kind = t[..., 0].long().clamp(0, 7); crop = t[..., 1].long().clamp(0, 5); anim = t[..., 2].long().clamp(0, 3)
        num = t[..., 3:].float() / torch.tensor([30, 6, 1, 3, 3, 1, 1, 3, 1, 3, 8, 3, 1, 9, 9], device=t.device, dtype=torch.float32)
        tt = self.tile_in(torch.cat([self.kind_emb(kind), self.crop_emb(crop), self.anim_emb(anim), num], -1)) + self.tile_pos + self.type_emb.weight[1]
        u = units.float() / torch.tensor([1, 1, 9, 9] + [40] * 12 + [1, 16], device=units.device, dtype=torch.float32)
        ut = self.unit_in(u) + self.unit_pos + self.type_emb.weight[2]
        if prev is not None: ut = ut + self.prev_dest_emb(prev[..., 0].long().clamp(0, 100)) + self.prev_op_emb(prev[..., 1].long().clamp(0, 44))
        it = self.item_in(items) + self.item_pos + self.type_emb.weight[3]
        gt = (self.glob_in(glob) + self.type_emb.weight[0]).unsqueeze(1)
        x = torch.cat([gt, tt, ut, it], 1); present = units[..., 0] > 0
        pad = torch.zeros(B, x.shape[1], dtype=torch.bool, device=x.device); pad[:, 1 + N_TILES:1 + N_TILES + MAX_UNITS] = ~present
        h = self.norm(self.enc(x, src_key_padding_mask=pad))
        return {"g": h[:, 0], "own": h[:, 1:1 + N_OWN], "units": h[:, 1 + N_TILES:1 + N_TILES + MAX_UNITS], "items": h[:, 1 + N_TILES + MAX_UNITS:], "locked": (t[:, :N_OWN, 0] == 0)}

    def dest_logits(self, H):
        q = self.dq(H["units"]); k = self.dk(H["own"])
        return torch.einsum("bud,btd->but", q, k) / (self.d ** 0.5)  # raw; callers mask LOCKED tiles

    def op_logits(self, H, dest):
        """dest [B,U] long tile indices -> op/qty logits conditioned on the destination tile token."""
        B, U = dest.shape; idx = dest.clamp(min=0)
        tok = torch.gather(H["own"], 1, idx.unsqueeze(-1).expand(B, U, self.d))
        z = torch.cat([H["units"], tok], -1)
        return self.op_mlp(z), self.qty_mlp(z)

    def market(self, H):
        g, hi = H["g"], H["items"]; B = g.shape[0]
        return {"sell": self.sell_head(hi), "buyp": self.buyp_head(hi[:, [0, 8]]), "seed": self.seed_head(hi[:, :5]), "anim": self.anim_head(g).view(B, 3, 5),
                "hire": self.hire_head(g), "land": self.land_head(g), "value": self.value_head(g).squeeze(-1)}

    def forward(self, tiles, units, items, glob, dest=None, prev=None):
        H = self.encode(tiles, units, items, glob, prev); out = self.market(H); out["dest"] = self.dest_logits(H)
        if dest is None: dest = out["dest"].masked_fill(H["locked"].unsqueeze(1), -1e9).argmax(-1)
        out["op"], out["qty"] = self.op_logits(H, dest); out["H"] = H
        return out


def bc_loss2(out, batch):
    dest = batch["dest"].long(); present = dest >= 0; dop = batch["dop"].long()
    tgt_d = F.one_hot(dest.clamp(min=0), out["dest"].shape[-1]).bool() & present.unsqueeze(-1)
    allowed = (~out["H"]["locked"]).unsqueeze(1) | tgt_d  # a recorded destination is reachable even if it is a LOCKED tile
    dl = out["dest"].masked_fill(~allowed, -1e9)
    cur = (batch["units"][..., 3] * 10 + batch["units"][..., 2]).long()
    wmove = torch.where(dest == cur, 1.0, 3.0)[present]
    l_dest = (F.cross_entropy(dl[present], dest[present], reduction="none") * wmove).sum() / wmove.sum()
    dm = batch["dmask"].clone(); dm |= F.one_hot(dop.clamp(min=0), dm.shape[-1]).bool() & present.unsqueeze(-1)
    lg = out["op"].masked_fill(~dm, -1e9)
    l_op = F.cross_entropy(lg[present], dop[present], weight=OP_WEIGHT.to(lg.device) if OP_WEIGHT is not None else None)
    is_q = present & (((dop >= 16) & (dop <= 27)) | ((dop >= 29) & (dop <= 40)))
    l_qty = F.cross_entropy(out["qty"][is_q], batch["dqty"].long()[is_q]) if is_q.any() else lg.new_zeros(())
    tg = market_targets_split(batch["mkt"].long())
    l_sell = _mkt_ce(out["sell"], tg["sell"], N_MB); l_buyp = _mkt_ce(out["buyp"], tg["buyp"], N_MB); l_seed = _mkt_ce(out["seed"], tg["seed"], N_MB)
    l_anim = _mkt_ce(out["anim"], tg["anim"], 5, 20.0); l_hire = _mkt_ce(out["hire"], tg["hire"], 13, 8.0); l_land = _mkt_ce(out["land"], tg["land"], 2, 30.0)
    loss = l_dest + l_op + 0.3 * l_qty + l_sell + 0.5 * (l_buyp + l_seed + l_anim) + 0.5 * l_hire + 0.5 * l_land
    with torch.no_grad():
        acc = {"dest": (dl.argmax(-1) == dest)[present].float().mean().item(), "op": (lg.argmax(-1) == dop)[present].float().mean().item(),
               "sell": (out["sell"].argmax(-1) == tg["sell"]).float().mean().item(), "hire": (out["hire"].argmax(-1) == tg["hire"]).float().mean().item(),
               "seed": (out["seed"].argmax(-1) == tg["seed"]).float().mean().item(), "anim": (out["anim"].argmax(-1) == tg["anim"]).float().mean().item()}
    return loss, acc
