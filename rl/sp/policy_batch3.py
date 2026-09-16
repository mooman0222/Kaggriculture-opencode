"""Batched action selection for Policy3 over VecEnv slots.

One decision per unit = a joint (tile, op) option among the top-K exact-legal candidates; a chosen remote option is held until
arrival (no re-decision while walking), tiles are claimed farmer-first. Returns everything PPO needs to recompute log-probs:
candidate option indices, the candidate mask, per-decision log-probs and the value.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from features import MAX_UNITS, N_OPS, OP_INDEX, SHED_TILES

TOPK = 64
PASS = OP_INDEX["PASS"]
SHED_IDX = np.array([y * 10 + x for (x, y) in SHED_TILES])
HEADS = ("sell", "buyp", "seed", "anim", "hire", "land")
QTY_OPS = lambda op: ((op >= 16) & (op <= 27)) | ((op >= 29) & (op <= 40))


def _sample(logits, greedy):
    idx = logits.argmax(-1) if greedy else torch.distributions.Categorical(logits=logits).sample()
    return idx, torch.log_softmax(logits, -1).gather(-1, idx.unsqueeze(-1)).squeeze(-1)


def option_legal(A, slots, dev):
    """Exact arrival legality as an option mask [S,U,100,N_OPS]: moves are never options, PASS only on the unit's own tile."""
    legal = torch.from_numpy(A["legal"][slots]).to(dev).clone()
    legal[..., 1:5] = False
    cur = torch.from_numpy(np.clip(A["pos"][slots].astype(np.int64), 0, 99)).to(dev)
    on_tile = torch.zeros(legal.shape[:3], dtype=torch.bool, device=dev).scatter_(2, cur.unsqueeze(-1), True)
    legal[..., PASS] &= on_tile
    return legal, cur


@torch.no_grad()
def act_batch3(model, dev, A, slots, held, temp=1.0, greedy=False):
    """held: int32 [n_slots, MAX_UNITS] option index kept from earlier steps (-1 none); updated in place for `slots`."""
    S = len(slots)
    tiles = torch.from_numpy(A["tiles"][slots]).to(dev)
    units = torch.from_numpy(A["units"][slots]).to(dev)
    items = torch.from_numpy(A["items"][slots]).to(dev)
    glob = torch.from_numpy(A["glob"][slots]).to(dev)
    H = model.encode(tiles, units, items, glob)
    logits_all = (model.option_logits(H) / temp).reshape(S, MAX_UNITS, -1)  # [S,U,4400]
    legal, cur = option_legal(A, slots, dev)
    present = units[..., 0] > 0
    pass_here = cur * N_OPS + PASS
    held_t = torch.from_numpy(held[slots].astype(np.int64)).to(dev)
    claimed = torch.zeros(S, 100, dtype=torch.bool, device=dev)
    shed = torch.zeros(100, dtype=torch.bool, device=dev)
    shed[SHED_IDX] = True
    option = torch.zeros(S, MAX_UNITS, dtype=torch.long, device=dev)
    cand = torch.zeros(S, MAX_UNITS, TOPK, dtype=torch.long, device=dev)
    cmask = torch.zeros(S, MAX_UNITS, TOPK, dtype=torch.bool, device=dev)
    lp_opt = torch.zeros(S, MAX_UNITS, device=dev)
    decide = torch.zeros(S, MAX_UNITS, dtype=torch.bool, device=dev)
    rows = torch.arange(S, device=dev)
    for i in range(MAX_UNITS):
        mask = (legal[:, i] & ~claimed.unsqueeze(-1)).reshape(S, -1)
        mask[rows, pass_here[:, i]] = True  # standing still is always possible
        h = held_t[:, i]
        keep = (h >= 0) & mask.gather(1, h.clamp(min=0).unsqueeze(1)).squeeze(1) & present[:, i]
        masked = logits_all[:, i].masked_fill(~mask, -1e9)
        top_lg, top_idx = masked.topk(TOPK, dim=-1)
        top_ok = top_lg > -1e8
        idx, lp = _sample(top_lg.masked_fill(~top_ok, -1e9), greedy)
        chosen = top_idx.gather(1, idx.unsqueeze(1)).squeeze(1)
        option[:, i] = torch.where(keep, h, chosen)
        lp_opt[:, i] = torch.where(keep, torch.zeros_like(lp), lp) * present[:, i]
        decide[:, i] = present[:, i] & ~keep
        cand[:, i] = top_idx
        cmask[:, i] = top_ok
        tile = option[:, i] // N_OPS
        claimed[rows[present[:, i] & ~shed[tile]], tile[present[:, i] & ~shed[tile]]] = True
    dest = option // N_OPS
    op = option % N_OPS
    at_dest = (cur == dest) & present
    qty_logits = model.qty_logits(H, dest) / temp
    qty, lp_qty = _sample(qty_logits, greedy)
    lp_qty = lp_qty * (at_dest & QTY_OPS(op))
    market = model.market(H)
    mk_idx, lp_mk = {}, {}
    for k in HEADS:
        mk_idx[k], lp_mk[k] = _sample(market[k] / temp, greedy)
    mkt = torch.cat([mk_idx["sell"], mk_idx["buyp"], mk_idx["seed"], mk_idx["anim"], mk_idx["hire"].unsqueeze(-1), mk_idx["land"].unsqueeze(-1)], -1)
    lp_dec = torch.cat([lp_opt, lp_qty, lp_mk["sell"], lp_mk["buyp"], lp_mk["seed"], lp_mk["anim"], lp_mk["hire"].unsqueeze(-1), lp_mk["land"].unsqueeze(-1)], -1)
    A["act_dest"][slots] = dest.cpu().numpy().astype(np.int16)
    A["act_op"][slots] = op.cpu().numpy().astype(np.int16)
    A["act_qty"][slots] = qty.cpu().numpy().astype(np.int16)
    A["act_mkt"][slots] = mkt.cpu().numpy().astype(np.int16)
    held[slots] = np.where((present & ~at_dest).cpu().numpy(), option.cpu().numpy(), -1).astype(np.int32)
    return {
        "option": option.cpu().numpy().astype(np.int16), "cand": cand.cpu().numpy().astype(np.int16), "cmask": cmask.cpu().numpy(),
        "qty": qty.cpu().numpy().astype(np.int16), "mkt": mkt.cpu().numpy().astype(np.int16), "decide": decide.cpu().numpy(),
        "at_dest": at_dest.cpu().numpy(), "present": present.cpu().numpy(),
        "lp_dec": lp_dec.cpu().numpy().astype(np.float32), "value": market["value"].cpu().numpy().astype(np.float32),
    }


def dists_batch3(model, bt, temp):
    """Log-distributions restricted to the stored candidates (option), plus qty and market heads; per-decision log-probs [B, 16+16+21]; value."""
    H = model.encode(bt["tiles"], bt["units"], bt["items"], bt["glob"])
    logits_all = (model.option_logits(H) / temp).reshape(bt["tiles"].shape[0], MAX_UNITS, -1)
    cand_lg = logits_all.gather(-1, bt["cand"]).masked_fill(~bt["cmask"], -1e9)
    lo = torch.log_softmax(cand_lg, -1)  # [B,U,K]
    pick = (bt["cand"] == bt["option"].unsqueeze(-1)).float().argmax(-1)  # index of the taken option among candidates
    decide = bt["decide"].float()
    dest = bt["option"] // N_OPS
    op = bt["option"] % N_OPS
    lq = torch.log_softmax(model.qty_logits(H, dest) / temp, -1)
    isq = (bt["at_dest"] & QTY_OPS(op)).float()
    market = model.market(H)
    ld = {"option": lo, "qty": lq}
    for k in HEADS:
        ld[k] = torch.log_softmax(market[k] / temp, -1)
    g = lambda lp, idx: lp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    lp = [g(lo, pick) * decide, g(lq, bt["qty"]) * isq]
    base = 0
    for k, n in (("sell", 9), ("buyp", 2), ("seed", 5), ("anim", 3)):
        lp.append(g(ld[k], bt["mkt"][:, base:base + n]))
        base += n
    for k in ("hire", "land"):
        lp.append(g(ld[k], bt["mkt"][:, base]).unsqueeze(-1))
        base += 1
    entropy = (-(lo.exp() * lo.clamp(min=-30)).sum(-1) * decide).sum(-1) / decide.sum(-1).clamp(min=1)
    return ld, torch.cat(lp, -1), entropy, market["value"]


def kl_full3(ld_t, ld_s, bt):
    """Analytic forward KL(teacher || student) over the candidate options (decided units), qty (at destination) and market heads."""
    decide = bt["decide"].float()
    at = bt["at_dest"].float()
    total, count = 0.0, 0.0
    for k, w in (("option", decide), ("qty", at)):
        kl = (ld_t[k].exp() * (ld_t[k].clamp(min=-30) - ld_s[k].clamp(min=-30))).sum(-1)
        total = total + (kl * w).sum()
        count = count + w.sum()
    for k in HEADS:
        kl = (ld_t[k].exp() * (ld_t[k].clamp(min=-30) - ld_s[k].clamp(min=-30))).sum(-1)
        total = total + kl.sum()
        count = count + float(kl.numel())
    return total / count.clamp(min=1)
