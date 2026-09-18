"""Batched action selection for Policy2 over VecEnv slots: dest with reselect-as-mask + claiming, op at destination, market heads.
Returns actions (written into buffers) plus everything PPO needs to recompute log-probs (masks, logp per head, value)."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from features import MAX_UNITS, N_OPS, OP_INDEX, SHED_TILES

TOPK = 8
COMMIT = True  # keep destination until arrival
PASS = OP_INDEX["PASS"]; SHED_IDX = np.array([y * 10 + x for (x, y) in SHED_TILES])
HEADS = ("sell", "buyp", "seed", "anim", "hire", "land")


def _sample(logits, greedy, gen):
    """logits torch [..., C] already masked with -1e9. Returns (idx, logp)."""
    if greedy: idx = logits.argmax(-1)
    else: idx = torch.distributions.Categorical(logits=logits).sample()
    lp = torch.log_softmax(logits, -1).gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    return idx, lp


@torch.no_grad()
def act_batch(model, dev, A, slots, prev, temp=1.0, greedy=False, greedy_dest=False):
    """A = VecEnv buffers, slots = np.array of slot indices handled by this model. prev: int16 [n_slots, MAX_UNITS, 2] (previous dest/op per slot).
    Writes act_dest/act_op/act_qty/act_mkt for those slots. Returns dict of numpy arrays aligned with `slots` for PPO storage.
    greedy_dest: 目的地だけ貪欲に固定し、探索を作業・数量・市場ヘッドに限る。log-prob を 0 にするので PPO 側は
    `live = (old != 0)` で自動的に除外される (committed と同じ扱い)。得点は目的地ノイズに (1-eps)^7.6 で落ちる
    ので、per-step の目的地サンプリングは探索の代償が大きすぎる (experiments.md の 4d 行)。"""
    S = len(slots)
    tiles = torch.from_numpy(A["tiles"][slots]).to(dev); units = torch.from_numpy(A["units"][slots]).to(dev); items = torch.from_numpy(A["items"][slots]).to(dev); glob = torch.from_numpy(A["glob"][slots]).to(dev)
    prev_t = torch.from_numpy(prev[slots]).to(dev)
    H = model.encode(tiles, units, items, glob, prev_t)
    dl = model.dest_logits(H) / temp  # [S,U,100]
    locked = H["locked"]  # [S,100]
    present_np = A["units"][slots][:, :, 0] > 0; present = torch.from_numpy(present_np).to(dev)
    dl = dl.masked_fill(locked.unsqueeze(1), -1e9)
    # candidate destinations = top-K by logit; arrival op is evaluated only there (op head on K tiles instead of 100)
    top_lg, top_idx = dl.topk(TOPK, dim=-1)  # [S,U,K]
    d = H["units"].shape[-1]
    own_k = torch.gather(H["own"].unsqueeze(1).expand(S, MAX_UNITS, 100, d), 2, top_idx.unsqueeze(-1).expand(S, MAX_UNITS, TOPK, d))
    op_k = model.op_mlp(torch.cat([H["units"].unsqueeze(2).expand(S, MAX_UNITS, TOPK, d), own_k], -1))  # [S,U,K,N_OPS]
    legal = torch.from_numpy(A["legal"][slots]).to(dev)  # [S,U,100,N_OPS]
    legal_k = torch.gather(legal, 2, top_idx.unsqueeze(-1).expand(S, MAX_UNITS, TOPK, N_OPS))
    productive = op_k.masked_fill(~legal_k, -1e9).argmax(-1) != PASS  # [S,U,K]
    productive = productive | ~productive.any(-1, keepdim=True)  # nothing productive among candidates: allow them all
    # CPU: sequential claiming (farmer first) + sampling over the K candidates. A unit keeps its committed destination until it arrives
    # (decide only at arrival / start of day): fewer decisions per game, far less credit-assignment noise.
    top_lg = top_lg.cpu().numpy(); top_idx_np = top_idx.cpu().numpy(); prod_np = productive.cpu().numpy()
    pos_np = A["pos"][slots].astype(np.int64); prev_np = prev[slots]
    committed = present_np & (prev_np[:, :, 0] < 100) & (prev_np[:, :, 0] != pos_np) if COMMIT else np.zeros_like(present_np)
    dest_np = np.zeros((S, MAX_UNITS), dtype=np.int64); dmask = np.zeros((S, MAX_UNITS, 100), dtype=bool); lp_dest = np.zeros((S, MAX_UNITS), dtype=np.float32)
    claimed = np.zeros((S, 100), dtype=bool); shed = np.zeros(100, dtype=bool); shed[SHED_IDX] = True; rng = np.random.default_rng()
    for i in range(MAX_UNITS):
        cand = top_idx_np[:, i]; ok = prod_np[:, i] & ~np.take_along_axis(claimed, cand, 1)
        ok |= ~ok.any(1, keepdims=True) & prod_np[:, i]  # every productive candidate claimed: ignore claims
        ok |= ~ok.any(1, keepdims=True)
        lg = np.where(ok, top_lg[:, i], -1e9); lg = lg - lg.max(1, keepdims=True); p = np.exp(lg); p /= p.sum(1, keepdims=True)
        if greedy or greedy_dest: j = p.argmax(1)
        else: j = (p.cumsum(1) > rng.random((S, 1))).argmax(1)
        dest_np[:, i] = np.where(committed[:, i], prev_np[:, i, 0], cand[np.arange(S), j])
        lp_dest[:, i] = 0.0 if greedy_dest else np.where(committed[:, i], 0.0, np.log(p[np.arange(S), j] + 1e-12))
        np.put_along_axis(dmask[:, i], cand, ok, 1)
        claim = present_np[:, i] & ~shed[dest_np[:, i]]; claimed[np.arange(S)[claim], dest_np[claim, i]] = True
    decide = present_np & ~committed
    lp_dest = torch.from_numpy(lp_dest * present_np).to(dev); dest = torch.from_numpy(dest_np).to(dev); dmask = torch.from_numpy(dmask)
    # op at destination
    opl, qtl = model.op_logits(H, dest); opl = opl / temp
    omask = torch.gather(legal, 2, dest.view(S, MAX_UNITS, 1, 1).expand(S, MAX_UNITS, 1, N_OPS)).squeeze(2)
    op, lp_op = _sample(opl.masked_fill(~omask, -1e9), greedy, None)
    pos = torch.from_numpy(A["pos"][slots].astype(np.int64)).to(dev); at_dest = (pos == dest) & present
    lp_op = lp_op * at_dest
    qty, lp_qty = _sample(qtl / temp, greedy, None)
    is_q = ((op >= 16) & (op <= 27)) | ((op >= 29) & (op <= 40)); lp_qty = lp_qty * at_dest * is_q
    mk = model.market(H); mk_idx = {}; lp_mk = []; lp_mk_el = {}
    for k in HEADS:
        idx, lp = _sample(mk[k] / temp, greedy, None); mk_idx[k] = idx; lp_mk.append(lp.reshape(S, -1).sum(-1)); lp_mk_el[k] = lp.reshape(S, -1) if k not in ("hire", "land") else lp.reshape(S)
    mkt = torch.cat([mk_idx["sell"], mk_idx["buyp"], mk_idx["seed"], mk_idx["anim"], mk_idx["hire"].unsqueeze(-1), mk_idx["land"].unsqueeze(-1)], -1)
    A["act_dest"][slots] = dest.cpu().numpy().astype(np.int16); A["act_op"][slots] = op.cpu().numpy().astype(np.int16); A["act_qty"][slots] = qty.cpu().numpy().astype(np.int16); A["act_mkt"][slots] = mkt.cpu().numpy().astype(np.int16)
    logp = lp_dest.sum(-1) + lp_op.sum(-1) + lp_qty.sum(-1) + sum(lp_mk)
    logp_h = torch.stack([lp_dest.sum(-1), lp_op.sum(-1), lp_qty.sum(-1), *lp_mk], -1)  # [S,9] per-head log-probs
    # per-decision log-probs [S,69]: dest(16) op(16) qty(16) sell(9) buyp(2) seed(5) anim(3) hire(1) land(1); zeros = no decision taken
    lp_dec = torch.cat([lp_dest, lp_op, lp_qty, lp_mk_el["sell"], lp_mk_el["buyp"], lp_mk_el["seed"], lp_mk_el["anim"], lp_mk_el["hire"].unsqueeze(-1), lp_mk_el["land"].unsqueeze(-1)], -1)
    return {"dest": dest.cpu().numpy().astype(np.int16), "op": op.cpu().numpy().astype(np.int16), "qty": qty.cpu().numpy().astype(np.int16), "mkt": mkt.cpu().numpy().astype(np.int16),
            "decide": decide, "dmask": dmask.numpy(), "omask": omask.cpu().numpy(), "at_dest": at_dest.cpu().numpy(), "present": present.cpu().numpy(),
            "logp": logp.cpu().numpy().astype(np.float32), "logp_h": logp_h.cpu().numpy().astype(np.float32), "lp_dec": lp_dec.cpu().numpy().astype(np.float32), "value": mk["value"].cpu().numpy().astype(np.float32)}
