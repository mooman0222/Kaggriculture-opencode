"""numpy-only forward pass of rl/model.Policy (for the Kaggle agent: no torch at inference). Weights from export_weights()."""
from __future__ import annotations
import numpy as np
from features import TILE_F, MAX_UNITS, PRODUCTS

N_TILES = 200; N_ITEMS = len(PRODUCTS)
TILE_SCALE = np.array([30, 6, 1, 3, 3, 1, 1, 3, 1, 3, 8, 3, 1, 9, 9], dtype=np.float32)
UNIT_SCALE = np.array([1, 1, 9, 9] + [40] * 12 + [1, 16], dtype=np.float32)


def export_weights(state_dict):
    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in state_dict.items()}


def _ln(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True); return (x - m) / np.sqrt(v + eps) * w + b


try:
    from scipy.special import erf as _erf
    _HAS_ERF = True
except Exception:
    _erf = None
    _HAS_ERF = False


def _gelu(x):
    from math import sqrt
    # torch default gelu (erf); tanh approximation fallback when scipy is absent
    if _HAS_ERF:
        return 0.5 * x * (1 + _erf(x / sqrt(2)))
    return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


class NpPolicy:
    def __init__(self, W, d=128, layers=3, heads=4):
        self.W = W; self.d = d; self.L = layers; self.h = heads

    def lin(self, x, name): return x @ self.W[name + ".weight"].T + self.W[name + ".bias"]

    def forward(self, f):
        W = self.W; d = self.d
        t = f["tiles"].reshape(N_TILES, TILE_F)
        kind = np.clip(t[:, 0], 0, 7); crop = np.clip(t[:, 1], 0, 5); anim = np.clip(t[:, 2], 0, 3)
        num = t[:, 3:].astype(np.float32) / TILE_SCALE
        tt = self.lin(np.concatenate([W["kind_emb.weight"][kind], W["crop_emb.weight"][crop], W["anim_emb.weight"][anim], num], -1), "tile_in") + W["tile_pos"] + W["type_emb.weight"][1]
        u = f["units"].astype(np.float32) / UNIT_SCALE
        ut = self.lin(u, "unit_in") + W["unit_pos"] + W["type_emb.weight"][2]
        it = self.lin(f["items"], "item_in") + W["item_pos"] + W["type_emb.weight"][3]
        gt = (self.lin(f["glob"], "glob_in") + W["type_emb.weight"][0])[None]
        x = np.concatenate([gt, tt, ut, it], 0)
        present = f["units"][:, 0] > 0
        pad = np.zeros(x.shape[0], dtype=bool); pad[1 + N_TILES:1 + N_TILES + MAX_UNITS] = ~present
        for l in range(self.L):
            p = f"enc.layers.{l}."
            hN = _ln(x, W[p + "norm1.weight"], W[p + "norm1.bias"])
            qkv = hN @ W[p + "self_attn.in_proj_weight"].T + W[p + "self_attn.in_proj_bias"]
            q, k, v = np.split(qkv, 3, -1); T = x.shape[0]; hd = d // self.h
            q = q.reshape(T, self.h, hd).transpose(1, 0, 2); k = k.reshape(T, self.h, hd).transpose(1, 0, 2); v = v.reshape(T, self.h, hd).transpose(1, 0, 2)
            s = q @ k.transpose(0, 2, 1) / np.sqrt(hd); s[:, :, pad] = -1e9
            s = s - s.max(-1, keepdims=True); a = np.exp(s); a /= a.sum(-1, keepdims=True)
            o = (a @ v).transpose(1, 0, 2).reshape(T, d)
            x = x + o @ W[p + "self_attn.out_proj.weight"].T + W[p + "self_attn.out_proj.bias"]
            hN = _ln(x, W[p + "norm2.weight"], W[p + "norm2.bias"])
            x = x + self.lin(_gelu(self.lin(hN, p + "linear1")), p + "linear2")
        h = _ln(x, W["norm.weight"], W["norm.bias"])
        g = h[0]; hu = h[1 + N_TILES:1 + N_TILES + MAX_UNITS]; hi = h[1 + N_TILES + MAX_UNITS:]
        return {"op": self.lin(hu, "op_head"), "qty": self.lin(hu, "qty_head"), "sell": self.lin(hi, "sell_head"), "buyp": self.lin(hi[[0, 8]], "buyp_head"),
                "seed": self.lin(hi[:5], "seed_head"), "anim": self.lin(g, "anim_head").reshape(3, 5), "hire": self.lin(g, "hire_head"), "land": self.lin(g, "land_head"), "value": self.lin(g, "value_head")[0]}
