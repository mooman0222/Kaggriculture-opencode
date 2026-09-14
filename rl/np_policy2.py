"""numpy forward for Policy2 (deployment). Mirrors model2.Policy2.encode / dest_logits / op_logits / market."""
from __future__ import annotations
import numpy as np
from np_policy import _ln, _gelu, TILE_SCALE, UNIT_SCALE, N_TILES, N_ITEMS
from features import TILE_F, MAX_UNITS


class NpPolicy2:
    def __init__(self, W, d=128, layers=3, heads=4): self.W = W; self.d = d; self.L = layers; self.h = heads
    def lin(self, x, name): return x @ self.W[name + ".weight"].T + self.W[name + ".bias"]

    def encode(self, f, prev=None):
        W = self.W; d = self.d; t = f["tiles"].reshape(N_TILES, TILE_F)
        kind = np.clip(t[:, 0], 0, 7); crop = np.clip(t[:, 1], 0, 5); anim = np.clip(t[:, 2], 0, 3); num = t[:, 3:].astype(np.float32) / TILE_SCALE
        tt = self.lin(np.concatenate([W["kind_emb.weight"][kind], W["crop_emb.weight"][crop], W["anim_emb.weight"][anim], num], -1), "tile_in") + W["tile_pos"] + W["type_emb.weight"][1]
        ut = self.lin(f["units"].astype(np.float32) / UNIT_SCALE, "unit_in") + W["unit_pos"] + W["type_emb.weight"][2]
        if prev is not None and "prev_dest_emb.weight" in W: ut = ut + W["prev_dest_emb.weight"][np.clip(prev[:, 0], 0, 100)] + W["prev_op_emb.weight"][np.clip(prev[:, 1], 0, 44)]
        it = self.lin(f["items"], "item_in") + W["item_pos"] + W["type_emb.weight"][3]
        gt = (self.lin(f["glob"], "glob_in") + W["type_emb.weight"][0])[None]
        x = np.concatenate([gt, tt, ut, it], 0); present = f["units"][:, 0] > 0
        pad = np.zeros(x.shape[0], dtype=bool); pad[1 + N_TILES:1 + N_TILES + MAX_UNITS] = ~present
        for l in range(self.L):
            p = f"enc.layers.{l}."; hN = _ln(x, W[p + "norm1.weight"], W[p + "norm1.bias"])
            qkv = hN @ W[p + "self_attn.in_proj_weight"].T + W[p + "self_attn.in_proj_bias"]; q, k, v = np.split(qkv, 3, -1); T = x.shape[0]; hd = d // self.h
            q = q.reshape(T, self.h, hd).transpose(1, 0, 2); k = k.reshape(T, self.h, hd).transpose(1, 0, 2); v = v.reshape(T, self.h, hd).transpose(1, 0, 2)
            s = q @ k.transpose(0, 2, 1) / np.sqrt(hd); s[:, :, pad] = -1e9; s = s - s.max(-1, keepdims=True); a = np.exp(s); a /= a.sum(-1, keepdims=True)
            x = x + (a @ v).transpose(1, 0, 2).reshape(T, d) @ W[p + "self_attn.out_proj.weight"].T + W[p + "self_attn.out_proj.bias"]
            hN = _ln(x, W[p + "norm2.weight"], W[p + "norm2.bias"]); x = x + self.lin(_gelu(self.lin(hN, p + "linear1")), p + "linear2")
        h = _ln(x, W["norm.weight"], W["norm.bias"])
        return {"g": h[0], "own": h[1:101], "units": h[1 + N_TILES:1 + N_TILES + MAX_UNITS], "items": h[1 + N_TILES + MAX_UNITS:], "locked": t[:100, 0] == 0}

    def dest_logits(self, H):
        lg = (self.lin(H["units"], "dq") @ self.lin(H["own"], "dk").T) / np.sqrt(self.d); lg[:, H["locked"]] = -1e9; return lg

    def op_logits(self, H, dest):
        z = np.concatenate([H["units"], H["own"][np.clip(dest, 0, 99)]], -1)
        f1 = lambda x, n: self.lin(_gelu(self.lin(x, n + ".0")), n + ".2")
        return f1(z, "op_mlp"), f1(z, "qty_mlp")

    def market(self, H):
        g, hi = H["g"], H["items"]
        return {"sell": self.lin(hi, "sell_head"), "buyp": self.lin(hi[[0, 8]], "buyp_head"), "seed": self.lin(hi[:5], "seed_head"), "anim": self.lin(g, "anim_head").reshape(3, 5),
                "hire": self.lin(g, "hire_head"), "land": self.lin(g, "land_head"), "value": self.lin(g, "value_head")[0]}
