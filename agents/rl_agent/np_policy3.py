"""numpy forward for Policy3 (deployment). Reuses NpPolicy2.encode/market (identical
architecture); adds the joint (destination, operation) option head + qty head.

Mirrors model3.Policy3.option_logits / qty_logits exactly.
"""
from __future__ import annotations

import numpy as np

from np_policy import _gelu
from np_policy2 import NpPolicy2

OPTION_RANK = 16


class NpPolicy3(NpPolicy2):
    def option_logits(self, H):
        W = self.W
        q = H["units"] @ W["option_q.weight"].T + W["option_q.bias"]
        k = H["own"] @ W["option_k.weight"].T + W["option_k.bias"]
        q = q.reshape(-1, q.shape[-1] // OPTION_RANK, OPTION_RANK)
        k = k.reshape(-1, k.shape[-1] // OPTION_RANK, OPTION_RANK)
        scores = np.einsum("uor,tor->uto", q, k) / np.sqrt(OPTION_RANK)
        return scores + (H["units"] @ W["option_bias.weight"].T + W["option_bias.bias"])[:, None, :]

    def qty_logits(self, H, dest):
        z = np.concatenate([H["units"], H["own"][np.clip(dest, 0, 99)]], -1)
        return self.lin(_gelu(self.lin(z, "qty_mlp.0")), "qty_mlp.2")
