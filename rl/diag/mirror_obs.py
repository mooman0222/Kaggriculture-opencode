"""Copycat probe for BC trained on mirror self-play: at inference, overwrite the opponent-farm features with our own
(training data had own == opp exactly). If closed-loop play recovers, the policy was reading the opponent's farm.
usage (repo root): .venv/bin/python rl/diag/mirror_obs.py CKPT [games]"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), "rl"))
import act3, features
import play3

_encode = features.encode


def mirrored(obs, seat):
    f = _encode(obs, seat)
    t = f["tiles"]; own = t[0].copy(); own[..., 15] = 1; own[..., 14] = 0; t[1] = own
    f["items"][:, 6] = f["items"][:, 5]
    g = f["glob"]; g[7], g[9], g[11] = g[6], g[8], g[10]
    return f


if __name__ == "__main__":
    ck = sys.argv[1]; games = sys.argv[2] if len(sys.argv) > 2 else "16"
    if os.environ.get("MIRROR", "1") == "1":
        act3.encode = mirrored
    sys.argv = ["play3.py", ck, "--games", games]
    play3.main()
