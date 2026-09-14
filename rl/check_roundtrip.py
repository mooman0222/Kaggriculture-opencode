"""Sanity: recorded top-team actions must be legal under our masks and survive encode->decode (quantities up to bucketing)."""
import json, glob, sys, collections
sys.path.insert(0, "rl")
import numpy as np
from features import encode, legal_ops, OPS
from actions import encode_action, decode_action
team = sys.argv[2] if len(sys.argv) > 2 else "M & M & P & Q"
viol = collections.Counter(); total = 0; mkt_mismatch = 0; unit_mismatch = 0; steps = 0
for f in sorted(glob.glob(sys.argv[1]))[:6]:
    r = json.load(open(f)); names = r["info"]["TeamNames"]
    if team not in names: continue
    seat = names.index(team); S = r["steps"]
    for t in range(0, 719):
        obs = S[t][seat]["observation"]; obs["step"] = t
        act = S[t + 1][seat]["action"] or {}
        n = 1 + len(obs["farms"][seat]["hands"])
        op, qty, mkt = encode_action(act, n); M = legal_ops(obs, seat); steps += 1
        for i in range(n):
            total += 1
            if op[i] >= 0 and not M[i, op[i]]: viol[OPS[op[i]]] += 1
        dec = decode_action(op, qty, mkt, obs, seat)
        acts = [act.get("farmer") or ["PASS"], *(act.get("hands") or [])]
        for i in range(n):
            a = acts[i] if i < len(acts) else ["PASS"]; d = [dec["farmer"], *dec["hands"]][i]
            if (a[0] if a else "PASS") != d[0] or (len(a) > 1 and a[1] != (d[1] if len(d) > 1 else None)): unit_mismatch += 1
        so = sorted(json.dumps(o) for o in act.get("market") or [] if o); sd = sorted(json.dumps(o) for o in dec["market"])
        if so != sd: mkt_mismatch += 1
    feats = encode(obs, seat); print(f, "feature shapes", {k: v.shape for k, v in feats.items()})
print(f"steps {steps} unit-actions {total} mask violations {sum(viol.values())} {dict(viol.most_common(8))}")
print(f"unit op mismatches after roundtrip {unit_mismatch}, market mismatches (order/qty bucketing) {mkt_mismatch}/{steps}")
