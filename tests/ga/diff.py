"""2つのテープ集合の差分を分類して要約。使い方: .venv/bin/python tests/ga/diff.py tmp/ga/tapes_guarded.json tmp/ga/best_run3.json"""
import json, sys
from collections import Counter
A = json.load(open(sys.argv[1])); B = json.load(open(sys.argv[2]))
c = Counter(); splice = []
for k in range(len(A)):
    diff_steps = [t for t in range(719) if A[k][t] != B[k][t]]
    if len(diff_steps) > 150: splice.append((k, diff_steps[0], len(diff_steps)))
    for t in diff_steps:
        a, b = A[k][t], B[k][t]
        ua = [a.get("farmer") or [], *(a.get("hands") or [])]; ub = [b.get("farmer") or [], *(b.get("hands") or [])]
        for x, y in zip(ua, ub):
            if x != y: c[f"unit {x[0] if x else '-'}->{y[0] if y else '-'}"] += 1
        ma = [tuple(o) for o in a.get("market") or [] if o]; mb = [tuple(o) for o in b.get("market") or [] if o]
        for o in set(mb) - set(ma): c[f"market +{o[0]} {o[1] if len(o)>1 else ''}"] += 1
        for o in set(ma) - set(mb): c[f"market -{o[0]} {o[1] if len(o)>1 else ''}"] += 1
print("plans with wholesale splice (plan, from step, n diffs):", splice)
for k, v in c.most_common(30): print(f"{v:5d}  {k}")
