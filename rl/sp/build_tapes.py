"""Real-replay opponent pool: (seed, seat, 719 actions) per recorded player → tmp/rl/tapes.pkl. Usage: .venv/bin/python rl/sp/build_tapes.py 'tmp/e061_live/*.json' 'tmp/e060_live/*.json' 'tmp/top0909/**/episode-*.json' 'tmp/top0911/**/episode-*.json'"""
import glob, json, pickle, sys
PASS = {"farmer": ["PASS"], "hands": [], "market": []}
tapes = []; seen = set()
for pat in sys.argv[1:]:
    for f in sorted(glob.glob(pat, recursive=True)):
        try: r = json.load(open(f))
        except Exception: continue
        if not isinstance(r, dict) or "steps" not in r: continue
        seed = r.get("info", {}).get("seed"); names = r.get("info", {}).get("TeamNames") or ["?", "?"]
        if not seed or len(r.get("steps", [])) != 720 or (r["id"], 0) in seen: continue
        for p in (0, 1):
            acts = [r["steps"][t][p]["action"] or PASS for t in range(1, 720)]
            if all(a.get("farmer") in (None, ["PASS"]) and not a.get("hands") for a in acts[:48]): continue  # dead seat
            tapes.append({"seed": int(seed), "seat": p, "team": names[p], "final": float(r["steps"][-1][p]["observation"]["farms"][p]["money"]), "acts": acts}); seen.add((r["id"], 0))
pickle.dump(tapes, open("tmp/rl/tapes.pkl", "wb"))
from collections import Counter
print(f"{len(tapes)} tapes from {len(seen)} replays; top teams: {Counter(t['team'] for t in tapes).most_common(8)}")
