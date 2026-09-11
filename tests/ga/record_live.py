"""反応するクローン (0909 base / E055) の行動列を、我々のテープ相手に L1 で記録してプールに追加する。
使い方: .venv/bin/python tests/ga/record_live.py --tapes tmp/ga/tapes_guarded.json --seeds 32 --out tmp/ga/pool_live.json"""
import argparse, importlib.util, json, os, sys
import kagsim
sys.path.insert(0, "tests/ga"); from fitness import build_stream
PASS = {"farmer": ["PASS"], "hands": [], "market": []}

def load(p):
    d = os.path.dirname(os.path.abspath(p)); sys.path.insert(0, d)
    s = importlib.util.spec_from_file_location("rec_" + os.path.basename(d), p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

ap = argparse.ArgumentParser(); ap.add_argument("--tapes", required=True); ap.add_argument("--seeds", type=int, default=32); ap.add_argument("--seed0", type=int, default=2000); ap.add_argument("--out", required=True)
a = ap.parse_args(); tapes = json.load(open(a.tapes))
pool = []
for label, path in (("SR0909live", "agents/sr0909_base/main.py"), ("E055live", "agents/sr0909_live/main.py")):
    mod = load(path)
    for seed in range(a.seed0, a.seed0 + a.seeds):
        for opp_seat in (0, 1):
            for k in ("_LIVE", "_POLICY"):
                if hasattr(mod, k): setattr(mod, k, None)
            g = kagsim.Game(seed); shops = None; rec = []
            while not g.done:
                t = g.step_count; o = g.observe(opp_seat)
                if t == 144: shops = list(o["town"]["unlocked_shops"])
                # our side plays the world-routed tape (needs shops at 144: use the game's own town)
                ours_shops = list(g.observe(1 - opp_seat)["town"]["unlocked_shops"])
                mine = build_stream(tapes, ours_shops)[t] if t >= 144 else tapes[0][t]
                oa = mod.agent(o); rec.append(oa)
                g.step(*( (oa, mine) if opp_seat == 0 else (mine, oa) ))
            pool.append({"episode": f"{label}-{seed}-{opp_seat}", "seat": opp_seat, "team": label, "lineage": label, "h72": "",
                         "seed": seed, "shops": list(g.observe(0)["town"]["unlocked_shops"]), "bank": g.reward(opp_seat), "stream": rec})
    print(label, "recorded", 2 * a.seeds, flush=True)
json.dump(pool, open(a.out, "w")); print(len(pool), "entries ->", a.out)
