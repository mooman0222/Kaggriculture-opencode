"""プランナーのパラメータをランダム探索: 各 config を env で渡し kag_eval (対 v41、両席) を回してログする。
使い方: .venv/bin/python tests/live_sweep.py --n 40 --switch 432 --games 16 --out tmp/e058/sweep.log"""
import argparse, json, os, random, subprocess, re, sys
SPACE = {"LE_HANDS": ["8,9,9,10,11", "8,9,9,10,12", "8,9,10,11,12", "8,9,9,10,13"], "LE_HANDS_LATE": ["10,9", "11,10", "12,10"],
         "LE_HERD": ["4", "5", "6"], "LE_CARE_HOUR": ["6", "10", "14"], "LE_DROP": ["3", "4", "6", "8"], "LE_TPP": ["10", "14", "20"],
         "LE_FLOAD": ["8", "12", "16"], "LE_FRES": ["12", "24", "40"], "LE_WATER_ALL": ["0", "1"], "LE_WANDER": ["2", "3", "6", "99"],
         "LE_STRAW": ["4", "6", "10"], "LE_STRAW_PER": ["3", "5", "7"], "LE_WHEAT_PER": ["2", "4", "6"], "LE_MATCH": ["0", "1", "2"], "LE_MATCH_STRAW": ["0", "1"],
         "LE_SW_DAY": ["8", "9", "11"], "LE_ANIMAL_LAST": ["10", "13", "16"]}
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=40); ap.add_argument("--switch", default="432"); ap.add_argument("--games", default="16")
ap.add_argument("--out", default="tmp/e058/sweep.log"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--vs", default="tmp/e058/agents/ahmedberatozer_kaggriculture-v41-review-candidate/main.py")
a = ap.parse_args(); rng = random.Random(a.seed)
def run(cfg):
    env = dict(os.environ, LE_SWITCH=a.switch, **cfg)
    out = subprocess.run([sys.executable, "tests/kag_eval.py", "agents/live_e/main.py", "--vs", a.vs, "--games", a.games, "--seed0", "5000"], env=env, capture_output=True, text=True).stdout
    m = re.search(r"(\d+)W(\d+)L avg ([-+][\d,]+)", out)
    return (int(m.group(1)), float(m.group(3).replace(",", ""))) if m else (None, None)
w, m = run({}); print(f"baseline {w}W avg {m:+.0f}", flush=True)
with open(a.out, "a") as f:
    f.write(json.dumps({"cfg": {}, "w": w, "avg": m, "switch": a.switch}) + "\n")
    for i in range(a.n):
        cfg = {k: rng.choice(v) for k, v in SPACE.items() if rng.random() < 0.5}
        w, m = run(cfg); f.write(json.dumps({"cfg": cfg, "w": w, "avg": m, "switch": a.switch}) + "\n"); f.flush()
        print(f"{i:3d} {w}W avg {m:+.0f} {cfg}", flush=True)
