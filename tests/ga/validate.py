"""進化テープを E055 ライブ層 (sr0909_live) に載せて L1 で検証。使い方: .venv/bin/python tests/ga/validate.py tmp/ga/best.json [--games 24]"""
import argparse, shutil, subprocess, sys
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("tapes"); ap.add_argument("--games", type=int, default=24); ap.add_argument("--pool", action="store_true")
a = ap.parse_args()
d = Path("agents/ga_live"); d.mkdir(exist_ok=True)
for f in ("main.py", "router.py", "LICENSE.txt"): shutil.copy(f"agents/sr0909_live/{f}", d / f)
shutil.copy(a.tapes, d / "actions.json")
py = sys.executable
for opp in ("agents/sr0909_base/main.py", "agents/sr0909_live/main.py"):
    out = subprocess.run([py, "tests/kag_eval.py", str(d / "main.py"), "--vs", opp, "--games", str(a.games), "--seed0", "2000"], capture_output=True, text=True).stdout.strip().split("\n")[-1]
    print(out)
if a.pool:
    out = subprocess.run([py, "tests/kag_eval.py", str(d / "main.py"), "--replays", "tmp/top0911/mine_e054/0_MMN0222/episode-*.json", "--team", "MMN0222", "--base", "agents/sr0909_live/main.py"], capture_output=True, text=True).stdout.strip().split("\n")
    print("\n".join(out[-9:]))
