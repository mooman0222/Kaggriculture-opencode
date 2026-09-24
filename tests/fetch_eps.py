"""エピソード ID 群のリプレイを並列 (既定 3) で取得する。429 は待って再試行。

使い方: .venv/bin/python tests/fetch_eps.py --out DIR ID [ID ...]   (または --traj tmp/traj/<sid>_traj.json)
"""
import argparse, json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

KAGGLE = str(Path(sys.executable).parent / "kaggle")
PACE = 2.0


def get(eid, out):
    f = out / f"episode-{eid}-replay.json"
    for k in range(8):
        if f.exists() and f.stat().st_size > 1000:
            return True
        r = subprocess.run([KAGGLE, "competitions", "replay", str(eid), "-p", str(out), "-q"], capture_output=True, text=True)
        if f.exists() and f.stat().st_size > 1000:
            time.sleep(PACE)
            return True
        time.sleep(60 * (k + 1) if "429" in (r.stderr + r.stdout) else 5)
    print("FAIL", eid, (r.stderr or r.stdout)[-200:], flush=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", type=int); ap.add_argument("--traj", nargs="*", default=[])
    ap.add_argument("--out", required=True); ap.add_argument("--jobs", type=int, default=1); ap.add_argument("--pace", type=float, default=2.0)
    a = ap.parse_args()
    global PACE; PACE = a.pace
    ids = list(a.ids)
    for t in a.traj:
        ids += [r["id"] for r in json.load(open(t))]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ids = sorted(set(ids))
    with ThreadPoolExecutor(a.jobs) as ex:
        ok = sum(ex.map(lambda e: get(e, out), ids))
    print(f"{ok}/{len(ids)} in {out}")


if __name__ == "__main__":
    main()
