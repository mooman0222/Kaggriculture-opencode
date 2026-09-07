"""提出のレーティング推移と相手レート帯別勝率 (非公式 ListEpisodes: initialScore/updatedScore 付き)

使い方: .venv/bin/python tests/rating_traj.py 56034377 [56046873 ...] [--out DIR]
"""
import argparse, json, urllib.request
from collections import defaultdict
from pathlib import Path

URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"


def pull(sid):
    req = urllib.request.Request(URL, data=json.dumps({"submissionId": sid}).encode(), headers={"content-type": "application/json"})
    eps = json.load(urllib.request.urlopen(req, timeout=60))["episodes"]
    rows = []
    for e in eps:
        if e.get("type") != "EPISODE_TYPE_PUBLIC" or e.get("state") != "COMPLETED":
            continue
        own = next(a for a in e["agents"] if a["submissionId"] == sid)
        opp = next(a for a in e["agents"] if a is not own)
        if own.get("reward") is None or opp.get("reward") is None:
            continue
        rows.append(dict(t=e["endTime"], id=e["id"], own=own["reward"], opp=opp["reward"], m=own["reward"] - opp["reward"],
                         r0=own.get("initialScore"), r1=own.get("updatedScore"), oppr=opp.get("initialScore"),
                         opp_sid=opp["submissionId"], opp_team=opp.get("teamId")))
    rows.sort(key=lambda r: r["t"])
    return rows


def report(sid, rows):
    print(f"\n== {sid} n={len(rows)} peak={max((r['r1'] or 0) for r in rows):.0f} now={rows[-1]['r1']:.0f}")
    print("block  start_time        W-L   own_r0  opp_r(mean)  d_rating")
    for i in range(0, len(rows), 20):
        b = rows[i:i + 20]
        w = sum(r["m"] > 0 for r in b); l = sum(r["m"] < 0 for r in b)
        oppr = [r["oppr"] for r in b if r["oppr"]]
        print(f"{i:4d}  {b[0]['t'][:16]}  {w:2d}-{l:2d}  {b[0]['r0'] or 0:6.0f}  {sum(oppr) / len(oppr):8.0f}  {(b[-1]['r1'] or 0) - (b[0]['r0'] or 0):+7.1f}")
    band = defaultdict(list)
    for r in rows:
        if r["oppr"]:
            band[int(r["oppr"] // 200) * 200].append(r)
    print("opp rating band: W-L wr mean_margin")
    for k in sorted(band):
        b = band[k]; w = sum(r["m"] > 0 for r in b); l = sum(r["m"] < 0 for r in b)
        print(f"  {k}-{k + 199}: {w:3d}-{l:3d}  wr={w / len(b):.2f}  mm={sum(r['m'] for r in b) / len(b):+7.0f}")
    cohort = defaultdict(list)  # 相手の提出 ID 帯 (新しい提出ほど強い傾向を見る)
    for r in rows:
        cohort[r["opp_sid"] // 20000 * 20000].append(r)
    print("opp submission-id cohort (20k刻み): W-L wr mean_margin")
    for k in sorted(cohort):
        b = cohort[k]; w = sum(r["m"] > 0 for r in b); l = sum(r["m"] < 0 for r in b)
        print(f"  {k}: {w:3d}-{l:3d}  wr={w / len(b):.2f}  mm={sum(r['m'] for r in b) / len(b):+7.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subs", nargs="+", type=int)
    ap.add_argument("--out", help="rows を <out>/<sid>_traj.json に保存")
    a = ap.parse_args()
    for sid in a.subs:
        rows = pull(sid)
        if not rows:
            print(f"== {sid}: no completed public episodes"); continue
        report(sid, rows)
        if a.out:
            Path(a.out).mkdir(parents=True, exist_ok=True)
            json.dump(rows, open(Path(a.out) / f"{sid}_traj.json", "w"), indent=1)


if __name__ == "__main__":
    main()
