"""LB 上位チームのベスト提出から直近エピソードのリプレイを取得する。

使い方: .venv/bin/python tests/fetch_top.py --lb leaderboard.csv --top 15 --per 8 --out DIR
出力: DIR/<teamId>_<teamName>/episode-<id>-replay.json と DIR/teams.json
"""
import argparse, csv, json, subprocess, sys, re
from pathlib import Path
from kagglesdk import KaggleClient
from kagglesdk.competitions.types.competition_api_service import (
    ApiListTeamPublicSubmissionsRequest, ApiListSubmissionEpisodesRequest)

ap = argparse.ArgumentParser()
ap.add_argument("--lb", required=True); ap.add_argument("--top", type=int, default=15)
ap.add_argument("--per", type=int, default=8); ap.add_argument("--out", required=True)
ap.add_argument("--skip", type=int, default=0)
a = ap.parse_args()
rows = [r for r in csv.reader(open(a.lb)) if len(r) > 4 and re.match(r"^\d+(\.\d+)?$", r[4])]
rows.sort(key=lambda r: -float(r[4]))
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
teams = []
kaggle = str(Path(sys.executable).parent / "kaggle")
with KaggleClient() as client:
    api = client.competitions.competition_api_client
    for r in rows[a.skip:a.skip + a.top]:
        team_id, name, score = int(r[1]), r[2], float(r[4])
        req = ApiListTeamPublicSubmissionsRequest(); req.team_id = team_id
        subs = api.list_team_public_submissions(req).submissions
        subs = sorted(subs, key=lambda s: -float(s.public_score or 0))
        if not subs:
            print("no subs", name); continue
        best = subs[0]
        er = ApiListSubmissionEpisodesRequest(); er.submission_id = best.id
        eps = api.list_submission_episodes(er).episodes
        eps = sorted(eps, key=lambda e: -e.id)[:a.per]
        d = out / f"{team_id}_{re.sub(r'[^A-Za-z0-9_-]', '_', name)}"; d.mkdir(exist_ok=True)
        for e in eps:
            f = d / f"episode-{e.id}-replay.json"
            if not f.exists():
                subprocess.run([kaggle, "competitions", "replay", str(e.id), "-p", str(d), "-q"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        teams.append({"team_id": team_id, "name": name, "score": score, "submission": best.id,
                      "n_episodes": len(list(d.glob("episode-*.json"))), "dir": str(d)})
        print(f"{name:<20} {score:7.1f} sub={best.id} episodes={teams[-1]['n_episodes']}", flush=True)
json.dump(teams, open(out / "teams.json", "w"), indent=1, ensure_ascii=False)
