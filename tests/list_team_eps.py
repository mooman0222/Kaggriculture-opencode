"""LB の順位帯のチームについて、最高スコアの提出の直近エピソード ID を列挙する (リプレイは取らない)。

  .venv/bin/python tests/list_team_eps.py --lb LB.csv --ranks 30-150 --per 4 --out tmp/band_eps.json
"""
import argparse, csv, json, re, time
from kagglesdk import KaggleClient
from kagglesdk.competitions.types.competition_api_service import (
    ApiListTeamPublicSubmissionsRequest, ApiListSubmissionEpisodesRequest)

ap = argparse.ArgumentParser()
ap.add_argument("--lb", required=True); ap.add_argument("--ranks", default="30-150"); ap.add_argument("--per", type=int, default=4)
ap.add_argument("--out", required=True)
a = ap.parse_args()
lo, hi = map(int, a.ranks.split("-"))
rows = [r for r in csv.DictReader(open(a.lb, encoding="utf-8-sig")) if lo <= int(r["Rank"]) <= hi]
out = []
with KaggleClient() as client:
    api = client.competitions.competition_api_client
    def call(fn, req):
        for k in range(8):
            try:
                return fn(req)
            except Exception as e:
                if "429" not in str(e): raise
                time.sleep(30 * (k + 1))
        raise RuntimeError("429 persists")
    for r in rows:
        try:
            req = ApiListTeamPublicSubmissionsRequest(); req.team_id = int(r["TeamId"])
            subs = sorted(call(api.list_team_public_submissions, req).submissions, key=lambda s: -float(s.public_score or 0))
            time.sleep(1.5)
            if not subs: continue
            er = ApiListSubmissionEpisodesRequest(); er.submission_id = subs[0].id
            eps = sorted(call(api.list_submission_episodes, er).episodes, key=lambda e: -e.id)[:a.per]
            time.sleep(1.5)
            out.append(dict(rank=int(r["Rank"]), team=r["TeamName"], team_id=int(r["TeamId"]), score=float(r["Score"]),
                            sub=subs[0].id, eps=[e.id for e in eps]))
            print(r["Rank"], r["TeamName"][:20], subs[0].id, len(eps), flush=True)
        except Exception as e:
            print("ERR", r["TeamName"], e, flush=True)
json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=0)
