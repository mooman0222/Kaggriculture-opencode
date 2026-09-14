"""提出 ID の直近エピソードのリプレイを取得する。使い方: .venv/bin/python tests/fetch_sub.py <submissionId> <n> <outdir>"""
import sys, subprocess
from pathlib import Path
from kagglesdk import KaggleClient
from kagglesdk.competitions.types.competition_api_service import ApiListSubmissionEpisodesRequest
sid, per, out = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
kaggle = str(Path(sys.executable).parent / "kaggle")
with KaggleClient() as client:
    api = client.competitions.competition_api_client
    er = ApiListSubmissionEpisodesRequest(); er.submission_id = sid
    eps = sorted(api.list_submission_episodes(er).episodes, key=lambda e: -e.id)
    print("episodes total", len(eps))
    for e in eps[:per]:
        f = out / f"episode-{e.id}-replay.json"
        if not f.exists():
            subprocess.run([kaggle, "competitions", "replay", str(e.id), "-p", str(out), "-q"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("downloaded", len(list(out.glob("episode-*.json"))))
