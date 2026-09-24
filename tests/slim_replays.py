"""リプレイを席差し替え・照合に要る部分だけに縮める (行動・info・rewards・最終ショップ列、観測は step 0 と最終のみ)。

  .venv/bin/python tests/slim_replays.py --zip tmp/kds/kaggriculture-episodes-2026-09-22.zip --out tmp/slim0922
  .venv/bin/python tests/slim_replays.py --glob 'tmp/own0924/episode-*.json' --out tmp/slim_own
"""
import argparse, glob, json, os, zipfile
from multiprocessing import Pool

ZIP = None


def init(z):
    global ZIP
    ZIP = z


def work(args):
    src, out = args
    r = json.loads(zipfile.ZipFile(ZIP).read(src)) if ZIP else json.load(open(src))
    eid = r["info"]["EpisodeId"]; dst = os.path.join(out, f"episode-{eid}-replay.json")
    if os.path.exists(dst): return 0
    st = r["steps"]; last = len(st) - 1
    slim = [[{"action": s.get("action"), **({"observation": s["observation"]} if t in (0, last) else {})} for s in row]
            for t, row in enumerate(st)]
    json.dump(dict(info=r["info"], rewards=r["rewards"], steps=slim), open(dst, "w"))
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip"); ap.add_argument("--glob"); ap.add_argument("--out", required=True); ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    srcs = [n for n in zipfile.ZipFile(a.zip).namelist() if n.endswith(".json")] if a.zip else sorted(glob.glob(a.glob))
    with Pool(a.jobs, initializer=init, initargs=(a.zip,)) as p:
        n = sum(p.imap_unordered(work, [(s, a.out) for s in srcs], chunksize=4))
    print(n, "written to", a.out)


if __name__ == "__main__":
    main()
