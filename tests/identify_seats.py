"""各リプレイの各席について、候補エージェントが記録どおりの手を何手目まで打てるかを測る (相手席は記録のテープ)。719 = 全手一致。

  SKIP_TEAM=MMN0222 .venv/bin/python tests/identify_seats.py 'tmp/band0925/episode-*.json' tmp/ident.json agents/pub_*/main.py agents/e07*/main.py
  我々の試合に限らず使える (match_versions.py は我々の席を固定する版)。agents/ 配下のみ実行する。
"""
import json, glob, sys, os, importlib.util, hashlib
from multiprocessing import Pool
import kagsim
PASS = {"farmer": ["PASS"], "hands": [], "market": []}
CANDS = []
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load(p, tag):
    assert os.path.abspath(p).startswith(os.path.join(ROOT, "agents") + os.sep), "only agents/ are executed"
    d = os.path.dirname(os.path.abspath(p)); sys.path.insert(0, d)
    s = importlib.util.spec_from_file_location("id_" + tag, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def norm(a):
    return json.dumps(a or PASS, sort_keys=True)
def one(job):
    f, seat = job
    r = json.load(open(f)); st = r["steps"]; names = r["info"]["TeamNames"]
    if len(st) < 720: return None
    shops = st[-1][0]["observation"]["town"]["unlocked_shops"]
    rec = [st[t + 1][seat].get("action") or PASS for t in range(719)]
    tape = [st[t + 1][1 - seat].get("action") or PASS for t in range(719)]
    recn = [norm(a) for a in rec]
    out = {}
    for ci, p in enumerate(CANDS):
        try:
            m = load(p, hashlib.md5((p + f + str(seat)).encode()).hexdigest()[:10])
            ag = getattr(m, "agent")
            g = kagsim.Game(r["info"]["seed"], 720, shops); first = None
            while not g.done:
                t = g.step_count
                a = ag(g.observe(seat))
                if norm(a) != recn[t]: first = t; break
                g.step(*((a, tape[t]) if seat == 0 else (tape[t], a)))
            out[os.path.basename(os.path.dirname(p))] = 719 if first is None else first
        except Exception as e:
            out[os.path.basename(os.path.dirname(p))] = -1
    return dict(eid=r["info"]["EpisodeId"], seat=seat, team=names[seat], res=out)
def init(c):
    global CANDS; CANDS = c
if __name__ == "__main__":
    files = sorted(glob.glob(sys.argv[1])); outp = sys.argv[2]; cands = sys.argv[3:]
    team_skip = os.environ.get("SKIP_TEAM")
    jobs = []
    for f in files:
        names = json.load(open(f))["info"]["TeamNames"]
        for s in (0, 1):
            if team_skip and names[s] == team_skip: continue
            jobs.append((f, s))
    with Pool(4, initializer=init, initargs=(cands,)) as pool:
        rows = [x for x in pool.imap_unordered(one, jobs) if x]
    json.dump(rows, open(outp, "w"))
    import collections
    best = collections.Counter(); full = collections.Counter()
    for r in rows:
        k = max(r["res"], key=r["res"].get); v = r["res"][k]
        best[(k, v == 719)] += 1
        for kk, vv in r["res"].items():
            if vv == 719: full[kk] += 1
    print("seats", len(rows)); print("full 719 matches:", dict(full))
    print("best prefix:", sorted(best.items(), key=lambda kv: -kv[1])[:20])
