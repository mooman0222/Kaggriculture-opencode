"""Replay JSON -> training shards (features, legality masks, targets) for one team's seat.
使い方: .venv/bin/python rl/extract.py --glob 'tmp/e058/mmpq/episode-*.json' --team 'M & M & P & Q' --out tmp/rl/mmpq
出力: <out>/<episode_id>.npz (tiles int16 [T,2,10,10,F], units int16, items f32, glob f32, mask bool [T,U,OPS], op/qty int8 [T,U], mkt int8 [T,N_MKT], reward f32)"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from features import encode, legal_ops, MAX_UNITS
from actions import encode_action
from labels import destination_labels
from features import legal_ops_at


def extract_replay(r, seat):
    S = r["steps"]; T = min(719, len(S) - 1)
    out = {k: [] for k in ("tiles", "units", "items", "glob", "mask", "op", "qty", "mkt", "dmask")}
    dest, dop, dqty = destination_labels(S, seat, T)
    for t in range(T):
        obs = S[t][seat]["observation"]; obs["step"] = t
        act = S[t + 1][seat]["action"] or {}
        f = encode(obs, seat); m = legal_ops(obs, seat)
        n = min(MAX_UNITS, 1 + len(obs["farms"][seat]["hands"])); op, qty, mkt = encode_action(act, n)
        for k in ("tiles", "units", "items", "glob"): out[k].append(f[k])
        out["mask"].append(m); out["op"].append(op); out["qty"].append(qty); out["mkt"].append(mkt)
        dm = np.zeros_like(m)
        for i in range(n):
            if dest[t, i] >= 0: dm[i] = legal_ops_at(obs, seat, i, int(dest[t, i]) % 10, int(dest[t, i]) // 10)
        out["dmask"].append(dm)
    res = {k: np.stack(v) for k, v in out.items()}
    res["dest"] = dest; res["dop"] = dop; res["dqty"] = dqty
    rw = r.get("rewards") or [0, 0]
    res["reward"] = np.array([float(rw[seat] or 0), float(rw[1 - seat] or 0)], dtype=np.float32)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--glob", required=True); ap.add_argument("--team", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--delete", action="store_true", help="処理後に JSON を削除")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); n = 0
    for f in sorted(glob.glob(a.glob)):
        try: r = json.load(open(f))
        except Exception: continue
        names = r["info"]["TeamNames"]
        if a.team not in names or len(r["steps"]) < 720: continue
        eid = r["info"]["EpisodeId"]; dst = os.path.join(a.out, f"{eid}.npz")
        if not os.path.exists(dst):
            res = extract_replay(r, names.index(a.team)); np.savez_compressed(dst, **res); n += 1
        if a.delete: os.remove(f)
    print(f"{a.team}: {n} new shards in {a.out} (total {len(glob.glob(a.out + '/*.npz'))})")


if __name__ == "__main__":
    main()
