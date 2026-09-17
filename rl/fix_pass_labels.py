"""既存 shard の PASS ラベル修復 (labels.py の日跨ぎ切断バグ、2026-09-17)。

旧 labels.py は「その日の残り時間に作業が無い」ユニットを一律 (現在タイル, PASS) にしていたため、
夕方に目的地へ歩いている途中のユニットまで PASS 教師になっていた (Majkel 実測 62/局 に対しラベル 461/局)。
誤ラベルの集合は shard だけで一意に決まる: dop == PASS かつ生の op が移動。そこを -1 (損失から除外) に戻す。

使い方: .venv/bin/python rl/fix_pass_labels.py 'tmp/rl/majkel_all/*.npz'
"""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import OP_INDEX

MOVE_IDS = np.array([OP_INDEX[m] for m in ("NORTH", "SOUTH", "EAST", "WEST")])


def fix(path):
    d = dict(np.load(path))
    bad = (d["dop"] == OP_INDEX["PASS"]) & np.isin(d["op"], MOVE_IDS)
    if not bad.any(): return 0
    d["dest"][bad] = -1; d["dop"][bad] = -1; d["dqty"][bad] = 0
    np.savez_compressed(path, **d)
    return int(bad.sum())


if __name__ == "__main__":
    files = sorted(glob.glob(sys.argv[1]))
    total = 0
    for n, f in enumerate(files, 1):
        total += fix(f)
        if n % 100 == 0: print(f"{n}/{len(files)}", flush=True)
    print(f"{len(files)} shards, {total} labels masked ({total / max(len(files), 1):.0f}/game)")
