"""オラクル (tests/overlay_headroom.py --family hold --feat) の日次特徴量 + hold ラベルから MLP を学習し、
numpy 推論層を土台エージェントの末尾に埋め込んだ main.py を書き出す。

  .venv/bin/python rl/hold/train_hold.py --data tmp/headroom_*.json --base agents/e087/main.py --dst agents/e088 [--thr 0.5 --epochs 300]
学習は局単位で 80/20 に分け、val の BCE で早期停止。閾値は閉ループ (swap_eval / kag_eval、HOLD_THR 環境変数) で選ぶ。
"""
import argparse, glob, json, os, random, shutil
import numpy as np, torch, torch.nn as nn

ITEMS = ("MILK", "WOOL", "STRAWBERRY")
KEYS = ["day", "money", "cows", "sheep", "geese"] + [f"{k}_{it}" for it in ITEMS for k in ("px", "pxmax", "pxmin", "px24", "inv", "dinv", "shed", "sold")] + [f"hy_{it}" for it in ITEMS]


def with_hy(rows):
    """hy_<item> = その品目を前日に hold したか (自分の直前の決定。推論時は層が自分で知っている)。"""
    prev = None
    for row in rows:
        for it in ITEMS:
            row[f"hy_{it}"] = int(bool(prev and prev["day"] == row["day"] - 1 and prev[f"hold_{it}"]))
        prev = row
    return rows


def load(files):
    games = []
    for f in files:
        for r in json.load(open(f)):
            if r.get("feat"):
                with_hy(r["feat"])
                X = np.array([[row[k] for k in KEYS] for row in r["feat"]], dtype=np.float64)
                Y = np.array([[row[f"hold_{it}"] for it in ITEMS] for row in r["feat"]], dtype=np.float64)
                games.append((X, Y, r["greedy"] - r["base"]))
    return games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True); ap.add_argument("--base", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--thr", type=float, default=0.5); ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--posw", action="store_true", help="正例を sqrt(neg/pos) 倍に重み付け")
    a = ap.parse_args()
    files = sorted({f for p in a.data for f in glob.glob(p)})
    games = load(files); random.Random(a.seed).shuffle(games)
    nval = max(1, len(games) // 5); val, tr = games[:nval], games[nval:]
    Xtr = np.concatenate([g[0] for g in tr]); Ytr = np.concatenate([g[1] for g in tr])
    Xva = np.concatenate([g[0] for g in val]); Yva = np.concatenate([g[1] for g in val])
    mean, std = Xtr.mean(0), Xtr.std(0) + 1e-6
    print(f"games {len(games)} (train {len(tr)} val {len(val)}), rows {len(Xtr)}/{len(Xva)}, positive rate {Ytr.mean(0).round(3)}, oracle gain mean {np.mean([g[2] for g in games]):+.0f}")
    torch.manual_seed(a.seed)
    net = nn.Sequential(nn.Linear(len(KEYS), a.hidden), nn.ReLU(), nn.Linear(a.hidden, a.hidden), nn.ReLU(), nn.Linear(a.hidden, len(ITEMS)))
    pos = torch.tensor(np.sqrt((1 - Ytr.mean(0)) / np.maximum(Ytr.mean(0), 1e-3)) if a.posw else np.ones(len(ITEMS)), dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-4)
    xt = torch.tensor((Xtr - mean) / std, dtype=torch.float32); yt = torch.tensor(Ytr, dtype=torch.float32)
    xv = torch.tensor((Xva - mean) / std, dtype=torch.float32); yv = torch.tensor(Yva, dtype=torch.float32)
    best, best_state, bad = 1e9, None, 0
    for ep in range(a.epochs):
        net.train(); perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 256):
            idx = perm[i:i + 256]; opt.zero_grad(); l = lossf(net(xt[idx]), yt[idx]); l.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            lv = lossf(net(xv), yv).item(); pv = torch.sigmoid(net(xv)).numpy()
        if lv < best - 1e-4: best, best_state, bad = lv, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else: bad += 1
        if ep % 20 == 0 or bad > 30:
            hit = ((pv > a.thr) == (Yva > 0.5)).mean(0); prec = [(Yva[pv[:, j] > a.thr, j].mean() if (pv[:, j] > a.thr).any() else float('nan')) for j in range(3)]
            rec = [((pv[:, j] > a.thr) & (Yva[:, j] > 0.5)).sum() / max(1, (Yva[:, j] > 0.5).sum()) for j in range(3)]
            print(f"ep {ep:3d} val bce {lv:.4f} acc {hit.round(3)} prec {np.round(prec, 2)} rec {np.round(rec, 2)}")
        if bad > 30: break
    net.load_state_dict(best_state)
    layers = []
    for mod in net:
        if isinstance(mod, nn.Linear):
            layers.append([mod.weight.detach().numpy().T.round(6).tolist(), mod.bias.detach().numpy().round(6).tolist()])
    model = dict(keys=KEYS, mean=mean.round(6).tolist(), std=std.round(6).tolist(), layers=layers, val_bce=best, games=len(games))
    os.makedirs(a.dst, exist_ok=True)
    tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hold_layer.py")).read()
    tail = tpl.replace("__HOLD_MODEL__", json.dumps(model)).replace("__HOLD_THR__", repr(a.thr))
    base = open(a.base).read().rstrip("\n")
    open(os.path.join(a.dst, "main.py"), "w").write(base + "\n\n" + tail)
    for f in ("LICENSE.txt", "NOTICE.txt"):
        src = os.path.join(os.path.dirname(a.base), f)
        if os.path.exists(src): shutil.copy(src, os.path.join(a.dst, f))
    json.dump(model, open(os.path.join(a.dst, "hold_model.json"), "w"))
    print(f"wrote {a.dst}/main.py (val bce {best:.4f}, thr {a.thr})")


if __name__ == "__main__":
    main()
