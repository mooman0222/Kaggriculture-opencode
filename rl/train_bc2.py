"""Behaviour cloning (v2: destination + op). 途中終了しても `--resume` で <out>.state から続きを学習できる。
使い方: .venv/bin/python rl/train_bc2.py --data 'tmp/rl/*/*.npz' --out tmp/rl/bc.pt --epochs 3"""
import argparse, glob, os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from model2 import Policy2 as Policy, bc_loss2 as bc_loss, bc_loss_mkt
import model2
KEYS = ("tiles", "units", "items", "glob", "mkt", "dest", "dop", "dqty", "dmask")


CAST = {"tiles": np.uint8, "units": np.uint8, "dmask": np.bool_, "dest": np.int8, "dop": np.int8, "dqty": np.int8, "mkt": np.int8}


def load_shards(files):
    """Concatenate shards with compact dtypes (~8.7 KB/step): tiles/units fit in uint8, labels in int8."""
    data = {k: [] for k in KEYS}
    for f in files:
        d = np.load(f)
        for k in KEYS: data[k].append(d[k].astype(CAST[k]) if k in CAST else d[k])
    return {k: np.concatenate(v) for k, v in data.items()}


def make_prev(data):
    """prev decision per unit = labels of the previous step within the same day (100/44 = none)."""
    dest, dop, glob = data["dest"], data["dop"], data["glob"]
    prev = np.full(dest.shape + (2,), 0, dtype=np.int16); prev[..., 0] = 100; prev[..., 1] = 44
    hour = np.rint(glob[:, 1] * 23).astype(int)
    ok = np.zeros(len(dest), dtype=bool); ok[1:] = hour[1:] > 0  # same game & same day as the previous row (shards are concatenated in step order)
    prev[1:][ok[1:], :, 0] = np.where(dest[:-1][ok[1:]] >= 0, dest[:-1][ok[1:]], 100); prev[1:][ok[1:], :, 1] = np.where(dop[:-1][ok[1:]] >= 0, dop[:-1][ok[1:]], 44)
    return prev


def to_t(data, idx, dev):
    d = {k: torch.from_numpy(data[k][idx].astype(np.int16) if k in ("tiles", "units") else data[k][idx]).to(dev) for k in KEYS}; d["prev"] = torch.from_numpy(data["prev"][idx]).to(dev); return d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True, help="glob, ** 可 (例 'tmp/rl/**/*.npz')"); ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--d", type=int, default=128); ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--val", type=float, default=0.1); ap.add_argument("--max-games", type=int, default=100000); ap.add_argument("--init", default=None)
    ap.add_argument("--resume", action="store_true", help="<out>.state から再開 (model/opt/sched/epoch/batch)"); ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--market-only", action="store_true", help="市場ヘッドのみ学習 (農場の dest/op/qty 損失を除外、E058 テープ上乗せ用)")
    a = ap.parse_args(); files = sorted(glob.glob(a.data, recursive=True)) if not a.data.endswith(".txt") else [l.strip() for l in open(a.data) if l.strip()]; random.Random(0).shuffle(files); files = files[: a.max_games]
    loss_fn = bc_loss_mkt if a.market_only else bc_loss
    nv = max(1, int(len(files) * a.val)); val_files, tr_files = files[:nv], files[nv:]
    t0 = time.time(); tr = load_shards(tr_files); va = load_shards(val_files); tr["prev"] = make_prev(tr); va["prev"] = make_prev(va)
    print(f"games train {len(tr_files)} val {len(val_files)}; steps train {len(tr['dest'])} val {len(va['dest'])} [{time.time()-t0:.0f}s]", flush=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    ops = tr["dop"][tr["dop"] >= 0]; freq = np.bincount(ops, minlength=44).astype(np.float64) + 10
    w = (freq.mean() / freq) ** 0.5; model2.OP_WEIGHT = torch.tensor(w / w.mean(), dtype=torch.float32); print("op class weights (min/max)", w.min().round(2), w.max().round(2))
    model = Policy(d=a.d, layers=a.layers).to(dev); print("params", sum(p.numel() for p in model.parameters()))
    if a.init: model.load_state_dict(torch.load(a.init, map_location=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    n = len(tr["dest"]); steps_per_epoch = n // a.bs; sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps_per_epoch + 1)
    state_path = a.out + ".state"; start_ep = 0; start_b = 0
    if a.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location=dev); model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); start_ep, start_b = st["epoch"], st["batch"]
        print(f"resumed from epoch {start_ep} batch {start_b}", flush=True)
    for ep in range(start_ep, a.epochs):
        model.train(); perm = np.random.RandomState(1000 + ep).permutation(n); t0 = time.time(); tot = 0; k = 0
        for b in range(start_b if ep == start_ep else 0, steps_per_epoch):
            if b % a.save_every == 0 and b > 0:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep, "batch": b}, state_path)
            idx = perm[b * a.bs:(b + 1) * a.bs]; batch = to_t(tr, idx, dev)
            out = model(batch["tiles"], batch["units"], batch["items"], batch["glob"], dest=batch["dest"].long().clamp(min=0), prev=batch["prev"]); loss, acc = loss_fn(out, batch)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tot += loss.item(); k += 1
            if b % 200 == 0: print(f"  ep {ep} b {b}/{steps_per_epoch} loss {loss.item():.3f} acc {acc} [{time.time()-t0:.0f}s]", flush=True)
        model.eval(); accs = []; vl = 0; m = 0
        with torch.no_grad():
            for b in range(0, len(va["dest"]), 1024):
                idx = np.arange(b, min(b + 1024, len(va["dest"]))); batch = to_t(va, idx, dev)
                out = model(batch["tiles"], batch["units"], batch["items"], batch["glob"], dest=batch["dest"].long().clamp(min=0), prev=batch["prev"]); loss, acc = loss_fn(out, batch); vl += loss.item() * len(idx); m += len(idx); accs.append(acc)
        vacc = {k: round(sum(x[k] for x in accs) / len(accs), 3) for k in accs[0]}
        print(f"epoch {ep} train loss {tot/max(1,k):.3f} val loss {vl/m:.3f} val acc {vacc} [{time.time()-t0:.0f}s]", flush=True)
        torch.save(model.state_dict(), a.out); torch.save(model.state_dict(), a.out.replace('.pt', f'_ep{ep}.pt'))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep + 1, "batch": 0}, state_path)


if __name__ == "__main__":
    main()
