"""Behaviour cloning on extracted shards. 使い方: .venv/bin/python rl/train_bc.py --data 'tmp/rl/*/*.npz' --out tmp/rl/bc.pt --epochs 3"""
import argparse, glob, os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from model import Policy, bc_loss
KEYS = ("tiles", "units", "items", "glob", "mask", "op", "qty", "mkt")


def load_shards(files):
    data = {k: [] for k in KEYS}
    for f in files:
        d = np.load(f)
        for k in KEYS: data[k].append(d[k])
    return {k: np.concatenate(v) for k, v in data.items()}


def to_t(data, idx, dev):
    return {k: torch.from_numpy(data[k][idx]).to(dev) for k in KEYS}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--d", type=int, default=128); ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--val", type=float, default=0.1); ap.add_argument("--max-games", type=int, default=100000); ap.add_argument("--init", default=None)
    a = ap.parse_args(); files = sorted(glob.glob(a.data)); random.Random(0).shuffle(files); files = files[: a.max_games]
    nv = max(1, int(len(files) * a.val)); val_files, tr_files = files[:nv], files[nv:]
    t0 = time.time(); tr = load_shards(tr_files); va = load_shards(val_files)
    print(f"games train {len(tr_files)} val {len(val_files)}; steps train {len(tr['op'])} val {len(va['op'])} [{time.time()-t0:.0f}s]", flush=True)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    import model as M
    ops = tr["op"][tr["op"] >= 0]; freq = np.bincount(ops, minlength=M.N_OPS).astype(np.float64) + 10
    w = (freq.mean() / freq) ** 0.5; M.OP_WEIGHT = torch.tensor(w / w.mean(), dtype=torch.float32); print("op class weights (min/max)", w.min().round(2), w.max().round(2))
    model = Policy(d=a.d, layers=a.layers).to(dev); print("params", sum(p.numel() for p in model.parameters()))
    if a.init: model.load_state_dict(torch.load(a.init, map_location=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    n = len(tr["op"]); steps_per_epoch = n // a.bs; sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps_per_epoch + 1)
    for ep in range(a.epochs):
        model.train(); perm = np.random.permutation(n); t0 = time.time(); tot = 0; k = 0
        for b in range(steps_per_epoch):
            idx = perm[b * a.bs:(b + 1) * a.bs]; batch = to_t(tr, idx, dev)
            out = model(batch["tiles"], batch["units"], batch["items"], batch["glob"]); loss, acc = bc_loss(out, batch)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tot += loss.item(); k += 1
            if b % 200 == 0: print(f"  ep {ep} b {b}/{steps_per_epoch} loss {loss.item():.3f} acc {acc} [{time.time()-t0:.0f}s]", flush=True)
        model.eval(); accs = []; vl = 0; m = 0
        with torch.no_grad():
            for b in range(0, len(va["op"]), 1024):
                idx = np.arange(b, min(b + 1024, len(va["op"]))); batch = to_t(va, idx, dev)
                out = model(batch["tiles"], batch["units"], batch["items"], batch["glob"]); loss, acc = bc_loss(out, batch); vl += loss.item() * len(idx); m += len(idx); accs.append(acc)
        vacc = {k: round(sum(x[k] for x in accs) / len(accs), 3) for k in accs[0]}
        print(f"epoch {ep} train loss {tot/max(1,k):.3f} val loss {vl/m:.3f} val acc {vacc} [{time.time()-t0:.0f}s]", flush=True)
        torch.save(model.state_dict(), a.out)


if __name__ == "__main__":
    main()
