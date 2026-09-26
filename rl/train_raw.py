"""Behaviour cloning on the raw action representation (rl/raw.py). Shards from rl/gen_selfplay.py --raw.
usage: .venv/bin/python rl/train_raw.py --data 'tmp/rl/v41raw/*.npz' --out tmp/rl/raw1.pt --epochs 4 --bs 128 --lr 5e-4"""
import argparse, glob, math, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from raw import Policy4, raw_loss

KEYS = ("tiles", "units", "items", "glob", "mask", "uop", "uqty", "mk", "mq")
CAST = {"tiles": np.uint8, "units": np.uint8}


def load(files):
    """Two-pass preallocated load (peak ~1x, same idea as train_bc2.load_shards)."""
    with np.load(files[0]) as d: meta = {k: (d[k].shape[1:], np.dtype(CAST.get(k, d[k].dtype))) for k in KEYS}
    ns = []
    for f in files:
        with np.load(f) as d: ns.append(len(d["uop"]))
    data = {k: np.empty((sum(ns),) + s, dtype=t) for k, (s, t) in meta.items()}; off = 0
    for f, n in zip(files, ns):
        with np.load(f) as d:
            for k in KEYS: data[k][off:off + n] = d[k]
        off += n
    return data


def batch(data, idx, dev):
    return {k: torch.from_numpy(data[k][idx].astype(np.int16) if k in CAST else data[k][idx]).to(dev) for k in KEYS}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True, nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-games", type=int, default=100000); ap.add_argument("--val", type=float, default=0.1); ap.add_argument("--init")
    ap.add_argument("--patience", type=int, default=0,
                    help="early stopping: break when val loss fails to improve by --min-delta for this many consecutive epochs (0 = off, legacy fixed-epoch behavior)")
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--market-only", action="store_true",
                    help="freeze everything except the market heads (mkind/mqty): fine-tune market timing on off-trajectory states without touching farm behavior")
    a = ap.parse_args()
    files = sorted(f for pattern in a.data for f in glob.glob(pattern)); random.Random(0).shuffle(files); files = files[:a.max_games]
    nv = max(1, int(len(files) * a.val)); t0 = time.time()
    tr, va = load(files[nv:]), load(files[:nv])
    print(f"games train {len(files) - nv} val {nv}; steps {len(tr['uop'])} / {len(va['uop'])} [{time.time() - t0:.0f}s]", flush=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = Policy4().to(dev)
    if a.init: model.load_state_dict(torch.load(a.init, map_location=dev)); print("init from", a.init, flush=True)
    if a.market_only:
        for n, p in model.named_parameters():
            if "mkind" not in n and "mqty" not in n: p.requires_grad = False
        print("market-only: trainable", sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
    print("params", sum(p.numel() for p in model.parameters()), "device", dev, flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01)
    spe = math.ceil(len(tr["uop"]) / a.bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * spe)
    best = float("inf"); stale = 0
    for ep in range(a.epochs):
        model.train(); perm = np.random.RandomState(ep).permutation(len(tr["uop"])); tot = 0.0; t1 = time.time()
        for bi, s in enumerate(range(0, len(perm), a.bs)):
            b = batch(tr, perm[s:s + a.bs], dev); loss, acc = raw_loss(model(b["tiles"], b["units"], b["items"], b["glob"]), b)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tot += loss.item()
            if bi % 500 == 0: print(f"  ep {ep} b {bi}/{spe} loss {loss.item():.3f} {acc}", flush=True)
        model.eval(); accs = []; vl = 0.0
        with torch.no_grad():
            for s in range(0, len(va["uop"]), 512):
                b = batch(va, np.arange(s, min(s + 512, len(va["uop"]))), dev); l, acc = raw_loss(model(b["tiles"], b["units"], b["items"], b["glob"]), b)
                vl += l.item(); accs.append(acc)
        mean = {k: round(float(np.mean([x[k] for x in accs])), 4) for k in accs[0]}
        vloss = vl / len(accs)
        tag = ""
        if vloss < best - a.min_delta:
            best = vloss; stale = 0
            torch.save(model.state_dict(), a.out.replace(".pt", "_best.pt")); tag = " [best]"
        elif a.patience:
            stale += 1; tag = f" [stale {stale}/{a.patience}]"
        print(f"epoch {ep} train loss {tot / spe:.3f} val loss {vloss:.3f} val acc {mean}{tag} [{time.time() - t1:.0f}s]", flush=True)
        torch.save(model.state_dict(), a.out); torch.save(model.state_dict(), a.out.replace(".pt", f"_ep{ep}.pt"))
        if a.patience and stale >= a.patience:
            print(f"early stopping at epoch {ep} (best val loss {best:.3f}, --patience {a.patience} --min-delta {a.min_delta})", flush=True)
            break


if __name__ == "__main__":
    main()
