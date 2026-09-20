"""value head の事前学習 — 09-19 診断への最後の処方。

診断 (`rl/diag_adv.py`) の結論: PPO の value head は乱数初期化のまま定数に留まり (R²≈0)、
adv が生の outcome になることが崩壊の主因候補。`--vlr 1e-3` だけではスケールが 10 分の 1
のまま (R²_mc 0.05) だったので、PPO を始める前に value head を教師ありで十分に学習させる
(probe は 1500 step で R²_val 0.6)。

手順:
  1. bc5_ep3 の方策で self-play + tape の窓を `--windows` 個収集 (train.py と同じ収集・GAE)。
  2. 凍結トランクの特徴 H['g'] を窓ごとにキャッシュし、value head だけを `--steps` 回回帰。
  3. 学習後の state_dict を `--out` に保存 (他は不変)。train.py --init に渡す。
トランクは凍結するので方策行動は bc5_ep3 のまま。

使い方:
  PYTHONPATH=.venv/lib/python3.14/site-packages /usr/bin/python3 rl/vpretrain.py \
    --init tmp/rl/bc5_ep3.pt --out tmp/rl/init_v.pt --tapes tmp/rl/tapes_top.pkl \
    --windows 8 --steps 2000 --workers 4 --games 120 --dev cuda
"""
from __future__ import annotations
import argparse, copy, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sp"))
import numpy as np, torch
from model2 import Policy2
from features import MAX_UNITS
from vec_env import VecEnv
from policy_batch import act_batch

KEYS_OBS = ("tiles", "units", "items", "glob")


def gae(reward, value, done, gamma, lam, v_last):
    T, L = reward.shape
    adv = np.zeros_like(reward); last = np.zeros(L, dtype=np.float32)
    for t in reversed(range(T)):
        nv = v_last if t + 1 == T else value[t + 1]
        nd = 1.0 - done[t]
        delta = reward[t] + gamma * nv * nd - value[t]
        last = delta + gamma * lam * nd * last
        adv[t] = last
    return adv, adv + value


def build_store(T, L, A):
    s = {k: np.zeros((T, L, *A[k].shape[1:]), dtype=A[k].dtype) for k in KEYS_OBS}
    s.update(prev=np.zeros((T, L, MAX_UNITS, 2), dtype=np.int16), dmask=np.zeros((T, L, MAX_UNITS, 100), dtype=bool),
             at_dest=np.zeros((T, L, MAX_UNITS), dtype=bool), present=np.zeros((T, L, MAX_UNITS), dtype=bool), decide=np.zeros((T, L, MAX_UNITS), dtype=bool),
             dest=np.zeros((T, L, MAX_UNITS), dtype=np.int16), op=np.zeros((T, L, MAX_UNITS), dtype=np.int16), qty=np.zeros((T, L, MAX_UNITS), dtype=np.int16),
             mkt=np.zeros((T, L, 21), dtype=np.int16), lp_dec=np.zeros((T, L, 69), dtype=np.float32),
             value=np.zeros((T, L), dtype=np.float32), reward=np.zeros((T, L), dtype=np.float32), done=np.zeros((T, L), dtype=bool))
    return s


def collect_window(a, model, dev, env, A, l_slots, prev):
    T, L = a.T, len(l_slots)
    store = build_store(T, L, A)
    outcome = np.full((T, L), np.nan)
    slot_pos = {int(s): j for j, s in enumerate(l_slots)}
    for t in range(T):
        if int(A["step"][0]) % 24 == 0: prev[:] = 0; prev[..., 0] = 100; prev[..., 1] = 44
        for k in KEYS_OBS: store[k][t] = A[k][l_slots]
        store["prev"][t] = prev[l_slots]
        out = act_batch(model, dev, A, l_slots, prev, temp=a.temp)
        for k in ("dmask", "at_dest", "present", "decide", "dest", "op", "qty", "mkt", "value"): store[k][t] = out[k]
        prev[l_slots, :, 0] = np.where(out["present"], out["dest"], 100); prev[l_slots, :, 1] = np.where(out["at_dest"], out["op"], 44)
        money_old = A["money"].copy()
        fin = env.step()
        money_new = A["money"]
        store["reward"][t] = a.dense * ((money_new[l_slots, 0] - money_new[l_slots, 1]) - (money_old[l_slots, 0] - money_old[l_slots, 1])) / 1000.0
        for g_, r0, r1, _, _ts in fin:
            for s_ in (0, 1):
                slot = g_ * 2 + s_
                if slot in slot_pos:
                    j = slot_pos[slot]; own, oth = (r0, r1) if s_ == 0 else (r1, r0); m0 = money_old[slot, 0] - money_old[slot, 1]
                    store["reward"][t, j] = a.dense * ((own - oth) - m0) / 1000.0 + a.terminal * (1.0 if own > oth else (-1.0 if own < oth else 0.0))
                    store["done"][t, j] = True; outcome[t, j] = own - oth
            prev[g_ * 2:g_ * 2 + 2] = 0; prev[g_ * 2:g_ * 2 + 2, :, 0] = 100; prev[g_ * 2:g_ * 2 + 2, :, 1] = 44
    with torch.no_grad():
        bt = {k: torch.from_numpy(A[k][l_slots]).to(dev) for k in KEYS_OBS}
        bt["prev"] = torch.from_numpy(prev[l_slots]).to(dev)
        H = model.encode(bt["tiles"], bt["units"], bt["items"], bt["glob"], bt["prev"])
        v_last = model.market(H)["value"].cpu().numpy()
    adv, ret = gae(store["reward"], store["value"], store["done"], a.gamma, a.lam, v_last)
    tfin = int(np.where(store["done"].any(1))[0].min()) if store["done"].any() else None
    has_term = np.zeros((T, L), dtype=bool)
    if tfin is not None: has_term[:tfin + 1] = True
    with torch.no_grad():
        gs = []
        for c0 in range(0, T * L, a.mb):
            ix = np.arange(c0, min(c0 + a.mb, T * L))
            bt = {k: torch.from_numpy(store[k].reshape(T * L, *store[k].shape[2:])[ix]).to(dev) for k in KEYS_OBS}
            bt["prev"] = torch.from_numpy(store["prev"].reshape(T * L, MAX_UNITS, 2)[ix]).to(dev)
            H = model.encode(bt["tiles"], bt["units"], bt["items"], bt["glob"], bt["prev"])
            gs.append(H["g"].cpu())
    return torch.cat(gs), torch.from_numpy(ret.reshape(-1)), torch.from_numpy(has_term.reshape(-1))


def r2(pred, target):
    v = target.var()
    return float(1.0 - (target - pred).var() / v) if v > 1e-12 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--windows", type=int, default=8); ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4096); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cap", type=int, default=400000, help="学習に使う行数の上限 (メモリ対策)")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--T", type=int, default=96); ap.add_argument("--mb", type=int, default=2048)
    ap.add_argument("--tape-frac", type=float, default=0.25); ap.add_argument("--tapes", default="tmp/rl/tapes_top.pkl")
    ap.add_argument("--seed0", type=int, default=600000); ap.add_argument("--dev", default="cuda")
    ap.add_argument("--temp", type=float, default=0.5); ap.add_argument("--gamma", type=float, default=0.997); ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--dense", type=float, default=0.05); ap.add_argument("--terminal", type=float, default=1.0)
    a = ap.parse_args(); dev = torch.device(a.dev); torch.manual_seed(0); np.random.seed(0)
    model = Policy2().to(dev); print("init:", model.load_state_dict(torch.load(a.init, map_location=dev), strict=False), flush=True); model.eval()
    env = VecEnv(a.workers, a.games, seed0=a.seed0, tapes_path=a.tapes if a.tape_frac > 0 else None, tape_frac=a.tape_frac)
    A = env.A; n = env.n
    l_slots = np.array([sl for sl in range(n) if A["role"][sl] == 0], dtype=np.int64)
    print(f"games {env.G} learner {len(l_slots)} tapes {env.n_tapes}", flush=True)
    prev = np.zeros((n, MAX_UNITS, 2), dtype=np.int16); prev[..., 0] = 100; prev[..., 1] = 44
    Xs = []; ys = []; ts = []; t0 = time.time()
    try:
        for w in range(a.windows):
            X, y, term = collect_window(a, model, dev, env, A, l_slots, prev)
            Xs.append(X); ys.append(y); ts.append(term)
            print(f"window {w}: rows {len(y)} term {int(term.sum())} [{time.time() - t0:.0f}s]", flush=True)
    finally:
        env.close()
    X = torch.cat(Xs); y = torch.cat(ys); term = torch.cat(ts)
    if a.cap and len(y) > a.cap:
        sel = torch.randperm(len(y))[:a.cap]; X = X[sel]; y = y[sel]; term = term[sel]
    print(f"rows {len(y)} (term {int(term.sum())}, mean ret {float(y.mean()):.3f}, std {float(y.std()):.3f})", flush=True)
    ntr = int(0.9 * len(y)); perm = torch.randperm(len(y)); tr, va = perm[:ntr], perm[ntr:]
    vh = copy.deepcopy(model.value_head).to(dev); vh.train()
    opt = torch.optim.Adam(vh.parameters(), lr=a.lr); bs = min(a.batch, len(tr))
    print(f"train value head: steps {a.steps} bs {bs} lr {a.lr}", flush=True)
    for s in range(a.steps):
        b = tr[torch.randint(0, len(tr), (bs,))]
        loss = ((vh(X[b].to(dev)).squeeze(-1) - y[b].to(dev)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % 500 == 0:
            vh.eval()
            with torch.no_grad(): pv = vh(X[va].to(dev)).squeeze(-1).cpu()
            yv = y[va]; tv = term[va]
            print(f"  step {s + 1}: R² val {r2(pv, yv):.3f} term {r2(pv[tv], yv[tv]):.3f} nonterm {r2(pv[~tv], yv[~tv]):.3f} [{time.time() - t0:.0f}s]", flush=True)
            vh.train()
    vh.eval()
    with torch.no_grad(): pv = vh(X[va].to(dev)).squeeze(-1).cpu()
    yv = y[va]; tv = term[va]
    print(f"FINAL R² val {r2(pv, yv):.3f} term {r2(pv[tv], yv[tv]):.3f} nonterm {r2(pv[~tv], yv[~tv]):.3f}", flush=True)
    model.value_head.load_state_dict(vh.state_dict())
    torch.save(model.state_dict(), a.out)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
