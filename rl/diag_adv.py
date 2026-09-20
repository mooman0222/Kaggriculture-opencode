"""Advantage/critic 診断 (学習なし): ある ckpt の方策でゲームを回し、PPO が使う信号の情報量を測る。

測るもの (train.py と同一の収集・GAE を再現):
  1. V の説明分散: 窓内 MC リターン G_t = Σ γ^k r_{t+k} (bootstrap なし) に対する R²。
     pooled / context (tape / self) / 終局を含む窓のみ。
  2. セグメント単位の sum(adv) と実際の最終 margin の相関・符号一致 (pooled / context)。
     adv が勝敗を当てているなら符号一致 > 0.5 + 相関 > 0。
  3. 報酬の分解: 窓内の dense 成分と terminal 成分の分散比と、V がどちらを追えているか。
  4. pg 勾配の信号/ノイズ比: 同じバッチで adv を permute した対照と勾配を比べ、
     ||g_real − g_shuf|| / ||g_real|| (1 に近いほど信号が乗っている、0 ならほぼノイズ)。
     ノイズ床として shuf 同士 2 本の差も出す。

ゲームは 720 手で終わるので、`--chunks` 個の窓 (各 T=96 手) を回すと 8 個目に終局が入る。
終局を含む窓でだけセグメント統計を取る。

使い方:
  PYTHONPATH=.venv/lib/python3.14/site-packages /usr/bin/python3 rl/diag_adv.py \
    --ckpt tmp/rl/bc5_ep3.pt --out tmp/rl/diag_bc5.json --workers 4 --games 120 --dev cuda
"""
from __future__ import annotations
import argparse, copy, json, os, sys, time
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "sp"))
import numpy as np, torch
from model2 import Policy2
from features import MAX_UNITS, N_OPS
from vec_env import VecEnv
from policy_batch import act_batch

QTY_OPS = lambda op: ((op >= 16) & (op <= 27)) | ((op >= 29) & (op <= 40))
KEYS_OBS = ("tiles", "units", "items", "glob")


def dists_batch(model, bt, temp):
    H = model.encode(bt["tiles"], bt["units"], bt["items"], bt["glob"], bt["prev"])
    dl = torch.log_softmax((model.dest_logits(H) / temp).masked_fill(~bt["dmask"], -1e9), -1)
    opl, qtl = model.op_logits(H, bt["dest"]); opl = torch.log_softmax((opl / temp).masked_fill(~bt["omask"], -1e9), -1); qtl = torch.log_softmax(qtl / temp, -1)
    mk = model.market(H); ld = {"dest": dl, "op": opl, "qty": qtl}
    for k in ("sell", "buyp", "seed", "anim", "hire", "land"): ld[k] = torch.log_softmax(mk[k] / temp, -1)
    decide = bt["decide"].float(); at = bt["at_dest"].float(); isq = at * QTY_OPS(bt["op"]).float()
    g = lambda lp, idx: lp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    lp = [g(dl, bt["dest"]) * decide, g(opl, bt["op"]) * at, g(qtl, bt["qty"]) * isq]; base = 0
    for k, n in (("sell", 9), ("buyp", 2), ("seed", 5), ("anim", 3)): lp.append(g(ld[k], bt["mkt"][:, base:base + n])); base += n
    for k in ("hire", "land"): lp.append(g(ld[k], bt["mkt"][:, base]).unsqueeze(-1)); base += 1
    return ld, torch.cat(lp, -1), mk["value"]


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


def mc_return(reward, gamma):
    T, L = reward.shape
    out = np.zeros_like(reward); acc = np.zeros(L, dtype=np.float32)
    for t in reversed(range(T)):
        acc = reward[t] + gamma * acc
        out[t] = acc
    return out


def r2(pred, target):
    v = target.var()
    return float(1.0 - (target - pred).var() / v) if v > 1e-12 else float("nan")


def pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 1e-12 else float("nan")


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(np.float64); ry = np.argsort(np.argsort(y)).astype(np.float64)
    return pearson(rx, ry)


def build_store(T, L, A, l_slots):
    s = {k: np.zeros((T, L, *A[k].shape[1:]), dtype=A[k].dtype) for k in KEYS_OBS}
    s.update(prev=np.zeros((T, L, MAX_UNITS, 2), dtype=np.int16), dmask=np.zeros((T, L, MAX_UNITS, 100), dtype=bool), omask=np.zeros((T, L, MAX_UNITS, N_OPS), dtype=bool),
             at_dest=np.zeros((T, L, MAX_UNITS), dtype=bool), present=np.zeros((T, L, MAX_UNITS), dtype=bool), decide=np.zeros((T, L, MAX_UNITS), dtype=bool),
             dest=np.zeros((T, L, MAX_UNITS), dtype=np.int16), op=np.zeros((T, L, MAX_UNITS), dtype=np.int16), qty=np.zeros((T, L, MAX_UNITS), dtype=np.int16),
             mkt=np.zeros((T, L, 21), dtype=np.int16), logp=np.zeros((T, L), dtype=np.float32), lp_dec=np.zeros((T, L, 69), dtype=np.float32),
             value=np.zeros((T, L), dtype=np.float32), reward=np.zeros((T, L), dtype=np.float32), done=np.zeros((T, L), dtype=bool))
    return s


def collect(a, model, dev, env, A, l_slots, store, prev):
    T, L = a.T, len(l_slots)
    roles = np.zeros((T, L), dtype=np.int8); opp_tape = np.zeros((T, L), dtype=bool); outcome = np.full((T, L), np.nan)
    slot_pos = {int(s): j for j, s in enumerate(l_slots)}
    for t in range(T):
        if int(A["step"][0]) % 24 == 0: prev[:] = 0; prev[..., 0] = 100; prev[..., 1] = 44
        for k in KEYS_OBS: store[k][t] = A[k][l_slots]
        store["prev"][t] = prev[l_slots]; roles[t] = A["role"][l_slots]; opp_tape[t] = A["role"][l_slots ^ 1] == 1
        out = act_batch(model, dev, A, l_slots, prev, temp=a.temp)
        for k in ("dmask", "omask", "at_dest", "present", "decide", "dest", "op", "qty", "mkt", "logp", "lp_dec", "value"): store[k][t] = out[k]
        prev[l_slots, :, 0] = np.where(out["present"], out["dest"], 100); prev[l_slots, :, 1] = np.where(out["at_dest"], out["op"], 44)
        money_old = A["money"].copy()
        fin = env.step()
        money_new = A["money"]
        store["reward"][t] = a.dense * ((money_new[l_slots, 0] - money_new[l_slots, 1]) - (money_old[l_slots, 0] - money_old[l_slots, 1])) / 1000.0
        for g_, r0, r1, _, tseat in fin:
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
        for k in ("dmask", "omask", "present", "decide", "at_dest"): bt[k] = torch.from_numpy(store[k][-1]).to(dev)
        for k in ("dest", "op", "qty", "mkt"): bt[k] = torch.from_numpy(store[k][-1].astype(np.int64)).to(dev)
        v_last = dists_batch(model, bt, a.temp)[2].cpu().numpy()
    return roles, opp_tape, outcome, v_last


def segment_stats(store, opp_tape, outcome, adv):
    T, L = store["reward"].shape
    segs = []
    for j in range(L):
        t0 = 0
        for t in range(T):
            if store["done"][t, j]:
                m = outcome[t, j]; ctx = "tape" if opp_tape[t, j] else "self"
                r_sum = float(store["reward"][t0:t + 1, j].sum()); term = float(np.sign(m))
                segs.append(dict(t0=t0, t1=t, margin=float(m), ctx=ctx, term=term, dense_sum=r_sum - term,
                                 adv_sum=float(adv[t0:t + 1, j].sum()), adv_mean=float(adv[t0:t + 1, j].mean())))
                t0 = t + 1
    return segs


def gradsnr(a, model, dev, store, adv_flat, n=4096):
    """分割ハーフ法: 別々のサンプルから出した 2 本の pg 勾配の cos が信号の自己一貫性。
    adv を permute した対照 (同じ 2 分割) と比べる。cos ~ 0 なら方向はノイズ。"""
    T, L = store["value"].shape; N = T * L
    flat = {k: v.reshape(N, *v.shape[2:]) for k, v in store.items()}
    rng = np.random.default_rng(7); idx = rng.choice(N, size=min(n, N), replace=False)
    h = len(idx) // 2; iA, iB = idx[:h], idx[h:2 * h]

    def grad(ix, av):
        bt = {k: torch.from_numpy(flat[k][ix]).to(dev) for k in KEYS_OBS + ("prev", "dmask", "omask", "at_dest", "present", "decide")}
        for k in ("dest", "op", "qty", "mkt"): bt[k] = torch.from_numpy(flat[k][ix].astype(np.int64)).to(dev)
        old = torch.from_numpy(flat["lp_dec"][ix]).to(dev)
        model.zero_grad()
        _, lp, _ = dists_batch(model, bt, a.temp)
        live = (old != 0).float(); ratio = torch.exp(lp - old)
        Aadv = torch.from_numpy(av).to(dev).unsqueeze(-1)
        pg = -(torch.min(ratio * Aadv, torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * Aadv) * live).sum() / live.sum().clamp(min=1)
        pg.backward()
        return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).clone()

    avA = adv_flat[iA].copy(); avB = adv_flat[iB].copy(); p = rng.permutation(h)
    gA, gB = grad(iA, avA), grad(iB, avB)
    gAs, gBs = grad(iA, avA[p]), grad(iB, avB[p])
    cos = lambda x, y: float(torch.nn.functional.cosine_similarity(x, y, dim=0).item())
    rel = lambda x, y: float((x - y).norm().item() / x.norm().item()) if x.norm() > 0 else float("nan")
    return {"n_half": int(h), "cos_half_real": cos(gA, gB), "cos_half_shuf": cos(gAs, gBs),
            "normA": float(gA.norm()), "normB": float(gB.norm()),
            "rel_diff_real": rel(gA, gB), "rel_diff_shuf": rel(gAs, gBs)}


def fit_value_probe(a, model, dev, store, G, m, steps=1500, mb=2048):
    """同じ観測特徴の上で value head を教師ありで焼き直し、表現力 vs 最適化を切り分ける。
    凍結特徴 H['g'] を取り、MC リターン G を回帰する。val 20%。"""
    T, L = store["value"].shape
    flat = {k: v.reshape(T * L, *v.shape[2:]) for k, v in store.items()}
    idx = np.where(m.reshape(-1))[0]
    gs = []
    with torch.no_grad():
        for c0 in range(0, len(idx), mb):
            ix = idx[c0:c0 + mb]
            bt = {k: torch.from_numpy(flat[k][ix]).to(dev) for k in KEYS_OBS + ("prev",)}
            H = model.encode(bt["tiles"], bt["units"], bt["items"], bt["glob"], bt["prev"])
            gs.append(H["g"].cpu())
    X = torch.cat(gs); y = torch.from_numpy(G.reshape(-1)[idx])
    perm = torch.randperm(len(idx)); ntr = int(0.8 * len(idx)); tr, va = perm[:ntr], perm[ntr:]
    vh = copy.deepcopy(model.value_head).to(dev); vh.train()
    opt = torch.optim.Adam(vh.parameters(), lr=1e-3)
    for _ in range(steps):
        b = tr[torch.randint(0, len(tr), (min(mb, len(tr)),))]
        loss = ((vh(X[b].to(dev)).squeeze(-1) - y[b].to(dev)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    vh.eval()
    with torch.no_grad():
        pt = vh(X[tr].to(dev)).squeeze(-1).cpu().numpy(); pv = vh(X[va].to(dev)).squeeze(-1).cpu().numpy()
    yt = y[tr].numpy(); yv = y[va].numpy()
    return {"n": int(len(idx)), "r2_train": r2(pt, yt), "r2_val": r2(pv, yv),
            "r2_val_const": r2(np.full_like(yv, yv.mean()), yv), "mse_val": float(((pv - yv) ** 2).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--T", type=int, default=96); ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--tape-frac", type=float, default=0.25); ap.add_argument("--tapes", default="tmp/rl/tapes_top.pkl")
    ap.add_argument("--seed0", type=int, default=500000); ap.add_argument("--dev", default="cuda")
    ap.add_argument("--temp", type=float, default=0.5); ap.add_argument("--gamma", type=float, default=0.997); ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--dense", type=float, default=0.05); ap.add_argument("--terminal", type=float, default=1.0)
    ap.add_argument("--clip", type=float, default=0.2); ap.add_argument("--gradsnr", type=int, default=1)
    ap.add_argument("--fit-value", type=int, default=1, help="同じ特徴で value head を教師あり再学習し、表現力 vs 最適化を切り分ける")
    a = ap.parse_args(); dev = torch.device(a.dev); torch.manual_seed(0); np.random.seed(0)
    model = Policy2().to(dev); print("init:", model.load_state_dict(torch.load(a.ckpt, map_location=dev), strict=False), flush=True); model.eval()
    env = VecEnv(a.workers, a.games, seed0=a.seed0, tapes_path=a.tapes if a.tape_frac > 0 else None, tape_frac=a.tape_frac)
    A = env.A; n = env.n
    l_slots = np.array([sl for sl in range(n) if A["role"][sl] == 0], dtype=np.int64)
    print(f"games {env.G} slots {n} learner {len(l_slots)} tapes {env.n_tapes}", flush=True)
    prev = np.zeros((n, MAX_UNITS, 2), dtype=np.int16); prev[..., 0] = 100; prev[..., 1] = 44
    out = {"ckpt": a.ckpt, "chunks": []}; t00 = time.time()
    try:
        for k in range(a.chunks):
            T, L = a.T, len(l_slots)
            store = build_store(T, L, A, l_slots)
            roles, opp_tape, outcome, v_last = collect(a, model, dev, env, A, l_slots, store, prev)
            adv, ret = gae(store["reward"], store["value"], store["done"], a.gamma, a.lam, v_last)
            G = mc_return(store["reward"], a.gamma)
            fin_mask = store["done"].any(0)  # slot j finishes somewhere in this chunk
            t_fin = int(np.where(store["done"].any(1))[0].min()) if store["done"].any() else None
            rec = {"chunk": k, "t_fin": t_fin, "n_fin": int(fin_mask.sum()), "n_tape_fin": int((opp_tape & store["done"]).any(0).sum())}
            if t_fin is not None:
                m = np.zeros_like(store["done"]); m[:t_fin + 1] = True  # 終局前の行だけ (新ゲームの部分窓を除く)
                rec["r2_mc_pooled"] = r2(store["value"][m], G[m])
                for ctx, rv in (("self", 0), ("tape", 1)):
                    sel = m & (roles == rv)
                    if sel.sum() > 100: rec[f"r2_mc_{ctx}"] = r2(store["value"][sel], G[sel])
                rec["var_G"] = float(G[m].var()); rec["mean_G"] = float(G[m].mean())
                rec["v_mean"] = float(store["value"][m].mean()); rec["v_std"] = float(store["value"][m].std()); rec["v_absmean"] = float(np.abs(store["value"][m]).mean())
                dm = store["done"][t_fin]
                rec["corr_v_last_margin"] = pearson(store["value"][t_fin][dm], outcome[t_fin][dm])
                rec["sign_acc_v_last"] = float((np.sign(store["value"][t_fin][dm]) == np.sign(outcome[t_fin][dm])).mean())
                mid = max(0, t_fin // 2); dmid = m[mid]
                rec["corr_v_mid_margin"] = pearson(store["value"][mid][dmid], outcome[t_fin][:len(dmid)][dmid]) if dmid.any() else float("nan")
                segs = [s for s in segment_stats(store, opp_tape, outcome, adv) if s["t1"] == t_fin]  # 完走セグメントのみ (新ゲーム側を除く)
                mg = np.array([s["margin"] for s in segs]); aa = np.array([s["adv_mean"] for s in segs]); aa_s = np.array([s["adv_sum"] for s in segs])
                ds = np.array([s["dense_sum"] for s in segs])
                rec["dense_sum_std"] = float(ds.std()); rec["terminal_frac"] = float(np.mean(np.abs(ds) < np.abs(np.array([s["term"] for s in segs]))))
                nz = mg != 0
                rec["segments"] = len(segs)
                if nz.sum() > 8:
                    rec["seg_corr_mean"] = pearson(aa[nz], mg[nz]); rec["seg_spearman_mean"] = spearman(aa[nz], mg[nz])
                    rec["seg_corr_sum"] = pearson(aa_s[nz], mg[nz])
                    rec["seg_sign_agree"] = float((np.sign(aa[nz]) == np.sign(mg[nz])).mean())
                    for ctx in ("self", "tape"):
                        cm = nz & np.array([s["ctx"] == ctx for s in segs])
                        if cm.sum() > 8:
                            rec[f"seg_sign_agree_{ctx}"] = float((np.sign(aa[cm]) == np.sign(mg[cm])).mean())
                            rec[f"seg_corr_{ctx}"] = pearson(aa[cm], mg[cm])
                rec["margin_std"] = float(mg.std()) if len(mg) else float("nan")
            if a.gradsnr and store["done"].any():
                rec["gradsnr"] = gradsnr(a, model, dev, store, adv.reshape(-1))
            if a.fit_value and t_fin is not None:
                rec["fit_value"] = fit_value_probe(a, model, dev, store, G, m)
            out["chunks"].append(rec)
            print(f"chunk {k}: t_fin {t_fin} fin {rec['n_fin']} " + " ".join(f"{kk} {vv:.3f}" if isinstance(vv, float) else f"{kk} {vv}" for kk, vv in rec.items() if kk not in ("chunk",)) , flush=True)
    finally:
        env.close()
    out["seconds"] = round(time.time() - t00, 1)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
