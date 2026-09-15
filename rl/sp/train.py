"""Self-play PPO for Policy2 on VecEnv (Majkel-style: one net plays both seats, frozen teacher with KL + promotion gate).
Resumable: every iteration saves <out>.state (model/opt/teacher/iter/windows); rerun with --resume to continue (games restart fresh).
Usage: caffeinate -i .venv/bin/python rl/sp/train.py --init tmp/rl/bc5_ep3.pt --out tmp/rl/sp1.pt --workers 10 --games 48 --T 64 --iters 100000
"""
from __future__ import annotations
import argparse, os, sys, time, copy, collections, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from model2 import Policy2
from features import MAX_UNITS, N_OPS
from vec_env import VecEnv
from policy_batch import act_batch, HEADS

QTY_OPS = lambda op: ((op >= 16) & (op <= 27)) | ((op >= 29) & (op <= 40))


def dists_batch(model, bt, temp):
    """Full log-distributions for every decision head (for analytic KL) plus per-decision log-probs [B,69], dest entropy [B], value [B]."""
    H = model.encode(bt["tiles"], bt["units"], bt["items"], bt["glob"], bt["prev"])
    dl = torch.log_softmax((model.dest_logits(H) / temp).masked_fill(~bt["dmask"], -1e9), -1)  # [B,U,100]
    opl, qtl = model.op_logits(H, bt["dest"]); opl = torch.log_softmax((opl / temp).masked_fill(~bt["omask"], -1e9), -1); qtl = torch.log_softmax(qtl / temp, -1)
    mk = model.market(H); ld = {"dest": dl, "op": opl, "qty": qtl}
    for k in ("sell", "buyp", "seed", "anim", "hire", "land"): ld[k] = torch.log_softmax(mk[k] / temp, -1)
    decide = bt["decide"].float(); at = bt["at_dest"].float(); isq = at * QTY_OPS(bt["op"]).float()
    g = lambda lp, idx: lp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    lp = [g(dl, bt["dest"]) * decide, g(opl, bt["op"]) * at, g(qtl, bt["qty"]) * isq]; base = 0
    for k, n in (("sell", 9), ("buyp", 2), ("seed", 5), ("anim", 3)): lp.append(g(ld[k], bt["mkt"][:, base:base + n])); base += n
    for k in ("hire", "land"): lp.append(g(ld[k], bt["mkt"][:, base]).unsqueeze(-1)); base += 1
    ent = (-(dl.exp() * dl.clamp(min=-30)).sum(-1) * decide).sum(-1) / decide.sum(-1).clamp(min=1)
    return ld, torch.cat(lp, -1), ent, mk["value"]


def kl_full(ld_t, ld_s, bt):
    """Analytic forward KL(teacher || student) summed over decision heads, averaged over live decisions (Majkel: full distributions, not k3 samples)."""
    decide = bt["decide"].float(); at = bt["at_dest"].float(); tot = 0.0; n = 0.0
    for k, w in (("dest", decide), ("op", at), ("qty", at)):
        kl = (ld_t[k].exp() * (ld_t[k].clamp(min=-30) - ld_s[k].clamp(min=-30))).sum(-1); tot = tot + (kl * w).sum(); n = n + w.sum()
    for k in ("sell", "buyp", "seed", "anim", "hire", "land"):
        kl = (ld_t[k].exp() * (ld_t[k].clamp(min=-30) - ld_s[k].clamp(min=-30))).sum(-1); tot = tot + kl.sum(); n = n + float(kl.numel())
    return tot / n.clamp(min=1) if torch.is_tensor(n) else tot / max(n, 1.0)


def logp_batch(model, bt, temp):
    _, lp, ent, v = dists_batch(model, bt, temp); return lp, ent, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True); ap.add_argument("--out", required=True); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--games", type=int, default=48); ap.add_argument("--T", type=int, default=64); ap.add_argument("--iters", type=int, default=1000000)
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--warmup", type=int, default=200, help="optimizer steps of linear lr warm-up (Adam's first steps move every weight by ~lr)"); ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--minibatches", type=int, default=4); ap.add_argument("--mb", type=int, default=768, help="samples per forward chunk (MPS memory); gradients accumulate to --minibatches optimizer steps per epoch")
    ap.add_argument("--gamma", type=float, default=0.997); ap.add_argument("--lam", type=float, default=0.95); ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.002); ap.add_argument("--vf", type=float, default=0.5); ap.add_argument("--kl", type=float, default=0.005, help="analytic forward KL(teacher||student) coefficient")
    ap.add_argument("--critic-warmup", type=int, default=60, help="optimizer steps with pg/entropy off (value + KL only): avoids the restart shock of an untrained critic")
    ap.add_argument("--temp", type=float, default=0.5, help="policy := softmax(logits/temp) for units (same in collection and update)")
    ap.add_argument("--dense", type=float, default=0.05, help="reward shaping: coef × Δ(own−opp)/1000 per step"); ap.add_argument("--terminal", type=float, default=1.0, help="terminal reward ±terminal by sign of final margin")
    ap.add_argument("--tapes", default="tmp/rl/tapes.pkl"); ap.add_argument("--tape-frac", type=float, default=0.5, help="fraction of games against a recorded real opponent (free, grounded)")
    ap.add_argument("--teacher-frac", type=float, default=0.3); ap.add_argument("--promote", type=float, default=0.53); ap.add_argument("--promote-min", type=int, default=150)
    ap.add_argument("--seed0", type=int, default=100000); ap.add_argument("--dev", default="mps")
    a = ap.parse_args(); dev = torch.device(a.dev); torch.manual_seed(0)
    n_upd = 0
    model = Policy2().to(dev); print("init:", model.load_state_dict(torch.load(a.init, map_location=dev), strict=False), flush=True)
    teacher = copy.deepcopy(model).eval(); opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    state_path = a.out + ".state"; it0 = 0; win_window = collections.deque(maxlen=400); tape_window = collections.deque(maxlen=400); promotions = 0; games_done = 0; seed_off = 0
    if a.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location=dev); model.load_state_dict(st["model"]); teacher.load_state_dict(st["teacher"]); opt.load_state_dict(st["opt"])
        it0 = st["iter"]; win_window = collections.deque(st["win_window"], maxlen=400); tape_window = collections.deque(st.get("tape_window", []), maxlen=400); promotions = st["promotions"]; games_done = st["games_done"]; seed_off = st.get("seed_off", 0) + 7_777_777; n_upd = st.get("n_upd", 0)
        print(f"resumed at iter {it0}, promotions {promotions}, games {games_done}", flush=True)
    env = VecEnv(a.workers, a.games, seed0=a.seed0 + seed_off, tapes_path=a.tapes if a.tape_frac > 0 else None, tape_frac=a.tape_frac); A = env.A; G, n = env.G, env.n; print(f"tapes in pool: {env.n_tapes}", flush=True)
    rng = np.random.default_rng(it0 + 1)
    # per-game role: 0 = learner both seats, 1 = teacher on seat0, 2 = teacher on seat1 (only for games without a recorded opponent)
    role = np.zeros(G, dtype=np.int8)
    def draw_role(gs):
        for g in gs: role[g] = 0 if rng.random() > a.teacher_frac else (1 + int(rng.random() < 0.5))
    draw_role(range(G))
    def slot_sets():
        tape_slots = set(np.where(A["role"] == 1)[0].tolist())
        t_slots = np.array([g * 2 + (0 if role[g] == 1 else 1) for g in range(G) if role[g] > 0 and (g * 2) not in tape_slots and (g * 2 + 1) not in tape_slots], dtype=np.int64)
        ts = set(t_slots.tolist()); l_slots = np.array([sl for sl in range(n) if sl not in tape_slots and sl not in ts], dtype=np.int64); return l_slots, t_slots
    prev = np.zeros((n, MAX_UNITS, 2), dtype=np.int16); prev[..., 0] = 100; prev[..., 1] = 44
    keys_obs = ("tiles", "units", "items", "glob")
    log = open(a.out + ".log", "a")
    try:
      for it in range(it0, a.iters):
          t0 = time.time(); model.eval()
          l_slots, t_slots = slot_sets(); L = len(l_slots)
          store = {k: np.zeros((a.T, L, *A[k].shape[1:]), dtype=A[k].dtype) for k in keys_obs}
          store.update(prev=np.zeros((a.T, L, MAX_UNITS, 2), dtype=np.int16), dmask=np.zeros((a.T, L, MAX_UNITS, 100), dtype=bool), omask=np.zeros((a.T, L, MAX_UNITS, N_OPS), dtype=bool),
                       at_dest=np.zeros((a.T, L, MAX_UNITS), dtype=bool), present=np.zeros((a.T, L, MAX_UNITS), dtype=bool), decide=np.zeros((a.T, L, MAX_UNITS), dtype=bool), dest=np.zeros((a.T, L, MAX_UNITS), dtype=np.int16),
                       op=np.zeros((a.T, L, MAX_UNITS), dtype=np.int16), qty=np.zeros((a.T, L, MAX_UNITS), dtype=np.int16), mkt=np.zeros((a.T, L, 21), dtype=np.int16),
                       logp=np.zeros((a.T, L), dtype=np.float32), logp_h=np.zeros((a.T, L, 9), dtype=np.float32), lp_dec=np.zeros((a.T, L, 69), dtype=np.float32), value=np.zeros((a.T, L), dtype=np.float32), reward=np.zeros((a.T, L), dtype=np.float32), done=np.zeros((a.T, L), dtype=bool))
          slot_pos = {int(s): j for j, s in enumerate(l_slots)}; margins_t = []; wins_t = []; wins_tape = []; margins_tape = []; t_inf = t_env = 0.0
          for t in range(a.T):
              if int(A["step"][0]) % 24 == 0: prev[:] = 0; prev[..., 0] = 100; prev[..., 1] = 44
              for k in keys_obs: store[k][t] = A[k][l_slots]
              store["prev"][t] = prev[l_slots]
              ti = time.perf_counter()
              out = act_batch(model, dev, A, l_slots, prev, temp=a.temp)
              if len(t_slots): out_t = act_batch(teacher, dev, A, t_slots, prev, temp=a.temp)
              t_inf += time.perf_counter() - ti
              for k in ("dmask", "omask", "at_dest", "present", "decide", "dest", "op", "qty", "mkt", "logp", "logp_h", "lp_dec", "value"): store[k][t] = out[k]
              prev[l_slots, :, 0] = np.where(out["present"], out["dest"], 100); prev[l_slots, :, 1] = np.where(out["at_dest"], out["op"], 44)
              if len(t_slots): prev[t_slots, :, 0] = np.where(out_t["present"], out_t["dest"], 100); prev[t_slots, :, 1] = np.where(out_t["at_dest"], out_t["op"], 44)
              money_old = A["money"].copy()
              te = time.perf_counter(); fin = env.step(); t_env += time.perf_counter() - te
              money_new = A["money"]
              r = a.dense * ((money_new[l_slots, 0] - money_new[l_slots, 1]) - (money_old[l_slots, 0] - money_old[l_slots, 1])) / 1000.0
              store["reward"][t] = r
              fin_games = []
              for g, r0, r1, _, tseat in fin:
                  games_done += 1; fin_games.append(g)
                  for s, own, oth in ((0, r0, r1), (1, r1, r0)):
                      slot = g * 2 + s
                      if slot in slot_pos:
                          j = slot_pos[slot]; m0 = money_old[slot, 0] - money_old[slot, 1]
                          store["reward"][t, j] = a.dense * ((own - oth) - m0) / 1000.0 + a.terminal * (1.0 if own > oth else (-1.0 if own < oth else 0.0)); store["done"][t, j] = True
                  if tseat >= 0:
                      lm = (r1 - r0) if tseat == 0 else (r0 - r1); margins_tape.append(lm); wins_tape.append(lm > 0); tape_window.append(lm > 0)
                  elif role[g] > 0:

                      lm = (r1 - r0) if role[g] == 1 else (r0 - r1); margins_t.append(lm); wins_t.append(lm > 0); win_window.append(lm > 0)
                  prev[g * 2:g * 2 + 2] = 0; prev[g * 2:g * 2 + 2, :, 0] = 100; prev[g * 2:g * 2 + 2, :, 1] = 44
              if fin_games:
                  draw_role(fin_games); new_l, new_t = slot_sets()
                  if not np.array_equal(new_l, l_slots):  # roles changed under us: keep collecting with the old slot set this iteration, roles apply next iteration
                      pass
          with torch.no_grad():
              _, _, v_last = logp_batch(model, {**{k: torch.from_numpy(A[k][l_slots]).to(dev) for k in keys_obs}, "prev": torch.from_numpy(prev[l_slots]).to(dev),
                                              "dmask": torch.from_numpy(store["dmask"][-1]).to(dev), "omask": torch.from_numpy(store["omask"][-1]).to(dev), "present": torch.from_numpy(store["present"][-1]).to(dev), "decide": torch.from_numpy(store["decide"][-1]).to(dev),
                                              "at_dest": torch.from_numpy(store["at_dest"][-1]).to(dev), "dest": torch.from_numpy(store["dest"][-1].astype(np.int64)).to(dev), "op": torch.from_numpy(store["op"][-1].astype(np.int64)).to(dev),
                                              "qty": torch.from_numpy(store["qty"][-1].astype(np.int64)).to(dev), "mkt": torch.from_numpy(store["mkt"][-1].astype(np.int64)).to(dev)}, a.temp)
              v_last = v_last.cpu().numpy()
          # GAE
          adv = np.zeros((a.T, L), dtype=np.float32); last = np.zeros(L, dtype=np.float32)
          for t in reversed(range(a.T)):
              nv = v_last if t + 1 == a.T else store["value"][t + 1]; nd = 1.0 - store["done"][t]
              delta = store["reward"][t] + a.gamma * nv * nd - store["value"][t]; last = delta + a.gamma * a.lam * nd * last; adv[t] = last
          ret = adv + store["value"]; N = a.T * L
          flat = {k: v.reshape(N, *v.shape[2:]) for k, v in store.items()}; adv_f = ((adv - adv.mean()) / (adv.std() + 1e-8)).reshape(N); ret_f = ret.reshape(N)
          if os.environ.get("SP_DEBUG_LP"):
              with torch.no_grad():
                  for t in (0, 1, a.T // 2, a.T - 1):
                      bt = {k: torch.from_numpy(store[k][t]).to(dev) for k in keys_obs + ("prev", "dmask", "omask", "at_dest", "present", "decide")}
                      for k in ("dest", "op", "qty", "mkt"): bt[k] = torch.from_numpy(store[k][t].astype(np.int64)).to(dev)
                      lp, _, _ = logp_batch(model, bt, a.temp); d = np.abs(lp.sum(-1).cpu().numpy() - store["logp"][t]); print(f"  debug t={t}: max|dlogp| {d.max():.4f} mean {d.mean():.4f} worst slot {d.argmax()}", flush=True)
          model.eval(); stats = []; mb = N // a.minibatches; t_upd = time.time()  # eval: dropout off so the recomputed log-probs match collection (train() gave |r-1| 0.76 before any update)
          for ep in range(a.epochs):
              perm = np.random.permutation(N)
              for b in range(0, N - mb + 1, mb):
                  opt.zero_grad(); chunks = list(range(b, b + mb, a.mb)); acc = np.zeros(5)
                  for c0 in chunks:
                      idx = perm[c0:min(c0 + a.mb, b + mb)]
                      bt = {k: torch.from_numpy(flat[k][idx]).to(dev) for k in keys_obs + ("prev", "dmask", "omask", "at_dest", "present", "decide")}
                      for k in ("dest", "op", "qty", "mkt"): bt[k] = torch.from_numpy(flat[k][idx].astype(np.int64)).to(dev)
                      ld_s, lp, ent, value = dists_batch(model, bt, a.temp); old = torch.from_numpy(flat["lp_dec"][idx]).to(dev); Aadv = torch.from_numpy(adv_f[idx]).to(dev).unsqueeze(-1); live = (old != 0).float()
                      ratio = torch.exp(lp - old); pg = -(torch.min(ratio * Aadv, torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * Aadv) * live).sum() / live.sum().clamp(min=1)
                      with torch.no_grad(): ld_t, _, _, _ = dists_batch(teacher, bt, a.temp)
                      kl = kl_full(ld_t, ld_s, bt)
                      vl = ((value - torch.from_numpy(ret_f[idx]).to(dev)) ** 2).mean()
                      warm = n_upd < a.critic_warmup; loss = ((0.0 if warm else pg) + a.vf * vl - (0.0 if warm else a.ent) * ent.mean() + a.kl * kl) * (len(idx) / mb)
                      loss.backward(); acc += np.array([pg.item(), vl.item(), ent.mean().item(), (((ratio - 1).abs() * live).sum() / live.sum().clamp(min=1)).item(), kl.item()]) * (len(idx) / mb)
                  n_upd += 1
                  for gp in opt.param_groups: gp["lr"] = a.lr * min(1.0, n_upd / max(1, a.warmup))
                  torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step(); stats.append(tuple(acc))
                  if os.environ.get("SP_DEBUG_LP") and len(stats) <= 4: print(f"  minibatch {len(stats)}: |r-1| {stats[-1][3]:.4f} kl {stats[-1][4]:.4f}", flush=True)
          s = np.mean(stats, 0); wr = float(np.mean(win_window)) if win_window else float("nan")
          promoted = "" if n_upd >= a.critic_warmup else " (critic warm-up)"
          if len(win_window) >= a.promote_min and wr >= a.promote:
              teacher.load_state_dict(model.state_dict()); teacher.eval(); promotions += 1; win_window.clear(); promoted = f" PROMOTED#{promotions}"
          el = time.time() - t0
          line = (f"it {it} steps {N} ({N / el:.0f} samples/s; inf {t_inf:.0f}s env {t_env:.0f}s upd {time.time() - t_upd:.0f}s) | vsT games {len(margins_t)} margin {np.mean(margins_t) if margins_t else float('nan'):+.0f} "
                  f"win {np.mean(wins_t) if wins_t else float('nan'):.2f} window {wr:.3f} (n{len(win_window)}) | vsTAPE {len(margins_tape)}g margin {np.mean(margins_tape) if margins_tape else float('nan'):+.0f} window {np.mean(tape_window) if tape_window else float('nan'):.3f} (n{len(tape_window)}) | pg {s[0]:.3f} vf {s[1]:.3f} ent {s[2]:.2f} |r-1| {s[3]:.3f} kl {s[4]:.4f} | games {games_done}{promoted}")
          print(line, flush=True); log.write(line + "\n"); log.flush()
          torch.save(model.state_dict(), a.out)
          torch.save({"model": model.state_dict(), "teacher": teacher.state_dict(), "opt": opt.state_dict(), "iter": it + 1, "win_window": list(win_window), "tape_window": list(tape_window), "promotions": promotions, "games_done": games_done, "seed_off": seed_off, "n_upd": n_upd}, state_path)
          if (it + 1) % 25 == 0: torch.save(model.state_dict(), a.out.replace(".pt", f"_it{it + 1}.pt"))
    finally:
        env.close()


if __name__ == "__main__":
    main()
