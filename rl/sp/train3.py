"""Self-play PPO for Policy3 on VecEnv (frozen teacher with analytic KL + promotion gate, recorded-opponent tapes).

Differences from train.py (Policy2): one joint (tile, op) decision per unit, held until arrival (far fewer decisions per game);
exact arrival legality as the option mask; dense shaping that also charges plant deaths and animal escapes directly, so the
credit for "water that plant / feed that animal" arrives within the day instead of at the season's end.
Resumable: every iteration saves <out>.state; rerun with --resume.
Usage: caffeinate -i .venv/bin/python rl/sp/train3.py --init tmp/kaggle_out_bc9/bc9k_ep3.pt --out tmp/rl/sp4.pt --workers 10 --games 48 --T 96
"""
from __future__ import annotations

import argparse
import collections
import copy
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from features import MAX_UNITS
from model3 import Policy3
from policy_batch3 import TOPK, act_batch3, dists_batch3, kl_full3
from vec_env import VecEnv

OBS_KEYS = ("tiles", "units", "items", "glob")
N_DEC = MAX_UNITS + MAX_UNITS + 21


def farm_counts(A, slots):
    """(weeds, animals) per slot from the own-farm tile kinds (2 = WEED, 6 = animal)."""
    kind = A["tiles"][slots][:, 0, :, :, 0]
    return (kind == 2).sum((1, 2)).astype(np.int32), (kind == 6).sum((1, 2)).astype(np.int32)


def to_batch(store_or_flat, idx, dev, flat=True):
    src = (lambda k: store_or_flat[k][idx]) if flat else (lambda k: store_or_flat[k])
    bt = {k: torch.from_numpy(src(k)).to(dev) for k in OBS_KEYS + ("cmask", "decide", "at_dest", "present")}
    for k in ("option", "cand", "qty", "mkt"):
        bt[k] = torch.from_numpy(src(k).astype(np.int64)).to(dev)
    return bt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--games", type=int, default=48)
    ap.add_argument("--T", type=int, default=96)
    ap.add_argument("--iters", type=int, default=1000000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--mb", type=int, default=512)
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.0)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--kl", type=float, default=0.02, help="analytic forward KL(teacher||student) coefficient")
    ap.add_argument("--critic-warmup", type=int, default=60)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--dense", type=float, default=0.2, help="coef x delta(own-opp)/1000 per step")
    ap.add_argument("--death", type=float, default=0.05, help="penalty per plant turned into a weed")
    ap.add_argument("--escape", type=float, default=0.2, help="penalty per animal that escaped")
    ap.add_argument("--terminal", type=float, default=1.0)
    ap.add_argument("--tapes", default="tmp/rl/tapes_top.pkl")
    ap.add_argument("--tape-frac", type=float, default=0.25)
    ap.add_argument("--teacher-frac", type=float, default=0.3)
    ap.add_argument("--promote", type=float, default=0.53)
    ap.add_argument("--promote-min", type=int, default=150)
    ap.add_argument("--seed0", type=int, default=100000)
    ap.add_argument("--dev", default="mps")
    a = ap.parse_args()
    dev = torch.device(a.dev)
    torch.manual_seed(0)
    model = Policy3().to(dev)
    print("init:", model.load_state_dict(torch.load(a.init, map_location=dev), strict=False), flush=True)
    teacher = copy.deepcopy(model).eval()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    state_path = a.out + ".state"
    it0 = 0
    n_upd = 0
    win_window = collections.deque(maxlen=400)
    tape_window = collections.deque(maxlen=400)
    promotions = games_done = seed_off = 0
    if a.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location=dev)
        model.load_state_dict(st["model"])
        teacher.load_state_dict(st["teacher"])
        opt.load_state_dict(st["opt"])
        it0, promotions, games_done, n_upd = st["iter"], st["promotions"], st["games_done"], st["n_upd"]
        win_window = collections.deque(st["win_window"], maxlen=400)
        tape_window = collections.deque(st["tape_window"], maxlen=400)
        seed_off = st["seed_off"] + 7_777_777
        print(f"resumed at iter {it0}, promotions {promotions}, games {games_done}", flush=True)
    env = VecEnv(a.workers, a.games, seed0=a.seed0 + seed_off, tapes_path=a.tapes if a.tape_frac > 0 else None, tape_frac=a.tape_frac)
    A, G, n = env.A, env.G, env.n
    print(f"tapes in pool: {env.n_tapes}", flush=True)
    rng = np.random.default_rng(it0 + 1)
    role = np.zeros(G, dtype=np.int8)  # 0 learner both seats, 1 teacher on seat0, 2 teacher on seat1

    def draw_role(gs):
        for g in gs:
            role[g] = 0 if rng.random() > a.teacher_frac else (1 + int(rng.random() < 0.5))

    draw_role(range(G))

    def slot_sets():
        tape_slots = set(np.where(A["role"] == 1)[0].tolist())
        t_slots = np.array([g * 2 + (0 if role[g] == 1 else 1) for g in range(G) if role[g] > 0 and (g * 2) not in tape_slots and (g * 2 + 1) not in tape_slots], dtype=np.int64)
        ts = set(t_slots.tolist())
        l_slots = np.array([s for s in range(n) if s not in tape_slots and s not in ts], dtype=np.int64)
        return l_slots, t_slots

    held = np.full((n, MAX_UNITS), -1, dtype=np.int32)
    log = open(a.out + ".log", "a")
    try:
        for it in range(it0, a.iters):
            t0 = time.time()
            model.eval()
            l_slots, t_slots = slot_sets()
            L = len(l_slots)
            store = {k: np.zeros((a.T, L, *A[k].shape[1:]), dtype=A[k].dtype) for k in OBS_KEYS}
            store.update(option=np.zeros((a.T, L, MAX_UNITS), np.int16), cand=np.zeros((a.T, L, MAX_UNITS, TOPK), np.int16), cmask=np.zeros((a.T, L, MAX_UNITS, TOPK), bool),
                         qty=np.zeros((a.T, L, MAX_UNITS), np.int16), mkt=np.zeros((a.T, L, 21), np.int16), decide=np.zeros((a.T, L, MAX_UNITS), bool),
                         at_dest=np.zeros((a.T, L, MAX_UNITS), bool), present=np.zeros((a.T, L, MAX_UNITS), bool), lp_dec=np.zeros((a.T, L, N_DEC), np.float32),
                         value=np.zeros((a.T, L), np.float32), reward=np.zeros((a.T, L), np.float32), done=np.zeros((a.T, L), bool))
            slot_pos = {int(s): j for j, s in enumerate(l_slots)}
            margins_t, wins_t, margins_tape, wins_tape = [], [], [], []
            t_inf = t_env = 0.0
            deaths = escapes = decisions = 0
            weeds_old, animals_old = farm_counts(A, l_slots)
            for t in range(a.T):
                if int(A["step"][0]) % 24 == 0:
                    held[:] = -1  # hands are dropped at the end of the day: unit index i is a different worker tomorrow
                for k in OBS_KEYS:
                    store[k][t] = A[k][l_slots]
                ti = time.perf_counter()
                out = act_batch3(model, dev, A, l_slots, held, temp=a.temp)
                if len(t_slots):
                    act_batch3(teacher, dev, A, t_slots, held, temp=a.temp)
                t_inf += time.perf_counter() - ti
                for k in ("option", "cand", "cmask", "qty", "mkt", "decide", "at_dest", "present", "lp_dec", "value"):
                    store[k][t] = out[k]
                decisions += int(out["decide"].sum())
                money_old = A["money"].copy()
                te = time.perf_counter()
                fin = env.step()
                t_env += time.perf_counter() - te
                money_new = A["money"]
                weeds_new, animals_new = farm_counts(A, l_slots)
                r = a.dense * ((money_new[l_slots, 0] - money_new[l_slots, 1]) - (money_old[l_slots, 0] - money_old[l_slots, 1])) / 1000.0
                d = np.clip(weeds_new - weeds_old, 0, None)
                e = np.clip(animals_old - animals_new, 0, None)
                fresh = np.zeros(L, dtype=bool)  # slots whose game restarted this step: their tile diff is meaningless
                for g, r0, r1, _, tseat in fin:
                    games_done += 1
                    for s, own, oth in ((0, r0, r1), (1, r1, r0)):
                        slot = g * 2 + s
                        if slot in slot_pos:
                            j = slot_pos[slot]
                            fresh[j] = True
                            m0 = money_old[slot, 0] - money_old[slot, 1]
                            r[j] = a.dense * ((own - oth) - m0) / 1000.0 + a.terminal * (1.0 if own > oth else (-1.0 if own < oth else 0.0))
                            store["done"][t, j] = True
                    if tseat >= 0:
                        lm = (r1 - r0) if tseat == 0 else (r0 - r1)
                        margins_tape.append(lm)
                        wins_tape.append(lm > 0)
                        tape_window.append(lm > 0)
                    elif role[g] > 0:
                        lm = (r1 - r0) if role[g] == 1 else (r0 - r1)
                        margins_t.append(lm)
                        wins_t.append(lm > 0)
                        win_window.append(lm > 0)
                    held[g * 2:g * 2 + 2] = -1
                d[fresh] = 0
                e[fresh] = 0
                deaths += int(d.sum())
                escapes += int(e.sum())
                store["reward"][t] = r - a.death * d - a.escape * e
                weeds_old, animals_old = weeds_new, animals_new
                if fin:
                    draw_role([g for g, *_ in fin])
            with torch.no_grad():
                last = {k: A[k][l_slots] for k in OBS_KEYS}
                last.update({k: store[k][-1] for k in ("cand", "cmask", "decide", "at_dest", "present", "option", "qty", "mkt")})
                _, _, _, v_last = dists_batch3(model, to_batch(last, None, dev, flat=False), a.temp)
                v_last = v_last.cpu().numpy()
            adv = np.zeros((a.T, L), dtype=np.float32)
            last_adv = np.zeros(L, dtype=np.float32)
            for t in reversed(range(a.T)):
                nv = v_last if t + 1 == a.T else store["value"][t + 1]
                nd = 1.0 - store["done"][t]
                delta = store["reward"][t] + a.gamma * nv * nd - store["value"][t]
                last_adv = delta + a.gamma * a.lam * nd * last_adv
                adv[t] = last_adv
            ret = adv + store["value"]
            N = a.T * L
            flat = {k: v.reshape(N, *v.shape[2:]) for k, v in store.items()}
            adv_f = ((adv - adv.mean()) / (adv.std() + 1e-8)).reshape(N)
            ret_f = ret.reshape(N)
            if os.environ.get("SP_DEBUG_LP"):
                with torch.no_grad():
                    for t in (0, a.T // 2, a.T - 1):
                        idx = np.arange(t * L, (t + 1) * L)
                        _, lp, _, _ = dists_batch3(model, to_batch(flat, idx, dev), a.temp)
                        diff = np.abs(lp.cpu().numpy() - flat["lp_dec"][idx])
                        print(f"  debug t={t}: max|dlogp| {diff.max():.5f} mean {diff.mean():.6f}", flush=True)
            model.eval()
            stats = []
            mb = N // a.minibatches
            t_upd = time.time()
            for _ in range(a.epochs):
                perm = np.random.permutation(N)
                for b in range(0, N - mb + 1, mb):
                    opt.zero_grad()
                    acc = np.zeros(5)
                    for c0 in range(b, b + mb, a.mb):
                        idx = perm[c0:min(c0 + a.mb, b + mb)]
                        bt = to_batch(flat, idx, dev)
                        ld_s, lp, ent, value = dists_batch3(model, bt, a.temp)
                        old = torch.from_numpy(flat["lp_dec"][idx]).to(dev)
                        adv_t = torch.from_numpy(adv_f[idx]).to(dev).unsqueeze(-1)
                        live = (old != 0).float()
                        ratio = torch.exp(lp - old)
                        pg = -(torch.min(ratio * adv_t, torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * adv_t) * live).sum() / live.sum().clamp(min=1)
                        with torch.no_grad():
                            ld_t, _, _, _ = dists_batch3(teacher, bt, a.temp)
                        kl = kl_full3(ld_t, ld_s, bt)
                        vl = ((value - torch.from_numpy(ret_f[idx]).to(dev)) ** 2).mean()
                        warm = n_upd < a.critic_warmup
                        loss = ((0.0 if warm else pg) + a.vf * vl - (0.0 if warm else a.ent) * ent.mean() + a.kl * kl) * (len(idx) / mb)
                        loss.backward()
                        acc += np.array([pg.item(), vl.item(), ent.mean().item(), (((ratio - 1).abs() * live).sum() / live.sum().clamp(min=1)).item(), kl.item()]) * (len(idx) / mb)
                    n_upd += 1
                    for gp in opt.param_groups:
                        gp["lr"] = a.lr * min(1.0, n_upd / max(1, a.warmup))
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    opt.step()
                    stats.append(tuple(acc))
            s = np.mean(stats, 0)
            wr = float(np.mean(win_window)) if win_window else float("nan")
            promoted = "" if n_upd >= a.critic_warmup else " (critic warm-up)"
            if len(win_window) >= a.promote_min and wr >= a.promote:
                teacher.load_state_dict(model.state_dict())
                teacher.eval()
                promotions += 1
                win_window.clear()
                promoted = f" PROMOTED#{promotions}"
            el = time.time() - t0
            line = (f"it {it} steps {N} ({N / el:.0f} samples/s; inf {t_inf:.0f}s env {t_env:.0f}s upd {time.time() - t_upd:.0f}s) decisions/step {decisions / N:.2f} deaths {deaths} escapes {escapes} | "
                    f"vsT {len(margins_t)}g margin {np.mean(margins_t) if margins_t else float('nan'):+.0f} window {wr:.3f} (n{len(win_window)}) | "
                    f"vsTAPE {len(margins_tape)}g margin {np.mean(margins_tape) if margins_tape else float('nan'):+.0f} window {np.mean(tape_window) if tape_window else float('nan'):.3f} (n{len(tape_window)}) | "
                    f"pg {s[0]:.3f} vf {s[1]:.3f} ent {s[2]:.2f} |r-1| {s[3]:.3f} kl {s[4]:.4f} | games {games_done}{promoted}")
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            torch.save(model.state_dict(), a.out)
            torch.save({"model": model.state_dict(), "teacher": teacher.state_dict(), "opt": opt.state_dict(), "iter": it + 1, "win_window": list(win_window), "tape_window": list(tape_window),
                        "promotions": promotions, "games_done": games_done, "seed_off": seed_off, "n_upd": n_upd}, state_path)
            if (it + 1) % 25 == 0:
                torch.save(model.state_dict(), a.out.replace(".pt", f"_it{it + 1}.pt"))
    finally:
        env.close()


if __name__ == "__main__":
    main()
