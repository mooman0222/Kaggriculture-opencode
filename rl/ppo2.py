"""PPO (Policy2) fine-tuning from a BC checkpoint against an opponent pool. 各 iter で <out>.state を保存、`--resume` で続きから。使い方:
.venv/bin/python rl/ppo.py --init tmp/rl/bc2.pt --out tmp/rl/ppo.pt --iters 100 --games 32 [--opps third_party/public_agents/v41/main.py agents/e060/main.py]"""
import argparse, os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from model2 import Policy2 as Policy
from rollout import ScriptedOpp
from rollout2 import rollout2 as rollout, logp2, model_out_fn, PolicyOpp


def gae(rew, val, gamma=1.0, lam=0.95):
    T, G = rew.shape; adv = np.zeros_like(rew); last = np.zeros(G, dtype=np.float32)
    for t in reversed(range(T)):
        nv = val[t + 1] if t + 1 < T else 0.0
        delta = rew[t] + gamma * nv - val[t]; last = delta + gamma * lam * last; adv[t] = last
    return adv, adv + val


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--init", required=True); ap.add_argument("--out", required=True); ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--games", type=int, default=32); ap.add_argument("--opps", nargs="+", default=["third_party/public_agents/v41/main.py", "agents/e060/main.py"], help="main.py (スクリプト) か .pt (凍結した学習方策) を混在可")
    ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--bs", type=int, default=512); ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.003); ap.add_argument("--vf", type=float, default=0.5); ap.add_argument("--opening", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="<out>.state から再開 (model/opt/iter)"); ap.add_argument("--temp", type=float, default=0.7, help="dest/op のサンプリング温度")
    a = ap.parse_args(); dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    import rollout2; rollout2.TEMP = a.temp
    model = Policy().to(dev); model.load_state_dict(torch.load(a.init, map_location=dev)); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    opps = [PolicyOpp(p, "cpu") if p.endswith(".pt") else ScriptedOpp(p) for p in a.opps]; opening = ScriptedOpp("agents/e058/main.py") if a.opening else None; fn = model_out_fn(model)
    state_path = a.out + ".state"; start = 0
    if a.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location=dev); model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); start = st["iter"]; print(f"resumed at iter {start}", flush=True)
    for it in range(start, a.iters):
        t0 = time.time(); traj, res = rollout(model, dev, opps, a.games, seed0=10000 + it * a.games, opening=opening, opening_steps=a.opening)
        margins = [r0 - r1 for r0, r1 in res]; own = [r0 for r0, _ in res]
        adv, ret = gae(traj["reward"], traj["value"]); T, G = adv.shape; N = T * G
        flat = {k: torch.from_numpy(v.reshape(N, *v.shape[2:])) for k, v in traj.items()}
        adv_t = torch.from_numpy(((adv - adv.mean()) / (adv.std() + 1e-8)).reshape(N)); ret_t = torch.from_numpy(ret.reshape(N))
        model.train(); stats = []
        for ep in range(a.epochs):
            perm = torch.randperm(N)
            for b in range(0, N, a.bs):
                idx = perm[b:b + a.bs]; bt = {k: v[idx].to(dev) for k, v in flat.items()}
                act = {k: bt[k].long() for k in ("dest", "op", "qty", "sell", "buyp", "seed", "anim", "hire", "land")}
                logp, ent = logp2(fn, bt, act, per_head=True)  # [B,K] per sub-action
                value = model.market(model.encode(bt["tiles"], bt["units"], bt["items"], bt["glob"], bt["prev"]))["value"]
                ratio = torch.exp(logp - bt["logp"]); A = adv_t[idx].to(dev).unsqueeze(-1)
                live = (bt["logp"] != 0)  # sub-actions that were actually taken (absent units / off-destination ops carry logp 0)
                pg = -(torch.min(ratio * A, torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * A) * live).sum() / live.sum().clamp(min=1)
                vl = ((value - ret_t[idx].to(dev)) ** 2).mean(); loss = pg + a.vf * vl - a.ent * ent.mean()
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()
                stats.append((pg.item(), vl.item(), ent.mean().item(), ((ratio - 1).abs() * live).sum().item() / live.sum().clamp(min=1).item()))
        s = np.mean(stats, 0)
        print(f"it {it} margin {np.mean(margins):+7.0f} own {np.mean(own):7.0f} wins {sum(m > 0 for m in margins)}/{len(margins)} | pg {s[0]:.3f} vf {s[1]:.3f} ent {s[2]:.2f} |r-1| {s[3]:.3f} [{time.time()-t0:.0f}s]", flush=True)
        torch.save(model.state_dict(), a.out); torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "iter": it + 1}, state_path)
        if it % 10 == 9: torch.save(model.state_dict(), a.out.replace(".pt", f"_it{it+1}.pt"))


if __name__ == "__main__":
    main()
