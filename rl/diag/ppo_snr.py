"""Is the policy gradient of the raw policy signal or noise at our batch sizes? (no training)
Roll out G games vs OPP sampling unit ops and market kinds at temperature TEMP (quantities greedy), return = final margin.
Split games into halves A/B, compute the REINFORCE gradient (return - mean) of each half, report cos(gA, gB);
control = the same with returns shuffled across games (should be ~0).
usage (repo root): .venv/bin/python rl/diag/ppo_snr.py CKPT [games] [temp]"""
import sys, os, importlib.util
import numpy as np, torch
sys.path.insert(0, os.path.join(os.getcwd(), "rl"))
import kagsim
from raw import Policy4, decode_raw
from features import encode
from act_common import trim_plants

torch.set_num_threads(2)
ck = sys.argv[1]; G = int(sys.argv[2]) if len(sys.argv) > 2 else 40; TEMP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
m = Policy4(); miss = m.load_state_dict(torch.load(ck, map_location="cpu"), strict=False); m.eval()
spec = importlib.util.spec_from_file_location("opp", "third_party/public_agents/v41/main.py"); opp = importlib.util.module_from_spec(spec); spec.loader.exec_module(opp)
rng = np.random.default_rng(0); games = []
K = ("tiles", "units", "items", "glob")
CACHE = os.environ.get("CACHE")   # pickle of rollouts: reuse the same games across split repetitions
import pickle
if CACHE and os.path.exists(CACHE): games = pickle.load(open(CACHE, "rb")); G = len(games)
for gi in range(len(games), G):
    seed, seat = 7000 + gi // 2, gi % 2; g = kagsim.Game(seed); rec = {k: [] for k in K + ("uop", "mk", "n")}
    while not g.done:
        o = g.observe(seat); f = encode(o, seat); n = 1 + len(o["farms"][seat]["hands"])
        with torch.no_grad(): out = m(*(torch.from_numpy(f[k]).unsqueeze(0) for k in K))
        pu = torch.softmax(out["uop"][0] / TEMP, -1).numpy(); pk = torch.softmax(out["mkind"][0] / TEMP, -1).numpy()
        uop = np.array([rng.choice(len(p), p=p / p.sum()) for p in pu]); mk = np.array([rng.choice(len(p), p=p / p.sum()) for p in pk])
        a = decode_raw(uop, out["uqty"][0].argmax(-1).numpy(), mk, out["mqty"][0].argmax(-1).numpy(), n)
        units = [a["farmer"], *a["hands"]]; trim_plants(units, o["private"]["seeds"]); a["farmer"], a["hands"] = units[0], units[1:]
        for k in K: rec[k].append(f[k])
        rec["uop"].append(uop); rec["mk"].append(mk); rec["n"].append(n)
        b = opp.agent(g.observe(1 - seat)); g.step(*((a, b) if seat == 0 else (b, a)))
    rec = {k: np.stack(v) for k, v in rec.items()}; rec["R"] = (g.reward(seat) - g.reward(1 - seat)) / 10000.0
    games.append(rec); print(f"game {gi}: own {g.reward(seat):.0f} margin {rec['R'] * 1e4:+.0f}", flush=True)

if CACHE and not os.path.exists(CACHE): pickle.dump(games, open(CACHE, "wb"))
R = np.array([x["R"] for x in games]); print(f"sampled policy (temp {TEMP}): mean margin {R.mean() * 1e4:+.0f} ± {R.std(ddof=1) / np.sqrt(len(R)) * 1e4:.0f}")


def grad(idx, adv):
    m.zero_grad(); params = [p for p in m.parameters() if p.requires_grad]
    for j, gi in enumerate(idx):
        x = games[gi]
        for s in range(0, len(x["uop"]), 256):
            sl = slice(s, s + 256)
            out = m(*(torch.from_numpy(x[k][sl].astype(np.int16) if k in ("tiles", "units") else x[k][sl]) for k in K))
            lu = torch.log_softmax(out["uop"] / TEMP, -1).gather(-1, torch.from_numpy(x["uop"][sl]).long().unsqueeze(-1)).squeeze(-1)
            live = torch.arange(lu.shape[1]).unsqueeze(0) < torch.from_numpy(x["n"][sl]).unsqueeze(1)
            lk = torch.log_softmax(out["mkind"] / TEMP, -1).gather(-1, torch.from_numpy(x["mk"][sl]).long().unsqueeze(-1)).squeeze(-1)
            (-(float(adv[j])) * ((lu * live).sum() + lk.sum())).backward()
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten().clone() for p in params])


cos = lambda a, b: float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))
adv = R - R.mean(); reals, nulls = [], []
for rep in range(int(os.environ.get("SPLITS", "1"))):
    if os.environ.get("PAIRS") == "1":   # keep both seats of a seed in the same half (games 2k and 2k+1 share seed 7000+k)
        pr = rng.permutation(G // 2); A = np.concatenate([[2 * k, 2 * k + 1] for k in pr[:G // 4]]); B = np.concatenate([[2 * k, 2 * k + 1] for k in pr[G // 4:]])
    else:
        order = rng.permutation(G); A, B = order[:G // 2], order[G // 2:]
    c = lambda x: x - x.mean()   # baseline within each half: a shared baseline makes the two halves' advantage sums opposite
    reals.append(cos(grad(A, c(R[A])), grad(B, c(R[B]))))   # and pushes any common gradient component to a negative cosine
    sh = rng.permutation(R); nulls.append(cos(grad(A, c(sh[A])), grad(B, c(sh[B]))))
    print(f"split {rep}: real {reals[-1]:+.3f}  shuffled {nulls[-1]:+.3f}", flush=True)
r, n = np.array(reals), np.array(nulls)
print(f"split-half cosine ({G // 2} games per half, {len(r)} splits): real {r.mean():+.3f} ± {r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else 0:.3f}   shuffled {n.mean():+.3f} ± {n.std(ddof=1) / np.sqrt(len(n)) if len(n) > 1 else 0:.3f}")
