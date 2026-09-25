"""Raw action representation (2026-09-24): the engine's own action space, so a perfect model reproduces the teacher exactly.
Policy3's "next work tile + op, walk there greedily" cannot express positioning moves (v41 steps NORTH at t=1 to place
the hands it hires, then returns) or waiting for a precondition, and its market heads (per-item sums, MKT_BUCKETS, fixed
order) cannot express the order list: with perfect labels v41 fell from 125k to ~30k (rl/diag/oracle.py).

  units : per unit and step one of features.OPS (moves included) + quantity class
  market: 10 slots (the engine processes 10 per turn), each a kind (END / NOP = slot-owning no-op such as [] / SELL x12 /
          BUY_PRODUCT x9 / BUY_SEED x5 / BUY_ANIMAL x3 / HIRE / BUY_LAND) + quantity class
  qty   : 0..99 exact, class 100 = ">=100" (decoded 1000). Shed capacity is 100, so every quantity >= 100 means "all".

Exactness check (repo root): .venv/bin/python rl/raw.py --check 'tmp/rootcause/v41json/episode-*.json' --team v41a
Evaluation vs v41:           .venv/bin/python rl/raw.py CKPT --games 64
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from features import ANIMALS, CROPS, ITEMS, MAX_UNITS, N_OPS, OP_INDEX, OPS, PRODUCTS, SHED_TILES, encode, legal_ops
from actions import unit_op
from act_common import trim_plants

N_Q = 101
N_SLOTS = 10
KINDS = (["END", "NOP"] + ["SELL_" + i for i in ITEMS] + ["BUYP_" + p for p in PRODUCTS] + ["SEED_" + c for c in CROPS]
         + ["ANIMAL_" + a for a in ANIMALS] + ["HIRE", "LAND"])
KIND_INDEX = {k: i for i, k in enumerate(KINDS)}
N_KINDS = len(KINDS)
_ORDER_PREFIX = {"SELL": "SELL_", "BUY_PRODUCT": "BUYP_", "BUY_SEED": "SEED_", "BUY_ANIMAL": "ANIMAL_"}
_KIND_ORDER = {v: k for k, v in _ORDER_PREFIX.items()}
QTY_OPS = np.array([o.startswith(("PICKUP_", "PLACE_")) for o in OPS])
QTY_KINDS = np.array([k.split("_")[0] in ("SELL", "BUYP", "SEED", "ANIMAL") for k in KINDS])


def qclass(q): return int(min(max(int(q), 0), 100))
def qvalue(c): return 1000 if int(c) >= 100 else int(c)


def encode_raw(action, n_units):
    """engine action dict -> (uop [MAX_UNITS] -1 absent, uqty [MAX_UNITS], mk [N_SLOTS], mq [N_SLOTS])."""
    action = action or {}
    acts = [action.get("farmer") or ["PASS"], *(action.get("hands") or [])]
    uop = np.full(MAX_UNITS, -1, np.int8); uqty = np.zeros(MAX_UNITS, np.int8)
    for i in range(min(n_units, MAX_UNITS)):
        name, q = unit_op(acts[i] if i < len(acts) else ["PASS"])
        uop[i] = OP_INDEX[name]; uqty[i] = qclass(q)
    mk = np.zeros(N_SLOTS, np.int8); mq = np.zeros(N_SLOTS, np.int8)
    for j, o in enumerate((action.get("market") or [])[:N_SLOTS]):
        kind, q = "NOP", 0
        if isinstance(o, list) and o:
            if o[0] == "HIRE": kind = "HIRE"
            elif o[0] == "BUY_LAND": kind = "LAND"
            elif o[0] in _ORDER_PREFIX and len(o) >= 3 and (_ORDER_PREFIX[o[0]] + str(o[1])) in KIND_INDEX:
                kind, q = _ORDER_PREFIX[o[0]] + str(o[1]), o[2]
        mk[j] = KIND_INDEX[kind]; mq[j] = qclass(q)
    return uop, uqty, mk, mq


def decode_raw(uop, uqty, mk, mq, n_units):
    units = []
    for i in range(n_units):
        name = OPS[int(uop[i])] if i < MAX_UNITS and int(uop[i]) >= 0 else "PASS"; q = qvalue(uqty[i]) if i < MAX_UNITS else 1
        if name.startswith("PLANT_"): units.append(["PLANT", name[6:]])
        elif name.startswith("PICKUP_"): units.append(["PICKUP", name[7:], q])
        elif name.startswith("PLACE_"): units.append(["PLACE", name[6:], q])
        else: units.append([name])
    market = []
    for k, q in zip(mk, mq):
        kind = KINDS[int(k)]
        if kind == "END": continue
        if kind == "NOP": market.append([])
        elif kind == "HIRE": market.append(["HIRE"])
        elif kind == "LAND": market.append(["BUY_LAND"])
        else:
            head, item = kind.split("_", 1); market.append([_KIND_ORDER[head + "_"], item, qvalue(q)])
    return {"farmer": units[0] if units else ["PASS"], "hands": units[1:], "market": market}


def unit_mask(obs, seat):
    """legal_ops plus PLACE of a carried animal at the shed (the engine puts it in the shed; legal_ops omits it)."""
    m = legal_ops(obs, seat)
    farm = obs["farms"][seat]; invs = obs["private"]["inventories"]
    for i, u in enumerate([farm["farmer"], *farm["hands"]][:MAX_UNITS]):
        if (int(u[0]), int(u[1])) in SHED_TILES:
            inv = invs[i] if i < len(invs) else {}
            for a in ANIMALS:
                if int(inv.get(a, 0) or 0) > 0: m[i, OP_INDEX["PLACE_" + a]] = True
    return m


def extract_raw(steps, seat, T=719):
    """steps[t][seat] = {"observation", "action"} with action[t+1] answering observation[t] (replay convention)."""
    out = {k: [] for k in ("tiles", "units", "items", "glob", "mask", "uop", "uqty", "mk", "mq")}
    for t in range(T):
        obs = steps[t][seat]["observation"]; obs["step"] = t
        f = encode(obs, seat); n = min(MAX_UNITS, 1 + len(obs["farms"][seat]["hands"]))
        for k in ("tiles", "units", "items", "glob"): out[k].append(f[k])
        out["mask"].append(unit_mask(obs, seat))
        for k, v in zip(("uop", "uqty", "mk", "mq"), encode_raw(steps[t + 1][seat]["action"], n)): out[k].append(v)
    return {k: np.stack(v) for k, v in out.items()}


# ------------------------------------------------------------------ model
try:
    import torch, torch.nn as nn, torch.nn.functional as F
    from features import GLOB_F, ITEM_F, TILE_F, UNIT_F

    class Policy4(nn.Module):
        """Policy3's token encoder (global + 200 tiles + units + items) plus N_SLOTS market-slot tokens."""
        def __init__(self, d=128, layers=3, heads=4, ff=256):
            super().__init__()
            self.kind_emb = nn.Embedding(8, 16); self.crop_emb = nn.Embedding(6, 8); self.anim_emb = nn.Embedding(4, 8)
            self.tile_in = nn.Linear(16 + 8 + 8 + (TILE_F - 3), d); self.tile_pos = nn.Parameter(torch.randn(200, d) * 0.02)
            self.unit_in = nn.Linear(UNIT_F, d); self.unit_pos = nn.Parameter(torch.randn(MAX_UNITS, d) * 0.02)
            self.item_in = nn.Linear(ITEM_F, d); self.item_pos = nn.Parameter(torch.randn(len(PRODUCTS), d) * 0.02)
            self.glob_in = nn.Linear(GLOB_F, d); self.type_emb = nn.Embedding(5, d)
            # Discrete clock: teacher schedules fire at exact (day, hour) slots (sell fertilizer on days 12/13/15/21..., buy
            # animals at day 9 h1); from the continuous day/29, hour/23 the clone missed them at the same steps every game.
            self.day_emb = nn.Embedding(30, d); self.hour_emb = nn.Embedding(24, d); self.slot = nn.Parameter(torch.randn(N_SLOTS, d) * 0.02)
            layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False); self.norm = nn.LayerNorm(d)
            self.uop = nn.Linear(d, N_OPS); self.uqty = nn.Linear(d, N_Q); self.mkind = nn.Linear(d, N_KINDS); self.mqty = nn.Linear(d, N_Q)
            self.value = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

        def forward(self, tiles, units, items, glob):
            B = tiles.shape[0]; t = tiles.reshape(B, 200, TILE_F)
            num = t[..., 3:].float() / torch.tensor([30, 6, 1, 3, 3, 1, 1, 3, 1, 3, 8, 3, 1, 9, 9], device=t.device, dtype=torch.float32)
            tt = self.tile_in(torch.cat([self.kind_emb(t[..., 0].long().clamp(0, 7)), self.crop_emb(t[..., 1].long().clamp(0, 5)),
                                         self.anim_emb(t[..., 2].long().clamp(0, 3)), num], -1)) + self.tile_pos + self.type_emb.weight[1]
            u = units.float() / torch.tensor([1, 1, 9, 9] + [40] * 12 + [1, 16], device=units.device, dtype=torch.float32)
            ut = self.unit_in(u) + self.unit_pos + self.type_emb.weight[2]
            it = self.item_in(items) + self.item_pos + self.type_emb.weight[3]
            day = (glob[:, 0] * 29).round().long().clamp(0, 29); hour = (glob[:, 1] * 23).round().long().clamp(0, 23)
            gt = (self.glob_in(glob) + self.type_emb.weight[0] + self.day_emb(day) + self.hour_emb(hour)).unsqueeze(1)
            st = (self.slot + self.type_emb.weight[4]).unsqueeze(0).expand(B, -1, -1)
            x = torch.cat([gt, tt, ut, it, st], 1)
            pad = torch.zeros(B, x.shape[1], dtype=torch.bool, device=x.device); pad[:, 201:201 + MAX_UNITS] = units[..., 0] <= 0
            h = self.norm(self.enc(x, src_key_padding_mask=pad))
            hu = h[:, 201:201 + MAX_UNITS]; hs = h[:, -N_SLOTS:]
            return {"uop": self.uop(hu), "uqty": self.uqty(hu), "mkind": self.mkind(hs), "mqty": self.mqty(hs), "value": self.value(h[:, 0]).squeeze(-1)}

    def raw_loss(out, b):
        """Plain likelihood of the teacher's action over all N_OPS, no legality mask and no class weights.
        A training mask leaves the masked logits untrained, so inference must apply the same mask, and legal_ops forbids
        the teacher's no-op waits (COLLECT_FERTILIZER with none available, ...): v41's clone left the trajectory at t=107
        that way. Class weights distort the imitated distribution (pos_w / OP_WEIGHT in Policy3)."""
        uop = b["uop"].long(); present = uop >= 0
        lu = F.cross_entropy(out["uop"][present], uop[present])
        qm = present & torch.as_tensor(QTY_OPS, device=uop.device)[uop.clamp(min=0)]
        lq = F.cross_entropy(out["uqty"][qm], b["uqty"].long()[qm]) if qm.any() else lu.new_zeros(())
        mk = b["mk"].long(); lk = F.cross_entropy(out["mkind"].reshape(-1, N_KINDS), mk.reshape(-1))
        km = torch.as_tensor(QTY_KINDS, device=mk.device)[mk]
        lmq = F.cross_entropy(out["mqty"][km], b["mq"].long()[km]) if km.any() else lu.new_zeros(())
        loss = lu + lq + lk + lmq
        with torch.no_grad():
            pu = out["uop"].argmax(-1); pk = out["mkind"].argmax(-1)
            acc = {"uop": (pu == uop)[present].float().mean().item(), "move": (pu == uop)[present & (uop >= 1) & (uop <= 4)].float().mean().item(),
                   "uqty": (out["uqty"].argmax(-1) == b["uqty"].long())[qm].float().mean().item() if qm.any() else 1.0,
                   "mkind": (pk == mk).float().mean().item(), "mstep": (pk == mk).all(-1).float().mean().item(),
                   "mqty": (out["mqty"].argmax(-1) == b["mq"].long())[km].float().mean().item() if km.any() else 1.0}
        return loss, acc

    class RawAgent:
        def __init__(self, path, device="cpu", mask=False):
            # mask=False for models trained by raw_loss (no mask). mask=True only for the first checkpoints (09-24 18:00,
            # kernel raw-bc-v41 v1), which were trained with a legal_ops mask and have untrained logits outside it.
            self.dev = torch.device(device); self.mask = mask; self.model = Policy4().to(self.dev)
            missing, unexpected = self.model.load_state_dict(torch.load(path, map_location=self.dev), strict=False)
            assert not unexpected and set(missing) <= {"day_emb.weight", "hour_emb.weight"}, (missing, unexpected)
            if missing: self.model.day_emb.weight.data.zero_(); self.model.hour_emb.weight.data.zero_()   # pre-clock checkpoints
            self.model.eval()

        def act(self, obs, seat):
            f = encode(obs, seat); n = 1 + len(obs["farms"][seat]["hands"])
            with torch.no_grad():
                o = self.model(*(torch.from_numpy(f[k]).unsqueeze(0).to(self.dev) for k in ("tiles", "units", "items", "glob")))
            lu = o["uop"][0].cpu().numpy()
            if self.mask: lu[~unit_mask(obs, seat)] = -1e9
            a = decode_raw(lu.argmax(-1), o["uqty"][0].argmax(-1).cpu().numpy(), o["mkind"][0].argmax(-1).cpu().numpy(), o["mqty"][0].argmax(-1).cpu().numpy(), n)
            units = [a["farmer"], *a["hands"]]; trim_plants(units, obs["private"]["seeds"]); a["farmer"], a["hands"] = units[0], units[1:]
            return a
except ImportError:   # numpy-only environments can still use the representation
    pass


# ------------------------------------------------------------------ CLI
def check(pattern, team):
    """encode -> decode must reproduce each recorded game exactly in kagsim (same seed, same shops, opponent = record)."""
    import kagsim
    for f in sorted(glob.glob(pattern)):
        r = json.load(open(f)); names = r["info"]["TeamNames"]; me = names.index(team); st = r["steps"]
        g = kagsim.Game(r["info"]["seed"], 720, st[-1][0]["observation"]["town"]["unlocked_shops"]); div = None
        while not g.done:
            t = g.step_count; obs = g.observe(me); rec = st[t + 1][me]["action"] or {}
            ro = st[t][me]["observation"]["farms"][me]; so = obs["farms"][me]
            if div is None and (ro["money"] != so["money"] or ro["farmer"] != so["farmer"] or ro["hands"] != so["hands"]): div = t
            n = 1 + len(obs["farms"][me]["hands"]); a = decode_raw(*encode_raw(rec, n), n)
            opp = st[t + 1][1 - me]["action"] or {}
            g.step(*((a, opp) if me == 0 else (opp, a)))
        got = (g.reward(me), g.reward(1 - me)); want = (r["rewards"][me], r["rewards"][1 - me])
        print(f"{os.path.basename(f)}: recorded {want[0]:.0f}/{want[1]:.0f}  decoded {got[0]:.0f}/{got[1]:.0f}  {'EXACT' if got == tuple(want) else 'DIFFERENT'}  first-divergence {div}")


def evaluate(ckpt, games, vs, seed0, mask=False):
    import kagsim, importlib.util
    agent = RawAgent(ckpt, mask=mask); sys.path.insert(0, os.path.dirname(os.path.abspath(vs)))
    spec = importlib.util.spec_from_file_location("opp_raw", vs); opp = importlib.util.module_from_spec(spec); spec.loader.exec_module(opp)
    rows = []; t0 = time.time()
    for gi in range(games):
        seed, seat = seed0 + gi // 2, gi % 2
        g = kagsim.Game(seed)
        while not g.done:
            a = agent.act(g.observe(seat), seat); b = opp.agent(g.observe(1 - seat))
            g.step(*((a, b) if seat == 0 else (b, a)))
        rows.append((g.reward(seat), g.reward(1 - seat)))
        print(f"seed {seed} seat {seat}: own {rows[-1][0]:8.0f} opp {rows[-1][1]:8.0f} margin {rows[-1][0] - rows[-1][1]:+8.0f}", flush=True)
    r = np.array(rows); m = r[:, 0] - r[:, 1]
    print(f"mean own {r[:, 0].mean():.0f} opp {r[:, 1].mean():.0f} margin {m.mean():+.0f} ± {m.std(ddof=1) / np.sqrt(len(m)) if len(m) > 1 else 0:.0f} wins {(m > 0).sum()}/{len(m)} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("ckpt", nargs="?"); ap.add_argument("--check"); ap.add_argument("--team", default="v41a")
    ap.add_argument("--games", type=int, default=16); ap.add_argument("--vs", default="third_party/public_agents/v41/main.py"); ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--mask", action="store_true", help="legal_ops mask at inference (only for checkpoints trained with it)")
    a = ap.parse_args()
    if a.check: check(a.check, a.team)
    else: evaluate(a.ckpt, a.games, a.vs, a.seed0, a.mask)
