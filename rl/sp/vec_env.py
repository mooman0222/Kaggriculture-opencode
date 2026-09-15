"""Vectorised self-play environment: W worker processes × K games, both seats driven by policies in the main process.
Workers do engine step + observation encoding + arrival legality (numpy), main does batched inference. Shared-memory buffers, lockstep protocol."""
from __future__ import annotations
import os, sys, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pickle, random
from multiprocessing import shared_memory

from features import MAX_UNITS, TILE_F, UNIT_F, ITEM_F, GLOB_F, N_OPS, OPS, PRODUCTS, CROPS, QTY_BUCKETS, unbucket, SHED_TILES
from actions import decode_action
from act2 import step_toward, trim_plants

N_MKT = 21  # sell 9 + buyp 2 + seed 5 + anim 3 + hire + land
SPEC = {  # per slot (game, seat)
    "tiles": ((2, 10, 10, TILE_F), np.int16), "units": ((MAX_UNITS, UNIT_F), np.int16), "items": ((9, ITEM_F), np.float32), "glob": ((GLOB_F,), np.float32),
    "legal": ((MAX_UNITS, 100, N_OPS), np.bool_), "pos": ((MAX_UNITS,), np.int16),  # unit tile index (-1 absent)
    "money": ((2,), np.float32),  # own, opp
    "act_dest": ((MAX_UNITS,), np.int16), "act_op": ((MAX_UNITS,), np.int16), "act_qty": ((MAX_UNITS,), np.int16), "act_mkt": ((N_MKT,), np.int16),
}
PASS = {"farmer": ["PASS"], "hands": [], "market": []}


class Buffers:
    """Shared arrays with leading dim [n_slots]. n_slots = games * 2."""
    def __init__(self, n_slots, create, names=None):
        self.n = n_slots; self.shm = {}; self.arr = {}; self.names = names or {}
        for k, (shape, dt) in SPEC.items():
            nbytes = int(np.prod((n_slots, *shape))) * np.dtype(dt).itemsize
            if create: s = shared_memory.SharedMemory(create=True, size=nbytes); self.names[k] = s.name
            else: s = shared_memory.SharedMemory(name=self.names[k])
            self.shm[k] = s; self.arr[k] = np.ndarray((n_slots, *shape), dtype=dt, buffer=s.buf)
        for k, dt in (("step", np.int32), ("role", np.int8)):  # role per slot: 0 = policy seat, 1 = tape (recorded real opponent) seat
            nb = int(np.dtype(dt).itemsize) * n_slots
            if create: s = shared_memory.SharedMemory(create=True, size=nb); self.names[k] = s.name
            else: s = shared_memory.SharedMemory(name=self.names[k])
            self.shm[k] = s; self.arr[k] = np.ndarray((n_slots,), dtype=dt, buffer=s.buf)

    def close(self, unlink=False):
        for s in self.shm.values():
            s.close()
            if unlink:
                try: s.unlink()
                except FileNotFoundError: pass


def _worker(wid, names, n_slots, slot0, K, seed0, conn, tapes, tape_frac):
    import kagsim
    buf = Buffers(n_slots, create=False, names=names); A = buf.arr
    seed_ctr = [seed0]; rng = random.Random(seed0); tape_of = [None] * K  # (seat, acts) when this game has a recorded opponent
    def new_game(gi):
        if tapes and rng.random() < tape_frac:
            tp = rng.choice(tapes); tape_of[gi] = (tp["seat"], tp["acts"]); g = kagsim.Game(tp["seed"])
            A["role"][slot0 + gi * 2 + tp["seat"]] = 1; A["role"][slot0 + gi * 2 + 1 - tp["seat"]] = 0
        else:
            tape_of[gi] = None; seed_ctr[0] += 1; g = kagsim.Game(seed_ctr[0]); A["role"][slot0 + gi * 2:slot0 + gi * 2 + 2] = 0
        return g
    games = [new_game(gi) for gi in range(K)]; obs = [[None, None] for _ in range(K)]
    def publish(gi):
        g = games[gi]
        for s in (0, 1):
            d = g.encode(s); obs[gi][s] = d; slot = slot0 + gi * 2 + s
            A["tiles"][slot] = d["tiles"]; A["units"][slot] = d["units"]; A["items"][slot] = d["items"]; A["glob"][slot] = d["glob"]; A["legal"][slot] = d["legal"]
            A["pos"][slot] = d["pos"]; A["money"][slot] = d["money"]; A["step"][slot] = int(d["step"])
    for gi in range(K): publish(gi)
    conn.send(("obs", []))
    while True:
        msg = conn.recv()
        if msg == "quit": break
        finished = []
        for gi in range(K):
            g = games[gi]; acts = []; tp = tape_of[gi]
            for s in (0, 1):
                if tp is not None and tp[0] == s: acts.append(tp[1][g.step_count] if g.step_count < len(tp[1]) else PASS); continue
                slot = slot0 + gi * 2 + s; d_ = obs[gi][s]; pos = d_["pos"]; n_units = int((pos >= 0).sum())
                prices = {p_: int(v) for p_, v in zip(PRODUCTS, d_["prices"])}; seeds = {c: int(v) for c, v in zip(CROPS, d_["seeds"])}
                o = {"farms": [{"hands": [None] * (n_units - 1)}, {"hands": [None] * (n_units - 1)}], "market": {"prices": prices}, "private": {"seeds": seeds}}  # minimal view for the decoders
                ua = []
                for i in range(n_units):
                    u = (int(pos[i]) % 10, int(pos[i]) // 10)
                    d = int(A["act_dest"][slot, i]); tx, ty = d % 10, d // 10; p = (int(u[0]), int(u[1]))
                    if p != (tx, ty): ua.append(step_toward(p, (tx, ty))); continue
                    name = OPS[int(A["act_op"][slot, i])]; q = unbucket(int(A["act_qty"][slot, i]), QTY_BUCKETS)
                    if name.startswith("PLANT_"): ua.append(["PLANT", name[6:]])
                    elif name.startswith("PICKUP_"): ua.append(["PICKUP", name[7:], int(q)])
                    elif name.startswith("PLACE_"): ua.append(["PLACE", name[6:], int(q)] if name[6:] in PRODUCTS else ["PLACE", name[6:]])
                    else: ua.append([name])
                trim_plants(ua, o["private"]["seeds"])
                a = decode_action(np.zeros(MAX_UNITS, dtype=int), np.zeros(MAX_UNITS, dtype=int), A["act_mkt"][slot].astype(int), o, s)
                a["farmer"] = ua[0] if ua else ["PASS"]; a["hands"] = ua[1:]; acts.append(a)
            g.step(acts[0], acts[1])
            if g.done:
                finished.append((gi, float(g.reward(0)), float(g.reward(1)), int(A["step"][slot0 + gi * 2]), -1 if tp is None else tp[0]))
                games[gi] = new_game(gi)
            publish(gi)
        conn.send(("obs", finished))
    buf.close()


class VecEnv:
    """games = W*K. Slot index = game*2 + seat. After construction and after each step(), buffers hold fresh observations."""
    def __init__(self, workers=10, games_per_worker=24, seed0=100000, tapes_path=None, tape_frac=0.5, max_team_share=0.3):
        self.W, self.K = workers, games_per_worker; self.G = workers * games_per_worker; self.n = self.G * 2
        self.buf = Buffers(self.n, create=True); self.A = self.buf.arr; self.procs = []; self.conns = []
        tapes = []
        if tapes_path:
            tapes = pickle.load(open(tapes_path, "rb")); rnd = random.Random(0); rnd.shuffle(tapes)
            from collections import Counter
            cap = int(max_team_share * len(tapes)); cnt = Counter(); kept = []
            for t in tapes:
                if cnt[t["team"]] < cap: kept.append(t); cnt[t["team"]] += 1
            tapes = kept
        self.n_tapes = len(tapes)
        ctx = mp.get_context("fork")  # tapes are shared copy-on-write with the workers
        for w in range(workers):
            a, b = ctx.Pipe(); p = ctx.Process(target=_worker, args=(w, self.buf.names, self.n, w * games_per_worker * 2, games_per_worker, seed0 + w * 1_000_000, b, tapes, tape_frac), daemon=True)
            p.start(); self.procs.append(p); self.conns.append(a)
        for c in self.conns: c.recv()

    def step(self):
        """Actions must already be written into act_* buffers. Returns list of (game_index, reward_seat0, reward_seat1, last_step, tape_seat or -1)."""
        for c in self.conns: c.send("step")
        finished = []
        for w, c in enumerate(self.conns):
            _, fin = c.recv()
            finished += [(w * self.K + gi, r0, r1, st, ts) for gi, r0, r1, st, ts in fin]
        return finished

    def close(self):
        for c in self.conns:
            try: c.send("quit")
            except Exception: pass
        for p in self.procs: p.join(timeout=5)
        self.buf.close(unlink=True)
