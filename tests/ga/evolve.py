"""(1+λ) 山登り: 13本テープに変異を加え、プール適応度 (系統バランス平均 margin) が上がれば採用。

使い方: .venv/bin/python tests/ga/evolve.py --init tmp/ga/tapes_guarded.json --out tmp/ga/best.json --iters 400 --lam 16
"""
import argparse, copy, json, random, sys, time
sys.path.insert(0, "tests/ga"); from fitness import Fitness
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}


def sell_orders(tape):
    return [(t, i) for t, a in enumerate(tape) for i, o in enumerate(a.get("market") or []) if o and o[0] == "SELL" and int(o[2]) < 900]


def mut_sell_shift(tape, rng):
    so = sell_orders(tape)
    if not so: return False
    t, i = rng.choice(so); o = tape[t]["market"][i]; d = rng.choice([-3, -2, -1, 1, 2])
    nt = t + d
    if not (2 <= nt < 718) or len(tape[nt]["market"]) >= 10: return False
    tape[t]["market"][i] = []  # keep the slot
    tape[nt]["market"].append(list(o)); return True


def mut_sell_qty(tape, rng):
    so = sell_orders(tape)
    if not so: return False
    t, i = rng.choice(so); o = tape[t]["market"][i]
    o[2] = max(1, int(o[2]) + rng.choice([-3, -2, -1, 1, 2, 3])); return True


def mut_sell_split(tape, rng):
    so = [(t, i) for t, i in sell_orders(tape) if int(tape[t]["market"][i][2]) >= 4]
    if not so: return False
    t, i = rng.choice(so); o = tape[t]["market"][i]; q = int(o[2]); k = rng.randint(1, q - 1)
    nt = t + rng.choice([-2, -1, 1, 2])
    if not (2 <= nt < 718) or len(tape[nt]["market"]) >= 10: return False
    o[2] = q - k; tape[nt]["market"].append(["SELL", o[1], k]); return True


def mut_crop_swap(tape, rng):
    """PLANT X -> PLANT Y on one unit action, funding it by retargeting the nearest earlier BUY_SEED X."""
    plants = [(t, u) for t, a in enumerate(tape) for u, act in enumerate([a.get("farmer") or [], *(a.get("hands") or [])]) if act and act[0] == "PLANT"]
    if not plants: return False
    t, u = rng.choice(plants)
    acts = [tape[t].get("farmer") or [], *(tape[t].get("hands") or [])]
    old = acts[u][1]; new = rng.choice([c for c in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY") if c != old])
    # find an earlier BUY_SEED of old crop
    for tb in range(t, max(-1, t - 48), -1):
        for o in tape[tb].get("market") or []:
            if o and o[0] == "BUY_SEED" and o[1] == old and int(o[2]) >= 1:
                if int(o[2]) == 1: o[1] = new
                else:
                    o[2] = int(o[2]) - 1
                    if len(tape[tb]["market"]) >= 10: return False
                    tape[tb]["market"].append(["BUY_SEED", new, 1])
                acts[u] = ["PLANT", new]
                tape[t]["farmer"] = acts[0]; tape[t]["hands"] = acts[1:]
                return True
    return False


def mut_sell_delete(tape, rng):
    """Drop one SELL: the stock is liquidated at the end instead (the GA decides if holding pays)."""
    so = sell_orders(tape)
    if not so: return False
    t, i = rng.choice(so); tape[t]["market"][i] = []; return True


_TRACE = {}


def unit_trace(tape):
    """Per-step unit positions/inventories/tiles for this tape in a mirror game (kagsim L1). Cached by id."""
    import kagsim
    key = id(tape)
    if key in _TRACE: return _TRACE[key]
    g = kagsim.Game(4242); out = []
    while not g.done:
        o = g.observe(0); farm = o["farms"][0]; invs = o["private"]["inventories"]
        units = [farm["farmer"], *farm["hands"]]
        out.append([(tuple(pos), invs[i] if i < len(invs) else {}, farm["tiles"][pos[1]][pos[0]]) for i, pos in enumerate(units)])
        t = g.step_count; g.step(tape[t], tape[t])
    _TRACE.clear(); _TRACE[key] = out
    return out


def mut_fertilize(tape, rng):
    """A unit carrying FERTILIZER on a PLANT tile: replace its PASS/WATER with FERTILIZE."""
    tr = unit_trace(tape)
    cands = []
    for t in range(48, 700):
        acts = [tape[t].get("farmer") or ["PASS"], *(tape[t].get("hands") or [])]
        for u, (pos, inv, tile) in enumerate(tr[t]):
            if u >= len(acts): break
            if inv.get("FERTILIZER", 0) > 0 and isinstance(tile, dict) and tile.get("kind") == "PLANT" and acts[u][0] in ("PASS", "WATER") \
                    and tile.get("fertilized_until_day", -1) < t // 24:
                cands.append((t, u))
    if not cands: return False
    t, u = rng.choice(cands)
    acts = [tape[t].get("farmer") or ["PASS"], *(tape[t].get("hands") or [])]
    acts[u] = ["FERTILIZE"]; tape[t]["farmer"] = acts[0]; tape[t]["hands"] = acts[1:]; return True


def mut_buy_earlier(tape, rng):
    """Move a BUY_ANIMAL / BUY_LAND / BUY_SEED order 1..8 turns earlier (cash permitting; the engine refuses silently)."""
    buys = [(t, i) for t, a in enumerate(tape) for i, o in enumerate(a.get("market") or []) if o and o[0] in ("BUY_ANIMAL", "BUY_LAND", "BUY_SEED") and t > 8]
    if not buys: return False
    t, i = rng.choice(buys); o = tape[t]["market"][i]; nt = t - rng.randint(1, 8)
    if len(tape[nt]["market"]) >= 10: return False
    tape[t]["market"][i] = []; tape[nt]["market"].append(list(o)); return True


_WORLD_STREAMS = None


def world_streams():
    """Recorded SR0909-family streams grouped by the first two shops (splice donors for the same world)."""
    global _WORLD_STREAMS
    if _WORLD_STREAMS is None:
        import json
        from collections import defaultdict
        _WORLD_STREAMS = defaultdict(list)
        for p in json.load(open("tmp/ga/pool.json")):
            if p["lineage"] == "SR0909": _WORLD_STREAMS[tuple(p["shops"][:2])].append(p["stream"])
    return _WORLD_STREAMS


def mut_world_splice(tapes, rng):
    """Replace one plan's continuation (from a 72-step boundary) with a donor recorded in the same world."""
    import sys
    sys.path.insert(0, "tests/ga"); from fitness import SHOP_PLANS
    ws = world_streams()
    worlds = [w for w in ws if w in SHOP_PLANS or True]
    if not worlds: return False
    w = rng.choice(worlds); plan = SHOP_PLANS.get(w, 0); donor = rng.choice(ws[w])
    cut = rng.choice([144, 216, 288, 360, 432])
    tapes[plan] = copy.deepcopy(tapes[plan][:cut]) + copy.deepcopy(donor[cut:648]) + copy.deepcopy(tapes[plan][648:])
    return True


OPS = [mut_sell_shift, mut_sell_qty, mut_sell_split, mut_crop_swap, mut_sell_delete, mut_fertilize, mut_buy_earlier]


def mutate(tapes, rng, n_ops):
    child = list(tapes)  # copy-on-write: deep-copy only the tapes we touch
    touched = set()
    for _ in range(n_ops):
        if rng.random() < 0.08:
            if 0 not in touched: child[0] = copy.deepcopy(child[0]); touched.add(0)
            mut_world_splice(child, rng); touched.update(range(len(child))); continue
        op = rng.choice(OPS)
        if rng.random() < 0.3:  # shared opening: mutate tape 0 before step 144 and mirror it into every plan
            if 0 not in touched: child[0] = copy.deepcopy(child[0]); touched.add(0)
            before = [copy.deepcopy(a) for a in child[0][:144]]
            if op(child[0], rng):
                for k in range(1, len(child)):
                    if k not in touched: child[k] = copy.deepcopy(child[k]); touched.add(k)
                    child[k][:144] = copy.deepcopy(child[0][:144])
        else:
            k = rng.randrange(len(child))
            if k not in touched: child[k] = copy.deepcopy(child[k]); touched.add(k)
            op(child[k], rng)
    return child


def wscore(w):
    return 0.6 * w["sr0909_base"]["mean"] + 0.4 * w["sr0909_live"]["mean"] + 2000 * (0.6 * w["sr0909_base"]["wr"] + 0.4 * w["sr0909_live"]["wr"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--init", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=200); ap.add_argument("--lam", type=int, default=16); ap.add_argument("--ops", type=int, default=2); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=2, help="L0 上位いくつをライブ層付き L1 で確認するか")
    a = ap.parse_args(); rng = random.Random(a.seed)
    sys.path.insert(0, "tests/ga"); from wrapped import evaluate_wrapped
    f = Fitness(split="train"); fh = Fitness(split="holdout")
    best = json.load(open(a.init)); bw = evaluate_wrapped(best); bws = wscore(bw)
    fmtw = lambda w: f"base {w['sr0909_base']['mean']:+.0f}/{w['sr0909_base']['wr']:.2f} e055 {w['sr0909_live']['mean']:+.0f}/{w['sr0909_live']['wr']:.2f}"
    print(f"init wrapped {fmtw(bw)} score {bws:+.0f}", flush=True)
    t0 = time.time(); acc = 0
    for it in range(a.iters):
        cands = [mutate(best, rng, a.ops) for _ in range(a.lam)]
        scored = sorted(((f.evaluate(c)["score"], i) for i, c in enumerate(cands)), reverse=True)
        for s0, i in scored[: a.top]:
            w = evaluate_wrapped(cands[i]); ws = wscore(w)
            if ws > bws + 25:
                best, bws, bw = cands[i], ws, w; acc += 1
                hold = evaluate_wrapped(best, seeds=range(2100, 2112))
                print(f"it {it} ACCEPT wrapped {fmtw(w)} score {bws:+.0f} | holdout {fmtw(hold)} [{time.time()-t0:.0f}s]", flush=True)
                json.dump(best, open(a.out, "w")); break
        if it % 20 == 0: print(f"it {it} best {bws:+.0f} [{time.time()-t0:.0f}s]", flush=True)
    print("accepted", acc)


if __name__ == "__main__":
    main()
