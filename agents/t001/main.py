# Tape router: replays recorded top-agent action tapes, switching to the tape whose
# revealed shop prefix matches the town and whose farm plan agrees longest with our own history.
import base64, copy, json, os, zlib

_BLOB = None  # embedded library (zlib + b85), filled in by the packer
_LIB = None
_STATE = {}
_EXCLUDE = set()  # episode ids to leave out (leave-one-out evaluation only)
PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def _load():
    global _LIB
    if _LIB is None:
        path = os.environ.get("TR_LIB")
        raw = open(path, "rb").read() if path else zlib.decompress(base64.b85decode(_BLOB))
        d = json.loads(raw)
        plan_key = {}
        plan = []
        for a in d["actions"]:
            k = json.dumps([a.get("farmer"), a.get("hands")], separators=(",", ":"))
            plan.append(plan_key.setdefault(k, len(plan_key)))
        for tp in d["tapes"]:
            tp["plan"] = [plan[i] for i in tp["ids"]]
        _LIB = dict(actions=d["actions"], tapes=d["tapes"])
    return _LIB


def _prefix(a, b, n):
    for t in range(n):
        if a[t] != b[t]:
            return t
    return n


def _choose(seat, revealed, hist, t):
    tapes = [tp for tp in _load()["tapes"] if tp["s"] == seat and tp["eid"] not in _EXCLUDE]
    if not tapes:
        return None
    k = len(revealed)
    while k >= 0:
        cands = [tp for tp in tapes if tp["shops"][:k] == revealed[:k]]
        if cands:
            break
        k -= 1
    if t == 0:  # most central opening: shares the longest day-0..2 plan with the other tapes
        return max(cands, key=lambda tp: (sum(_prefix(tp["plan"], o["plan"], 72) for o in cands), tp["own"]))
    return max(cands, key=lambda tp: (_prefix(tp["plan"], hist, t), tp["own"]))


def agent(observation, configuration=None):
    try:
        step = int(observation["day"]) * 24 + int(observation["hour"])
        seat = int(observation["player"])
        st = _STATE.get(seat)
        if st is None or step == 0 or step <= st["last"]:
            st = _STATE[seat] = dict(last=-1, tape=None, hist=[])
        st["last"] = step
        revealed = list((observation.get("town") or {}).get("unlocked_shops") or [])
        tp = st["tape"]
        if tp is None or tp["shops"][:len(revealed)] != revealed:
            nxt = _choose(seat, revealed, st["hist"], step)
            if nxt is not None:
                st["tape"] = tp = nxt
        if tp is None or step >= len(tp["ids"]):
            return copy.deepcopy(PASS)
        st["hist"].append(tp["plan"][step])
        return copy.deepcopy(_load()["actions"][tp["ids"][step]])
    except Exception:
        return copy.deepcopy(PASS)
