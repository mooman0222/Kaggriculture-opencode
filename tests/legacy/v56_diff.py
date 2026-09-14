"""リプレイ中の指定席の観測列をローカル v56 に食わせ、実際の行動との差分を測る。

「v56 派生か」「どこ (農場/市場) で・いつ (step) から逸脱するか」を判定する。
使い方: .venv/bin/python tests/v56_diff.py REPLAY.json --seat 0 [--agent agents/kaito_v56.py]
"""
import argparse, importlib.util, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def load(p):
    s = importlib.util.spec_from_file_location("m" + str(abs(hash(p))), p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m.agent

def diff(replay, seat, agent_path="agents/kaito_v56.py", verbose=False):
    r = json.load(open(replay)); steps = r["steps"]
    ag = load(agent_path)
    cfg = dict(r.get("configuration") or {})
    farm_mis = market_mis = 0; first_farm = first_market = None; per_day_farm = [0] * 30
    for i in range(len(steps) - 1):
        obs = dict(steps[i][seat]["observation"])
        obs.setdefault("step", i)
        real = steps[i + 1][seat].get("action") or {}
        try:
            pred = ag(obs, cfg)
        except Exception as e:
            pred = {"farmer": ["ERR"], "hands": [], "market": []}
        rf = [real.get("farmer")] + list(real.get("hands") or [])
        pf = [pred.get("farmer")] + list(pred.get("hands") or [])
        if [list(x) if x else x for x in rf] != [list(x) if x else x for x in pf]:
            farm_mis += 1; per_day_farm[i // 24] += 1
            if first_farm is None: first_farm = i
        rm = sorted(json.dumps(o) for o in (real.get("market") or []))
        pm = sorted(json.dumps(o) for o in (pred.get("market") or []))
        if rm != pm:
            market_mis += 1
            if first_market is None: first_market = i
            if verbose and market_mis <= 8:
                print(f"  step {i}: real={real.get('market')} pred={pred.get('market')}")
    n = len(steps) - 1
    return {"farm_mismatch": farm_mis, "market_mismatch": market_mis, "steps": n,
            "first_farm_dev": first_farm, "first_market_dev": first_market, "farm_mis_by_day": per_day_farm}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("replay"); ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--agent", default="agents/kaito_v56.py"); ap.add_argument("-v", action="store_true")
    a = ap.parse_args(); print(json.dumps(diff(a.replay, a.seat, a.agent, a.v)))
