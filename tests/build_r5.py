"""R5提出パッケージ (Router農場+移植市場層) の再現組立 (E041)。

入力:
  - tmp/router/extracted/main.py + agent.so (tests/build_router.py の成果物)
  - agents/kaito_v56_e039phase.py (E039、層の供給元)
出力:
  - tmp/sub/main.py + tmp/sub/agent.so + tmp/r5_submit.tar.gz
検証: main.py の末尾callableが agent であること + 期待sha256との一致。

使い方:
  PYTHONPATH=.venv/lib/python3.14/site-packages /usr/bin/python3 tests/build_r5.py
"""
import hashlib
import importlib.util
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTRACTED = ROOT / "tmp" / "router" / "extracted"
E039 = ROOT / "agents" / "kaito_v56_e039phase.py"
SUB = ROOT / "tmp" / "sub"

# tmp/sub/main.py の期待sha256 (層・ブリッジ変更時は意図的に更新する)
EXPECTED_MAIN_SHA = "837cf59c058d78e878a584d35a41b0857657ae21efff04a6010b76349d2ab414"

_R_STEP0_WHEAT = 13  # Router開幕のstep0小麦買い (テープn13判定用)


_G_HELPER = '''def _g(v, k, d=None):
    if isinstance(v, dict):
        return v.get(k, d)
    if hasattr(v, "get"):
        return v.get(k, d)
    return getattr(v, k, d)


'''


def _grab_dict(lines, idx):
    out = [lines[idx]]
    depth = lines[idx].count("{") - lines[idx].count("}")
    j = idx
    while depth > 0:
        j += 1
        out.append(lines[j])
        depth += lines[j].count("{") - lines[j].count("}")
    return "".join(out)


def _find(lines, pref, start=0):
    return next(i for i, line in enumerate(lines) if i >= start and line.startswith(pref))


def build_main():
    bridge = (EXTRACTED / "main.py").read_text()
    bridge = bridge.replace(
        "def agent(observation, configuration=None):",
        "def _ROUTER_FARM(observation, configuration=None):",
        1,
    )
    src = E039.read_text().splitlines(keepends=True)

    # eager層: 定数 + _eager_sell (E035a/E039)
    e0 = _find(src, "_EAGER_ITEMS")
    e1 = _find(src, "# ===== E037")
    eager = "".join(src[e0:e1]).replace("_sweep_get", "_g")

    # sweep層 (E022a+E028)
    s0 = _find(src, "# ===== E022a")
    s1 = _find(src, "_pre_sweep_agent")
    sweep = "".join(src[s0:s1]).replace("_sweep_get", "_g")

    # tape層 (E031-E034): テープ表 + 検出 + 先回り。_E030_N はRouter開幕13に置換
    t0 = _find(src, "# ===== E031")
    t1 = _find(src, "def agent(", t0)
    tape = "".join(src[t0:t1]).replace("_orak_get", "_g").replace("_E030_N", "_R_STEP0_WHEAT")
    shops = _grab_dict(src, _find(src, "_ORAK_SHOPS"))
    g0 = _find(src, "def _orak_get")
    g1 = _find(src, "def _orak_phantom")
    h0 = _find(src, "def _orak_town_demand")
    h1 = _find(src, "def _orak_front_run")
    h2 = _find(src, "_pre_orak_agent")
    helpers = "".join(src[g0:g1] + src[h0:h2]).replace("_orak_get", "_g")
    tape = f"_R_STEP0_WHEAT = {_R_STEP0_WHEAT}  # Router opener step0 wheat buy\n" + shops + helpers + tape
    tape += '''
def _r_tape(act, obs):
    _tape_observe(obs)
    if _tape_state.get("mode") in ("tape", "tape2"):
        try:
            preds = _tape_preds(int(_g(obs, "step", 0) or 0))
            if preds:
                act = _orak_front_run(dict(act), obs, preds)
        except Exception:
            pass
    return act

'''

    # carrot層 (E037)
    c0 = _find(src, "# ===== E037")
    c1 = _find(src, "def agent_entry")
    carrot = "".join(src[c0:c1]).replace("_sweep_get", "_g")
    final = '''
def agent(obs, configuration=None):
    """Kaggle entrypoint: Router farm+market with ported alpha layers."""
    try:
        base = _ROUTER_FARM(obs, configuration)
    except Exception:
        return {"farmer": ["PASS"], "hands": [], "market": []}
    try:
        act = dict(base)
        _tape_observe(obs)
        if _tape_state.get("mode") in ("tape", "tape2"):
            try:
                preds = _tape_preds(int(_g(obs, "step", 0) or 0))
                if preds:
                    act = _orak_front_run(dict(act), obs, preds)
            except Exception:
                pass
        act = _sweep(dict(act), obs)
        act = _carrot_swap(dict(act), obs)
        act = _carrot_seeds(act, obs)
        return _eager_sell(dict(act), obs)
    except Exception:
        return base
'''
    return bridge.rstrip() + "\n\n" + _G_HELPER + eager + sweep + tape + carrot + final


def verify(main_text):
    env = {}
    exec(compile(main_text, "main.py", "exec"), env)
    last = [k for k, v in env.items() if callable(v)][-1]
    assert last == "agent", f"last callable is {last}, want agent"
    sha = hashlib.sha256(main_text.encode()).hexdigest()
    if EXPECTED_MAIN_SHA != "PLACEHOLDER":
        assert sha == EXPECTED_MAIN_SHA, f"sha mismatch: {sha}"
    return sha


def main():
    assert (EXTRACTED / "agent.so").exists(), "run tests/build_router.py first"
    assert E039.exists(), f"{E039} missing"
    SUB.mkdir(parents=True, exist_ok=True)
    main_text = build_main()
    (SUB / "main.py").write_text(main_text)
    (SUB / "agent.so").write_bytes((EXTRACTED / "agent.so").read_bytes())
    sha = verify(main_text)
    print("main.py sha256:", sha)
    tarball = ROOT / "tmp" / "r5_submit.tar.gz"
    with tarfile.open(tarball, "w:gz") as bundle:
        bundle.add(SUB / "main.py", arcname="main.py")
        bundle.add(SUB / "agent.so", arcname="agent.so")
    print("tarball:", tarball, tarball.stat().st_size, "bytes")
    # ロード smoke (zoo ハーネスと同一条件)
    spec = importlib.util.spec_from_file_location("r5sub", str(SUB / "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(getattr(mod, "agent", None))
    print("build_r5 ok")


if __name__ == "__main__":
    main()
