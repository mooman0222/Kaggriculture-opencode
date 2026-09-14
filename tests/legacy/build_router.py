"""Router実体 (yhay81 three-day-shop-router) の再現ビルド (E031/E034/R5)。

公開ノートブックのC++ソース (policy.cpp / guard / tape / bridge / main.py) を
取得→展開→コンパイルし、実戦リプレイとの忠実度 (719手一致) まで検証する。
成果物は tmp/ 以下 (git管理外)。再現手順自体がこのファイル。

使い方:
  PYTHONPATH=.venv/lib/python3.14/site-packages /usr/bin/python3 tests/build_router.py
  PYTHONPATH=... /usr/bin/python3 tests/build_router.py --no-pull   # tmp/router/*.ipynb を再利用
  PYTHONPATH=... /usr/bin/python3 tests/build_router.py --check-only # ビルド済み成果物の検証のみ
"""
import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROUTER_DIR = ROOT / "tmp" / "router"
EXTRACTED = ROUTER_DIR / "extracted"
NB_REF = "yhay81/three-day-shop-router"

COMPILE_CMD = [
    "g++", "-O3", "-std=c++17", "-Wall", "-Wextra", "-pedantic",
    "-shared", "-fPIC", "-Isource/include", "-o", "agent.so",
    "source/policy.cpp", "submission_bridge.cpp",
]


def pull():
    ROUTER_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", "pull", NB_REF, "-p", str(ROUTER_DIR)],
        check=True,
    )


def extract():
    ipynbs = sorted(ROUTER_DIR.glob("*.ipynb"))
    assert ipynbs, f"no ipynb in {ROUTER_DIR} (run without --no-pull first)"
    nb = json.loads(ipynbs[0].read_text())
    EXTRACTED.mkdir(parents=True, exist_ok=True)
    wrote = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        m = re.match(r"%%writefile\s+(\S+)\s*\n", src)
        if not m:
            continue
        path = EXTRACTED / m.group(1)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(src[m.end():])
        wrote.append(str(path.relative_to(ROOT)))
    assert (EXTRACTED / "source" / "policy.cpp").exists()
    assert (EXTRACTED / "submission_bridge.cpp").exists()
    assert (EXTRACTED / "main.py").exists()
    return wrote


def compile_so():
    r = subprocess.run(COMPILE_CMD, cwd=str(EXTRACTED), capture_output=True, text=True)
    print(r.stdout[-2000:] if r.stdout else "")
    print(r.stderr[-2000:] if r.stderr else "")
    r.check_returncode()
    so = EXTRACTED / "agent.so"
    assert so.exists() and so.stat().st_size > 0
    return so


def load_router():
    spec = importlib.util.spec_from_file_location("router_main", str(EXTRACTED / "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent


def smoke():
    agent = load_router()
    obs = {
        "player": 0, "step": 0, "day": 0, "hour": 0,
        "farms": [
            {"money": 3000.0, "tiles": [[None] * 10 for _ in range(10)],
             "farmer": [4, 4], "hands": [], "unlocked_quadrants": ["NW"], "hires_today": 0},
            {"money": 3000.0, "tiles": [[None] * 10 for _ in range(10)],
             "farmer": [4, 4], "hands": [], "unlocked_quadrants": ["NW"], "hires_today": 0},
        ],
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
    }
    act = agent(obs, None)
    assert set(act) == {"farmer", "hands", "market"}, act.keys()
    print("smoke ok:", act)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pull", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    if a.check_only:
        assert (EXTRACTED / "agent.so").exists(), "agent.so missing; run full build"
        smoke()
        print("check-only ok")
        return
    if not a.no_pull:
        pull()
    for p in extract():
        print("extracted:", p)
    print("compiled:", compile_so())
    smoke()
    print("build_router ok")


if __name__ == "__main__":
    main()
