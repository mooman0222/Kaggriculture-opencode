#!/bin/bash
# 候補の定型評価 (使い方: bash tests/live_eval.sh agents/X/main.py): 対 v41・対 E058 (両席 16 戦) と、我々が弱かった世界 (固定ショップ) での own bank
cd /Users/jp17373/workspace/Kaggriculture-opencode
A=${1:-agents/e060/main.py}; V41=third_party/public_agents/v41/main.py
echo "== vs v41"; .venv/bin/python tests/kag_eval.py $A --vs $V41 --games 16 --seed0 5000 2>&1 | grep -v Warn | tail -1
echo "== vs E058"; .venv/bin/python tests/kag_eval.py $A --vs agents/e058/main.py --games 16 --seed0 5000 2>&1 | grep -v Warn | tail -1
echo "== worst worlds (fixed shops, vs v41, own bank)"; .venv/bin/python - $A <<'PY' 2>&1 | grep -v Warn
import json,sys,os,importlib.util,hashlib,kagsim
def load(p):
    d=os.path.dirname(os.path.abspath(p)); sys.path.insert(0,d)
    s=importlib.util.spec_from_file_location("c_"+hashlib.md5(p.encode()).hexdigest()[:8],p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
a=load(sys.argv[1]); b=load("third_party/public_agents/v41/main.py")
rows=json.load(open("tmp/e058/world_gap.json")); rows.sort(key=lambda r:-(r["pair_rec"]-r["pair_ours"]))
own=[]; opp=[]
for r in rows[:8]:
    for seat in (0,1):
        for k in ("_LIVE","_POLICY"):
            if hasattr(a,k): setattr(a,k,None)
        g=kagsim.Game(r["seed"],720,r["shops"])
        while not g.done:
            x=a.agent(g.observe(seat)); y=b.agent(g.observe(1-seat)); g.step(*((x,y) if seat==0 else (y,x)))
        own.append(g.reward(seat)); opp.append(g.reward(1-seat))
    print(f"  {','.join(s[:3] for s in r['shops'][:3]):12s} own {own[-2]:7.0f}/{own[-1]:7.0f} v41 {opp[-2]:7.0f}/{opp[-1]:7.0f}  (E058 own was {r['own']:.0f}, MMPQ {r['mmpq']:.0f})")
print(f"  worst-8 mean own {sum(own)/len(own):.0f} vs v41 {sum(opp)/len(opp):.0f} margin {sum(o-p for o,p in zip(own,opp))/len(own):+.0f}")
PY
