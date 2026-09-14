"""ライブエージェントを 1 試合走らせ、例外を表面化し日別サマリ (現金・雇用・畑・家畜・注文拒否) を出す。
使い方: LE_DEBUG=1 .venv/bin/python tests/live_debug.py agents/live_e/main.py [--vs OPP] [--seed N] [--shops A,B,...]"""
import argparse, importlib.util, json, os, sys, hashlib, time, traceback
from collections import defaultdict
import kagsim
def load(p):
    d=os.path.dirname(os.path.abspath(p)); sys.path.insert(0,d)
    s=importlib.util.spec_from_file_location("dbg_"+hashlib.md5(p.encode()).hexdigest()[:8],p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
ap=argparse.ArgumentParser(); ap.add_argument("agent"); ap.add_argument("--vs",default="tmp/e058/agents/ahmedberatozer_kaggriculture-v41-review-candidate/main.py"); ap.add_argument("--seed",type=int,default=5000); ap.add_argument("--shops"); ap.add_argument("--seat",type=int,default=0)
a=ap.parse_args(); me=load(a.agent); opp=load(a.vs); seat=a.seat
g=kagsim.Game(a.seed) if not a.shops else kagsim.Game(a.seed,720,a.shops.split(","))
day=defaultdict(lambda: defaultdict(int)); tmax=0; sellq={'me':defaultdict(int),'op':defaultdict(int)}; sellv={'me':defaultdict(float),'op':defaultdict(float)}
while not g.done:
    t=g.step_count; o=g.observe(seat); farm=o["farms"][seat]; D=t//24; h=t%24
    t0=time.perf_counter()
    try: x=me.agent(o)
    except Exception: traceback.print_exc(); print("STEP",t); sys.exit(1)
    tmax=max(tmax,time.perf_counter()-t0)
    y=opp.agent(g.observe(1-seat))
    px=o["market"]["prices"]
    for side,act in (("me",x),("op",y)):
        for q in act.get("market") or []:
            if q and q[0]=="SELL" and len(q)>=3 and int(q[2])<900: sellq[side][q[1]]+=int(q[2]); sellv[side][q[1]]+=int(q[2])*px.get(q[1],0)
    if h==0:
        of=o["farms"][1-seat]; c2=defaultdict(int)
        for row in of["tiles"]:
            for tl in row:
                if isinstance(tl,dict) and tl.get("kind")=="PLANT": c2[tl["crop"][:3]]+=1
                elif isinstance(tl,dict) and "animal" in tl: c2[tl["animal"][:3]]+=1
        day[D]["opp"]=(round(of["money"]),len(of["unlocked_quadrants"]),dict(c2))
    for u in [x.get("farmer") or ["PASS"],*(x.get("hands") or [])]:
        k=u[0] if u else "PASS"; day[D][k if k not in ("NORTH","SOUTH","EAST","WEST") else "MOVE"]+=1
        if k=="PLANT": day[D]["plant_"+u[1][:3]]+=1
    for q in x.get("market") or []:
        if q: day[D]["mk_"+q[0]]+=1
    if h==0:
        c=defaultdict(int)
        for row in farm["tiles"]:
            for tl in row:
                if isinstance(tl,dict) and tl.get("kind")=="PLANT": c[tl["crop"][:3]]+=1
                elif isinstance(tl,dict) and "animal" in tl: c["anim"]+=1
                elif isinstance(tl,dict) and tl.get("kind")=="WEED": c["weed"]+=1
        day[D]["money"]=round(farm["money"]); day[D]["hands"]=len(farm["hands"]); day[D]["quads"]=len(farm["unlocked_quadrants"]); day[D]["tiles"]=dict(c); day[D]["shed"]={k:v for k,v in o["private"]["shed"].items() if v}
    if h==12: day[D]["hands12"]=len(farm["hands"])
    g.step(*((x,y) if seat==0 else (y,x)))
print("final", g.reward(seat), "vs", g.reward(1-seat), "margin", g.reward(seat)-g.reward(1-seat), "max step s", round(tmax,3))
print("telemetry me ", {k:v for k,v in g.telemetry(seat).items() if v})
print("telemetry opp", {k:v for k,v in g.telemetry(1-seat).items() if v})
print("shops", g.observe(0)["town"]["unlocked_shops"])
for side in ("me","op"): print(side,"sell orders:",{k:v for k,v in sorted(sellq[side].items(),key=lambda x:-sellv[side][x[0]])},"value k:",{k:round(v/1000,1) for k,v in sorted(sellv[side].items(),key=lambda x:-x[1])})
for D in range(30):
    d=day[D]; print(f"d{D:02d} $ {d['money']:>6} hands {d['hands']}/{d.get('hands12',0)} q{d['quads']} tiles {d['tiles']} shed {d.get('shed',{})}")
    print("      acts", {k:v for k,v in d.items() if k not in ("money","hands","hands12","quads","tiles","shed")})
