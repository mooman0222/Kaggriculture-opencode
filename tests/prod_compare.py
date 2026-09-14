"""同シードで 2 エージェントを v41 相手に走らせ、from_day 以降の日別「収穫ユニット数 (品目別)」「給水・施肥・給餌」を並べる。
使い方: .venv/bin/python tests/prod_compare.py agents/e058/main.py agents/e060/main.py [--seeds 5000,5003] [--from 18]"""
import sys,os,importlib.util,hashlib,kagsim
from collections import defaultdict
def load(p):
    d=os.path.dirname(os.path.abspath(p)); sys.path.insert(0,d)
    s=importlib.util.spec_from_file_location("c_"+hashlib.md5((p+str(os.environ.get('LE_SWITCH'))).encode()).hexdigest()[:8],p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
PROD={"COW":"MILK","SHEEP":"WOOL","GOOSE":"EGG"}
def run(agent_path, seed, from_day=18):
    me=load(agent_path); opp=load("third_party/public_agents/v41/main.py")
    g=kagsim.Game(seed); harv=defaultdict(lambda: defaultdict(int)); acts=defaultdict(lambda: defaultdict(int)); shed_sold=defaultdict(int)
    prev_shed=None
    while not g.done:
        t=g.step_count; o=g.observe(0); farm=o["farms"][0]; D=t//24
        x=me.agent(o); y=opp.agent(g.observe(1))
        units=[farm["farmer"],*farm["hands"]]; al=[x.get("farmer") or ["PASS"],*(x.get("hands") or [])]
        for pos,a in zip(units,al):
            k=a[0] if a else "PASS"; tl=farm["tiles"][pos[1]][pos[0]]
            if k=="HARVEST" and isinstance(tl,dict):
                item=PROD.get(tl.get("animal")) if "animal" in tl else tl.get("crop")
                harv[D][item]+=int(tl.get("yield_units",0))
            if k in ("WATER","FERTILIZE","FEED","CARE","PLANT","DIG","PASS"): acts[D][k]+=1
            elif k in ("NORTH","SOUTH","EAST","WEST"): acts[D]["MOVE"]+=1
        g.step(x,y)
    tel=g.telemetry(0)
    return g.reward(0), g.reward(1), harv, acts, tel
import argparse
_ap=argparse.ArgumentParser(); _ap.add_argument("a"); _ap.add_argument("b"); _ap.add_argument("--seeds",default="5000,5003"); _ap.add_argument("--from",dest="from_day",type=int,default=18); A=_ap.parse_args()
for seed in [int(x) for x in A.seeds.split(",")]:
    print(f"########## seed {seed}")
    res={}
    for name,path in ((os.path.basename(os.path.dirname(A.a)),A.a),(os.path.basename(os.path.dirname(A.b)),A.b)):
        r0,r1,harv,acts,tel=run(path,seed,A.from_day); res[name]=(harv,acts)
        tot=defaultdict(int)
        for D in range(A.from_day,30):
            for k,v in harv[D].items(): tot[k]+=v
        print(f"{name}: final {r0:.0f} vs {r1:.0f} | sold_units {tel.get('sold_units')} revenue {tel.get('sell_revenue')} discarded {tel.get('shed_discarded_units',0)} | harvested from day: {dict(sorted(tot.items(),key=lambda x:-x[1]))}")
        for D in range(A.from_day,30):
            a=acts[D]; print(f"   d{D} harv {dict(harv[D])} water {a['WATER']} fert {a['FERTILIZE']} feed {a['FEED']} care {a['CARE']} plant {a['PLANT']} dig {a['DIG']} pass {a['PASS']} move {a['MOVE']}")
