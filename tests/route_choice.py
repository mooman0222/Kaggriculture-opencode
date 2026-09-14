"""64 の最初2店ペアそれぞれで、既定 (0909) プラン vs 新家系のプランを v41 相手に測り、ペア別に良い方を選ぶ (E060 の手順)。
使い方: .venv/bin/python tests/route_choice.py [--agent agents/e060/main.py] [--tapes .opencode/data/shop_router_0913_tapes.json] [--offset 13] [--out tmp/route_choice.json]
新家系のテープは agents/<X>/actions.json に既定 13 本の後ろ (index offset〜) に追記されている前提。"""
import json,sys,os,random,importlib.util,hashlib,re,kagsim
from concurrent.futures import ProcessPoolExecutor
SHOPS=sorted(["BAKERY","PIZZA_SHOP","BRUNCH_SPOT","YARN_STORE","ICE_CREAM_SHOP","PET_CAFE","SMOOTHIE_SHOP","FARMERS_MARKET"])
V41="third_party/public_agents/v41/main.py"
import argparse
_ap=argparse.ArgumentParser(); _ap.add_argument("--agent",default="agents/e060/main.py"); _ap.add_argument("--tapes",default=".opencode/data/shop_router_0913_tapes.json"); _ap.add_argument("--offset",type=int,default=13); _ap.add_argument("--out",default="tmp/route_choice.json"); _ap.add_argument("--seeds",default="7100,7101,7102,7103")
ARGS=_ap.parse_args() if __name__=="__main__" else _ap.parse_args([])
YARN={('BAKERY','YARN_STORE'):3,('BRUNCH_SPOT','YARN_STORE'):4,('FARMERS_MARKET','YARN_STORE'):5,('ICE_CREAM_SHOP','YARN_STORE'):6,('PET_CAFE','YARN_STORE'):5,('PIZZA_SHOP','YARN_STORE'):7,('SMOOTHIE_SHOP','YARN_STORE'):8,('YARN_STORE','BAKERY'):9,('YARN_STORE','BRUNCH_SPOT'):9,('YARN_STORE','FARMERS_MARKET'):1,('YARN_STORE','ICE_CREAM_SHOP'):9,('YARN_STORE','PET_CAFE'):10,('YARN_STORE','PIZZA_SHOP'):6,('YARN_STORE','SMOOTHIE_SHOP'):11,('YARN_STORE','YARN_STORE'):12}
D=json.load(open(ARGS.tapes)); T13={tuple(r["shops"]):ARGS.offset+r["plan"] for r in D["routes"]}
def load(p,tag):
    d=os.path.dirname(os.path.abspath(p)); sys.path.insert(0,d)
    s=importlib.util.spec_from_file_location("c_"+hashlib.md5((p+tag).encode()).hexdigest()[:8],p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def job(args):
    pair, seeds = args
    me=load(ARGS.agent,"x"); opp=load(V41,"o"); res={}
    for label,plan in (("0909",YARN.get(pair,0)),("0913",T13[pair])):
        out=[]
        for seed in seeds:
            shops=list(pair)+random.Random(seed).choices(SHOPS,k=6)
            for seat in (0,1):
                me._LIVE=None; g=kagsim.Game(seed,720,shops)
                while not g.done:
                    if g.step_count==144:
                        ch=me._LIVE.b._IMPL.chassis; st=ch.players.get(seat) or ch._state(seat,144)
                        st["router_state"]["route"]=plan; st["router_state"]["day6"]=True
                    x=me.agent(g.observe(seat)); y=opp.agent(g.observe(1-seat)); g.step(*((x,y) if seat==0 else (y,x)))
                out.append((g.reward(seat),g.reward(1-seat)))
        res[label]=out
    return pair,res
if __name__=="__main__":
    seeds=[int(x) for x in ARGS.seeds.split(",")]; pairs=[(a,b) for a in SHOPS for b in SHOPS]
    with ProcessPoolExecutor(10) as ex: res=list(ex.map(job,[(p,seeds) for p in pairs]))
    table={}; gain=0
    for pair,r in res:
        m={k:sum(a-b for a,b in v)/len(v) for k,v in r.items()}; o={k:sum(a for a,b in v)/len(v) for k,v in r.items()}
        pick="0913" if m["0913"]-m["0909"]>1000 else "0909"
        table[",".join(pair)]={"pick":pick,"m0909":m["0909"],"m0913":m["0913"]}
        if pick=="0913": gain+=m["0913"]-m["0909"]
        print(f"{pair[0][:3]},{pair[1][:3]} 0909 {m['0909']:+7.0f} (own {o['0909']:6.0f}) 0913 {m['0913']:+7.0f} (own {o['0913']:6.0f}) -> {pick}")
    json.dump(table,open(ARGS.out,"w"),indent=0)
    print("pairs picking 0913:",sum(1 for v in table.values() if v["pick"]=="0913"),"avg gain over all 64 pairs:",round(gain/64))
