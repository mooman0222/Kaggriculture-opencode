"""使い方: .venv/bin/python tests/team_blueprint.py DIR TEAMNAME
チームの記録から日別の平均スケジュール (雇用・植栽・収穫・水・土地・畑タイル数・現金) とユニット行動の内訳を出す。"""
import json,glob,sys
from collections import defaultdict
PASS={"farmer":["PASS"],"hands":[],"market":[]}
MOVES={"NORTH","SOUTH","EAST","WEST"}
d,team=sys.argv[1],sys.argv[2]
day=defaultdict(lambda: defaultdict(float)); n=0; unitacts=defaultdict(int); final=defaultdict(list)
for f in sorted(glob.glob(f"{d}/episode-*.json")):
    r=json.load(open(f)); names=r["info"]["TeamNames"]; steps=r["steps"]
    if team not in names or len(steps)<720: continue
    me=names.index(team); n+=1
    for t in range(1,720):
        a=steps[t][me].get("action") or PASS; obs=steps[t-1][0]["observation"]; farm=obs["farms"][me]; D=(t-1)//24; h=(t-1)%24
        acts=[a.get("farmer") or ["PASS"],*(a.get("hands") or [])]
        for x in acts:
            k=x[0] if x else "PASS"; unitacts["MOVE" if k in MOVES else k]+=1
            if k=="PLANT" and len(x)>1: day[D]["plant_"+x[1][:3]]+=1
            elif k in ("HARVEST","WATER","FEED","FERTILIZE","DIG"): day[D][k.lower()]+=1
        for o in a.get("market") or []:
            if o and o[0]=="HIRE": day[D]["hire"]+=1
            if o and o[0]=="BUY_ANIMAL": day[D]["buy_"+o[1][:3]]+=int(o[2])
        if h==0:
            day[D]["hands_h0"]+=len(farm["hands"]); day[D]["money"]+=farm["money"]; day[D]["quads"]+=len(farm.get("unlocked_quadrants",[]))
            c=defaultdict(int)
            for row in farm["tiles"]:
                for tl in row:
                    if isinstance(tl,dict) and tl.get("kind")=="PLANT": c["tiles_"+tl.get("crop","?")[:3]]+=1
                    if isinstance(tl,dict) and "animal" in tl: c["anim"]+=1
            for k,v in c.items(): day[D][k]+=v
        if h==12: day[D]["hands_h12"]+=len(farm["hands"])
    last=steps[-1][0]["observation"]["farms"][me]; final["bank"].append(r["rewards"][me])
print(f"{team}: n={n} games, final bank mean {sum(final['bank'])/n:.0f}")
tot=sum(unitacts.values()); print("unit-step mix:", {k:f"{v/tot:.0%}" for k,v in sorted(unitacts.items(),key=lambda x:-x[1])})
cols=["hands_h0","hands_h12","hire","quads","money","tiles_WHE","tiles_CAR","tiles_STR","tiles_MEL","tiles_TOM","anim","plant_WHE","plant_CAR","plant_STR","plant_MEL","plant_TOM","harvest","water","feed","fertilize","buy_COW","buy_SHE","buy_GOO"]
print("day "+" ".join(f"{c[:9]:>9s}" for c in cols))
for D in range(30):
    print(f"d{D:02d} "+" ".join(f"{day[D][c]/n:9.1f}" for c in cols))
