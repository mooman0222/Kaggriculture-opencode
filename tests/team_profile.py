"""チームの記録ゲーム群をプロファイル: 相手・margin・署名・購入・品目別売上、試合間の農場行動一致 (固定テープか)。
使い方: .venv/bin/python tests/team_profile.py DIR TEAMNAME [--brief]  (DIR = tests/fetch_top.py の出力など)"""
import json,glob,sys,hashlib,itertools
from collections import defaultdict
PASS={"farmer":["PASS"],"hands":[],"market":[]}
def sig24(steps,pi): return hashlib.sha256("\n".join(json.dumps(steps[i][pi].get("action"),sort_keys=True) for i in range(1,25)).encode()).hexdigest()[:10]
def plan(a):
    a=a or PASS; return (tuple(a.get("farmer") or ["PASS"]), tuple(tuple(h) for h in (a.get("hands") or [])))
def h72(steps,s): return hashlib.sha256(json.dumps([plan(steps[t+1][s].get("action")) for t in range(72)]).encode()).hexdigest()[:8]
def animals(farm):
    c=defaultdict(int)
    for row in farm["tiles"]:
        for t in row:
            if isinstance(t,dict) and "animal" in t: c[t["animal"][0]]+=1
    return "C%dS%dG%d"%(c["C"],c["S"],c["G"])
def summarize(steps,s):
    sells=defaultdict(int); rev=defaultdict(float); buys=defaultdict(int); hires=0; land=[]; seeds=defaultdict(int)
    for t in range(1,len(steps)):
        a=steps[t][s].get("action") or PASS; px=steps[t-1][0]["observation"]["market"]["prices"]
        for o in a.get("market") or []:
            if not o: continue
            if o[0]=="SELL" and len(o)>=3 and int(o[2])<900: sells[o[1]]+=int(o[2]); rev[o[1]]+=int(o[2])*px.get(o[1],0)
            elif o[0]=="BUY_ANIMAL": buys[o[1]]+=int(o[2])
            elif o[0]=="HIRE": hires+=1
            elif o[0]=="BUY_LAND": land.append(t-1)
            elif o[0]=="BUY_SEED": seeds[o[1]]+=int(o[2])
    return sells,rev,buys,hires,land,seeds
d,team=sys.argv[1],sys.argv[2]; brief="--brief" in sys.argv
games=[]
for f in sorted(glob.glob(f"{d}/episode-*.json")):
    r=json.load(open(f)); names=r["info"]["TeamNames"]; steps=r["steps"]
    if team not in names or len(steps)<720: continue
    me=names.index(team); op=1-me; games.append((r,me))
    rw=r["rewards"]; last=steps[-1][0]["observation"]; sm=summarize(steps,me); so=summarize(steps,op)
    print(f"{r['info']['EpisodeId']} seat{me} vs {names[op][:14]:<14} {rw[me]:>7.0f} vs {rw[op]:>7.0f} ({rw[me]-rw[op]:+7.0f}) sig24 {sig24(steps,me)} h72 {h72(steps,me)} {animals(last['farms'][me])} | opp {sig24(steps,op)} {animals(last['farms'][op])} shops {','.join(s[:3] for s in last['town']['unlocked_shops'][:3])}")
    if not brief:
        print(f"     me : hires {sm[3]} land {sm[4]} animals {dict(sm[2])} seeds {dict(sm[5])} sold {dict(sm[0])}")
        print(f"          rev~ {{{', '.join(f'{k}:{v/1000:.1f}k' for k,v in sorted(sm[1].items(), key=lambda x:-x[1]))}}}")
        print(f"     opp: hires {so[3]} land {so[4]} animals {dict(so[2])} sold {dict(so[0])} rev~ {{{', '.join(f'{k}:{v/1000:.1f}k' for k,v in sorted(so[1].items(), key=lambda x:-x[1]))}}}")
def plans(steps,s,n): return [json.dumps([(steps[t+1][s].get("action") or PASS).get("farmer"),(steps[t+1][s].get("action") or PASS).get("hands")]) for t in range(n)]
if len(games)>1:
    for n in (72,300,719):
        P=[plans(g[0]["steps"],g[1],n) for g in games]
        agree=[sum(x==y for x,y in zip(a,b)) for a,b in itertools.combinations(P,2)]
        print(f"farm-action agreement over first {n} steps across {len(games)} games: min {min(agree)} median {sorted(agree)[len(agree)//2]} max {max(agree)} / {n}")
