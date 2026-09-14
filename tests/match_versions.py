"""相手の版特定: 実戦リプレイを kagsim で再現し (我々の席 = 自分の提出物、相手席 = 候補 NB)、記録された相手行動と最初にずれる step を出す。None = 719 手完全再現。
使い方: .venv/bin/python tests/match_versions.py --replays 'tmp/e058/e057_battles/episode-*.json' --me agents/live_d --cands DIR1 DIR2 ... [--team MMN0222] [--out res.json]"""
import json,hashlib,glob,os,sys,importlib.util,collections
from concurrent.futures import ProcessPoolExecutor
import kagsim
_DEFAULT_CANDS=["tmp/e058/agents/ahmedberatozer_kaggriculture-v38-smarter-feed-stronger-margins","tmp/e058/agents/ahmedberatozer_kaggriculture-v39-ready-before-the-rush","tmp/e058/agents/ahmedberatozer_more-yield-smarter-labor","tmp/e058/agents/aurax7_kaggriculture-shop-router-reactive-v4","tmp/e058/agents/guru_master_engine_v3","tmp/e058/agents/pilkwang_structured_economic_policy","tmp/e058/agents/yhay81_shop-router-0913","tmp/e058/agents/ahmedberatozer_kaggriculture-v41-review-candidate"]
def load(d,name):
    p=os.path.join(d,"main.py"); sys.path.insert(0,d)
    s=importlib.util.spec_from_file_location(name,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def reset(m):
    for k in ("_LIVE","_POLICY","_ROUTER","_IMPL"):
        pass
CANDS=[]; ME="agents/live_d"; TEAM="MMN0222"
def one(f):
    r=json.load(open(f)); names=r["info"]["TeamNames"]; me=names.index(TEAM); op=1-me; seed=r["info"]["seed"]
    rec_op=[r["steps"][t+1][op].get("action") for t in range(719)]; rec_me=[r["steps"][t+1][me].get("action") for t in range(719)]
    out={"ep":r["info"]["EpisodeId"],"opp":names[op],"seed":seed,"me_seat":me,"res":{}}
    for d in CANDS+[ME]:
        tag=os.path.basename(d)
        try:
            mine=load(ME,"me_"+str(seed)+tag); cand=load(d,"c_"+str(seed)+tag)
            g=kagsim.Game(seed); first=None; first_me=None
            while not g.done:
                t=g.step_count; a=mine.agent(g.observe(me)); b=cand.agent(g.observe(op))
                if first is None and json.dumps(b,sort_keys=True)!=json.dumps(rec_op[t],sort_keys=True): first=t
                if first_me is None and json.dumps(a,sort_keys=True)!=json.dumps(rec_me[t],sort_keys=True): first_me=t
                g.step(*((a,b) if me==0 else (b,a)))
            out["res"][tag]=(first,first_me,g.reward(op)-r["rewards"][op])
        except Exception as e: out["res"][tag]=("ERR",str(e)[:60],None)
    return out
def _init(c, m, t):
    global CANDS, ME, TEAM; CANDS, ME, TEAM = c, m, t
if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--replays",required=True); ap.add_argument("--me",default="agents/live_d"); ap.add_argument("--cands",nargs="+",default=_DEFAULT_CANDS)
    ap.add_argument("--team",default="MMN0222"); ap.add_argument("--out",default="tmp/match_versions.json"); ap.add_argument("--workers",type=int,default=8)
    a=ap.parse_args(); files=sorted(glob.glob(a.replays))
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.cands,a.me,a.team)) as ex: res=list(ex.map(one,files))
    json.dump(res,open(a.out,"w"),indent=0)
    keys=[k for k in res[0]["res"] if k!=os.path.basename(a.me)]
    exact=collections.Counter(); best=collections.Counter()
    for o in res:
        ex_=[k for k in keys if o["res"][k][0] is None]
        for k in ex_: exact[k]+=1
        b=max(keys,key=lambda k:(10**6 if o["res"][k][0] is None else (o["res"][k][0] if isinstance(o["res"][k][0],int) else -1))); best[b]+=1
        print(str(o["ep"])[-6:], o["opp"][:14].ljust(14), "exact:",",".join(k[:20] for k in ex_) or "-", "| div:", " ".join(f"{k[:14]}={o['res'][k][0]}" for k in keys if o["res"][k][0] is not None))
    print("\nexact reproductions:",dict(exact)); print("longest prefix:",dict(best))
