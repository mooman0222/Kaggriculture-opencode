"""Development-only, one-turn sale deferral across a visible demand tick.

Only inherited non-consumable SELL quantities are reduced. All worker actions,
other products and order slots remain unchanged. Delayed units are released on
the next call; inability to add an order postpones release, never drops an order.
"""
import importlib.util
from pathlib import Path

_spec=importlib.util.spec_from_file_location('x092_base',Path(__file__).resolve().parents[1]/'e090/main.py')
_base=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
_ITEMS=('MILK','WOOL','STRAWBERRY')
_MULT=1
_SKIP_SIMILAR=False
_STATE={}
_SH_REPORT=dict(deferred=0,released=0,turns=0,errors=0)


def agent(obs,configuration=None):
    action=_base.agent(obs,configuration)
    step=int(obs['step']);me=int(obs['player'])
    st=_STATE.get(me)
    if st is None or step<=st['last']:
        st=_STATE[me]=dict(last=-1,pending={})
        if step==0:_SH_REPORT.update(deferred=0,released=0,turns=0,errors=0)
    st['last']=step
    orders=[list(o) for o in (action.get('market') or [])]
    stock=_base.projected_shed(action,_base.FarmView(obs))
    changed=False
    if st['pending']:
        for item,q in list(st['pending'].items()):
            planned=sum(max(0,int(o[2])) for o in orders if len(o)>=3 and o[:2]==['SELL',item])
            extra=min(q,max(0,stock.get(item,0)-planned))
            if extra>0:
                same=next((o for o in orders if len(o)>=3 and o[:2]==['SELL',item]),None)
                if same is not None:same[2]+=extra
                elif len(orders)<10:orders.insert(0,['SELL',item,extra])
                else:continue
                changed=True
            # A native all-stock SELL can already include these units.
            del st['pending'][item]
            _SH_REPORT['released']+=q
    if (not st['pending'] and 144<=step<692 and step%4==0
            and obs['farms'][me]['money']>=5000
            and sum(stock.values())<=70
            and not (_SKIP_SIMILAR and _base._r37_similarity(obs)>=.90)
            and all(not o or o[0]=='SELL' for o in orders)):
        draw=_base._race_town(step,obs['town']['unlocked_shops'])
        for item in _ITEMS:
            original=min(stock.get(item,0),sum(max(0,int(o[2])) for o in orders
                         if len(o)>=3 and o[:2]==['SELL',item]))
            q=min(original,_MULT*draw.get(item,0),10)
            if q<=0 or obs['market']['prices'].get(item,0)<=1:continue
            left=original-q
            for o in orders:
                if len(o)>=3 and o[:2]==['SELL',item]:
                    o[2]=min(left,max(0,int(o[2])));left-=o[2]
            st['pending'][item]=q
            _SH_REPORT['deferred']+=q
            changed=True
    if changed:
        _SH_REPORT['turns']+=1
        return dict(action,market=orders)
    return action
