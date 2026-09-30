"""Development-only wheat roundtrip overlay; not a standalone submission."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location('x091_base', Path(__file__).resolve().parents[1] / 'e090/main.py')
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
_Q = 60
_PHASE = 0
_XR_REPORT = dict(turns=0, quantity=0, errors=0)


def agent(obs, configuration=None):
    a = _base.agent(obs, configuration)
    t = int(obs['step'])
    if t == 0:
        _XR_REPORT.update(turns=0, quantity=0, errors=0)
    if t < 144 or t >= 696 or t % 4 != _PHASE or t % 24 == 23:
        return a
    orders = a.get('market') or []
    if len(orders)>8 or any(len(o)>1 and (o[1]=='WHEAT' or o[0] in
                         ('BUY_PRODUCT','BUY_ANIMAL')) for o in orders):
        return a
    stock = _base.projected_shed(a, _base.FarmView(obs))
    q = min(_Q, max(0, 100-sum(stock.values())))
    if q <= 0:
        return a
    inv = obs['market']['inventory']['WHEAT']
    cash_bound = q*_base._r37_market_price('WHEAT', inv-q-100)
    if obs['farms'][obs['player']]['money'] <= cash_bound+5000:
        return a
    _XR_REPORT['turns'] += 1
    _XR_REPORT['quantity'] += q
    return dict(a, market=[['BUY_PRODUCT','WHEAT',q]]+orders+[['SELL','WHEAT',q]])
