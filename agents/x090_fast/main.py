"""Development-only exact memoization of E090's repeated fixed-sell search.

The cache lives for one outer agent call. Full worker actions and market orders
are part of its key; the observation is constant during that call. Cached orders
are copied to avoid aliasing with subsequent wrappers. Diagnostic counters are
replayed as well. This changes computation, not the policy.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location('x090_fast_base', Path(__file__).resolve().parents[1] / 'e090/main.py')
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
_original_reorder = _base._s793_reorder
_cache = {}
_FAST_REPORT = dict(hits=0, misses=0)


def _memo_reorder(obs, action):
    key = repr((action.get('farmer'), action.get('hands'), action.get('market')))
    if key in _cache:
        unchanged, market, delta = _cache[key]
        for name, value in delta.items():
            _base._S793_REPORT[name] += value
        _FAST_REPORT['hits'] += 1
        if unchanged:
            return action
        return dict(action, market=_base.copy.deepcopy(market))
    before = dict(_base._S793_REPORT)
    result = _original_reorder(obs, action)
    delta = {k: v-before[k] for k, v in _base._S793_REPORT.items()
             if isinstance(v, (int, float)) and v != before[k]}
    _cache[key] = (result is action, _base.copy.deepcopy(result.get('market')), delta)
    _FAST_REPORT['misses'] += 1
    return result


_base._s793_reorder = _memo_reorder


def agent(obs, configuration=None):
    _cache.clear()
    if int(obs['step']) == 0:
        _FAST_REPORT.update(hits=0, misses=0)
    return _base.agent(obs, configuration)
