"""実エンジンで A vs B を 1 局。両者の状態・報酬・最大手番時間と、A の層の発動回数を出す。"""
import sys, time, importlib.util
from kaggle_environments import make
pa, pb, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
def load(p, tag):
    s = importlib.util.spec_from_file_location(tag, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
ma, mb = load(pa, 'ra'), load(pb, 'rb')
mx = [0.0]
def timed(f):
    def g(obs, cfg=None):
        t0 = time.time(); r = f(obs, cfg); mx[0] = max(mx[0], time.time() - t0); return r
    return g
env = make('kaggriculture', configuration={'episodeSteps': 720}, debug=False); env.info['seed'] = seed
env.run([timed(ma.agent), timed(mb.agent)])
last = env.steps[-1]
print('statuses', [s.status for s in last], 'rewards', [s.reward for s in last], f'max act {mx[0]:.3f}s',
      'A wheat', getattr(ma, '_X101_REPORT', None), 'A squeeze', getattr(ma, '_X096_REPORT', None))
