"""提出前チェック: 実エンジン (kaggle_environments) で自己対戦し、両者 DONE・対称・最大手番時間を確認する。"""
import sys, time, importlib.util
from kaggle_environments import make
path = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 938001
def load(tag):
    s = importlib.util.spec_from_file_location('pc_' + tag, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m.agent
A, B = load('a'), load('b')
mx = [0.0]
def timed(f):
    def g(obs, cfg=None):
        t0 = time.time(); r = f(obs, cfg); mx[0] = max(mx[0], time.time() - t0); return r
    return g
env = make('kaggriculture', configuration={'episodeSteps': 720}, debug=False)
env.info['seed'] = seed
env.run([timed(A), timed(B)])
last = env.steps[-1]
print('statuses', [s.status for s in last], 'rewards', [s.reward for s in last], 'steps', len(env.steps), f'max act {mx[0]:.3f}s')
