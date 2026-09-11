"""1試合の直接対決: .venv/bin/python tests/quick.py A.py B.py SEED"""
import importlib.util, sys
from kaggle_environments import make
def load(p):
    s=importlib.util.spec_from_file_location('m'+str(abs(hash(p))),p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return getattr(m,'agent_entry',m.agent)
A=load(sys.argv[1]); B=load(sys.argv[2]); seed=int(sys.argv[3])
env=make("kaggriculture",configuration={"episodeSteps":720}); env.info["seed"]=seed; env.run([A,B])
r=[s.reward for s in env.state]; print(sys.argv[1].split('/')[-2], 'vs', sys.argv[2].split('/')[-2], 'seed',seed, 'rewards',r,'margin',r[0]-r[1], [s.status for s in env.state])
