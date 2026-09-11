import importlib.util, sys, os, kagsim
from collections import Counter
def load(p):
    d=os.path.dirname(os.path.abspath(p)); sys.path.insert(0,d)
    s=importlib.util.spec_from_file_location('m'+str(abs(hash(p))),p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
A=load('agents/live_b/main.py'); B=load('agents/sr0909_base/main.py')
tot=0
for seed in [int(x) for x in sys.argv[1:]] or (7,11):
    A._LIVE=None; B._POLICY=None
    g=kagsim.Game(seed); ops=Counter(); sells=Counter()
    while not g.done:
        o=g.observe(0); t=g.step_count; a=A.agent(o)
        if t>=288:
            for u in [a.get('farmer') or ['PASS'], *(a.get('hands') or [])]: ops[u[0]]+=1
            for od in a.get('market') or []:
                if od and od[0]=='SELL': sells[od[1]]+=min(int(od[2]), o['private']['shed'].get(od[1],0))
        g.step(a, B.agent(g.observe(1)))
    tot+=g.reward(0)-g.reward(1)
    print(f"seed {seed} live_b {g.reward(0):,.0f} vs 0909 {g.reward(1):,.0f} margin {g.reward(0)-g.reward(1):+,.0f} | ops {dict(ops.most_common(8))} | sells {dict(sells)}")
print('mean margin', tot/len(sys.argv[1:] or (7,11)))
