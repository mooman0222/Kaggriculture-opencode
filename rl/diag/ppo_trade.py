"""Did PPO checkpoints drift toward liquidation? Per-game spend/sell telemetry vs v41 for Policy2 ckpts. usage: ppo_trade.py CKPT... """
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), "rl"))
import kagsim
from play2 import BCAgent, load_py

opp = load_py("third_party/public_agents/v41/main.py")
for ck in sys.argv[1:]:
    ag = BCAgent(ck); rows = []
    for gi in range(8):
        seed, seat = 5000 + gi // 2, gi % 2
        for k in ("_LIVE", "_POLICY"):
            if hasattr(opp, k): setattr(opp, k, None)
        g = kagsim.Game(seed); ordered = {"BUY_SEED": 0, "BUY_ANIMAL": 0, "BUY_PRODUCT": 0, "SELL": 0, "HIRE": 0}
        while not g.done:
            a = ag.act(g.observe(seat), seat)
            for o in a.get("market") or []:
                if o and o[0] in ordered: ordered[o[0]] += int(o[2]) if len(o) > 2 else 1
            b = opp.agent(g.observe(1 - seat))
            g.step(*((a, b) if seat == 0 else (b, a)))
        t = g.telemetry(seat)
        rows.append([g.reward(seat), g.reward(seat) - g.reward(1 - seat), t["total_spend"], t["hire_paid"], t["spend_by"]["LAND"],
                     t["total_spend"] - t["hire_paid"] - t["spend_by"]["LAND"], t["sell_revenue"], t["sold_units"],
                     ordered["BUY_SEED"], ordered["BUY_ANIMAL"], ordered["BUY_PRODUCT"], ordered["SELL"]])
    r = np.mean(rows, 0)
    print(f"{os.path.basename(ck):14s} own {r[0]:7.0f} margin {r[1]:+7.0f} | spend {r[2]:7.0f} (hire {r[3]:5.0f} land {r[4]:5.0f} seeds+animals+wheat {r[5]:6.0f}) "
          f"| sell rev {r[6]:7.0f} units {r[7]:5.0f} | ordered seed {r[8]:5.0f} animal {r[9]:4.0f} wheat {r[10]:5.0f} sell {r[11]:6.0f}", flush=True)
