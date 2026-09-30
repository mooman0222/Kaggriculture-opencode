"""Small policy-contract tests; no simulations or multiprocessing."""
import copy
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


spec=importlib.util.spec_from_file_location('short_hold_test',Path(__file__).resolve().parents[1]/'agents/x092/main.py')
policy=importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class ShortHoldContracts(unittest.TestCase):
    def setUp(self):
        policy._STATE.clear()
        policy._MULT=1
        policy._SKIP_SIMILAR=False
        policy._base=SimpleNamespace(agent=lambda o,c:copy.deepcopy(o['action']),
            projected_shed=lambda a,v:v['private']['shed'],FarmView=lambda o:o,
            _race_town=lambda t,s:{'MILK':2,'WOOL':1},_r37_similarity=lambda o:1.0)

    def obs(self,step,market,stock=None):
        return dict(step=step,player=0,farms=[{'money':10000},{}],
                    private={'shed':stock or {'MILK':8}},
                    town={'unlocked_shops':[]},market={'prices':{'MILK':100,'WOOL':100}},
                    action={'farmer':['PASS'],'hands':[['NORTH']], 'market':market})

    def test_deferral_and_release(self):
        a=policy.agent(self.obs(144,[['SELL','MILK',8],['SELL','WHEAT',3]]))
        self.assertEqual(a['market'],[['SELL','MILK',6],['SELL','WHEAT',3]])
        self.assertEqual(a['hands'],[['NORTH']])
        self.assertEqual(policy.agent(self.obs(145,[],{'MILK':2}))['market'],[['SELL','MILK',2]])
        self.assertFalse(policy._STATE[0]['pending'])

    def test_oversell_is_reduced_physically(self):
        a=policy.agent(self.obs(144,[['SELL','MILK',1000],['SELL','MILK',1000]]))
        self.assertEqual(a['market'],[['SELL','MILK',6],['SELL','MILK',0]])

    def test_purchases_and_terminal_not_changed(self):
        for step,orders in [(144,[['SELL','MILK',8],['HIRE']]),(692,[['SELL','MILK',8]])]:
            obs=self.obs(step,orders)
            self.assertEqual(policy.agent(obs),obs['action'])

    def test_full_market_keeps_pending(self):
        policy.agent(self.obs(144,[['SELL','MILK',8]]))
        orders=[['SELL','WOOL',1] for _ in range(10)]
        self.assertEqual(policy.agent(self.obs(145,orders))['market'],orders)
        self.assertEqual(policy._STATE[0]['pending'],{'MILK':2})
        self.assertEqual(policy.agent(self.obs(146,[]))['market'],[['SELL','MILK',2]])

    def test_existing_sweep_not_duplicated(self):
        policy.agent(self.obs(144,[['SELL','MILK',8]]))
        obs=self.obs(145,[['SELL','MILK',1000]],{'MILK':2})
        self.assertEqual(policy.agent(obs),obs['action'])
        self.assertFalse(policy._STATE[0]['pending'])

    def test_similar_layout_keeps_native_policy(self):
        policy._SKIP_SIMILAR=True
        obs=self.obs(144,[['SELL','MILK',8]])
        self.assertEqual(policy.agent(obs),obs['action'])
        self.assertFalse(policy._STATE[0]['pending'])


if __name__=='__main__':unittest.main()
