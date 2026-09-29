import unittest
import numpy as np
from src.semantic_complementarity import describe, direction_screen, edit_vectors, ranking


class ComplementarityTest(unittest.TestCase):
    def test_ranks_and_ties(self):
        for scores, auc, ap in (([.1,.9],1.,1.), ([.9,.1],0.,.5), ([.5,.5],.5,.5)):
            r = ranking(scores, [False,True])
            self.assertEqual(r["auroc"], auc); self.assertEqual(r["ap"], ap)
        self.assertIsNone(ranking([],[])["auroc"])
        self.assertIsNone(ranking([1],[True])["auroc"])

    def test_rank_matches_brute_pairs(self):
        rng = np.random.default_rng(17)
        for _ in range(20):
            s = rng.integers(0,10,100); y = rng.random(100)>.7
            diff = s[y,None]-s[None,~y]
            self.assertAlmostEqual(ranking(s,y)["auroc"], float(((diff>0)+.5*(diff==0)).mean()))

    def vectors(self):
        gt=np.array([[4,0,2,0,4,3,1]],np.uint8)
        base=np.array([[0,0,0,4,4,4,1]],np.uint8)
        p=np.array([[.5,1.,.75,0.,.49,.1,.9]],np.float32)
        new=np.array([[4,4,4,0,0,0,1]],np.uint8)
        dist=np.ones_like(p)
        return edit_vectors(gt,base,new,p,p,dist,dist)

    def test_cohorts_include_neutral_structural_errors(self):
        for vectors in self.vectors().values():
            d=describe(vectors,[.5,.75,1.000001],[4,8])
            self.assertEqual((d["pixels"],d["corrected"],d["harmed"],d["neutral"]),(3,1,1,1))
            self.assertEqual(sum(r["pixels"] for r in d["confidence_bins"]),3)
            self.assertEqual(d["agreement"]["other_agrees"]["pixels"],3)
            self.assertEqual(d["ranking"]["confidence"]["prevalence"],1/3)

    def test_protected_and_probability_replay_guards(self):
        z=np.zeros((2,2),np.uint8); p=np.ones_like(z,dtype=float)
        with self.assertRaisesRegex(ValueError,"fusion"):
            edit_vectors(z,z,z,p,p,p,p)
        with self.assertRaisesRegex(ValueError,"Protected"):
            edit_vectors(z,np.ones_like(z),z,p,p,p,p)

    def test_screen_requires_all_folds_and_budget(self):
        good={str(i):{"corrected":400,"harmed":1} for i in range(3)}
        self.assertTrue(direction_screen(good))
        good["2"]["harmed"]=401
        self.assertFalse(direction_screen(good))
        self.assertFalse(direction_screen({"0":good["0"]}))
