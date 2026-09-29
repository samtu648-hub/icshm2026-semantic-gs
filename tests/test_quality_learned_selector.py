import unittest
import numpy as np
from src.quality_learned_selector import candidates,choose,feature_rows,neighbourhood,predict,ridge_fit,sample_rows
from scripts.audit_quality_learned_selector import reference_rows,local_reference
class SelectorTests(unittest.TestCase):
    def test_ridge_matches_augmented_least_squares(self):
        rng=np.random.default_rng(42);x=np.column_stack([rng.normal(size=(80,4)),np.ones(80)]);y=rng.normal(size=80);w=rng.uniform(.1,1,80)
        m=ridge_fit(x,y,w,.01);z=(x-m['mean'])/m['scale'];pen=np.eye(5)*.1;pen[-1,-1]=0
        a=np.vstack([z*np.sqrt(w/w.sum())[:,None],pen]);b=np.r_[y*np.sqrt(w/w.sum()),np.zeros(5)]
        ref=np.linalg.lstsq(a,b,rcond=None)[0];np.testing.assert_allclose(ref,m['coef'],atol=1e-12)
    def test_acceptance_and_abstention(self):
        raw=np.array([2,2,2,2,2]);allowed=np.ones((5,5),bool);scores=np.full((5,5),-1.);scores[0,0]=.1;scores[0:2,1]=.1;scores[0,2]=0;allowed[:,3]=False;scores[4,4]=.2
        np.testing.assert_array_equal(choose(raw,allowed,scores),[0,2,2,2,4])
        maps=np.full((5,5),5);self.assertFalse(candidates(raw,maps).any())
    def test_features_sampling_and_off(self):
        rng=np.random.default_rng(11);raw=rng.integers(0,5,(7,8),dtype=np.uint8);maps=rng.integers(0,6,(5,7,8),dtype=np.uint8);maps[0]%=5
        probs=[rng.dirichlet(np.ones(5),56).T.reshape(5,7,8) for _ in range(3)];data=(raw,maps,probs,rng.random((2,7,8)),neighbourhood(raw))
        np.testing.assert_array_equal(data[4],local_reference(raw));ix,cs=sample_rows(data,9,'fold','name',10)
        self.assertLessEqual(len(np.unique(ix)),10);self.assertTrue(np.all(candidates(raw,maps).reshape(5,-1)[cs,ix]))
        np.testing.assert_allclose(feature_rows(data,ix,cs),reference_rows(data,ix,cs),atol=1e-14)
        np.testing.assert_array_equal(predict(data,dict(fallback=True)),raw)
        self.assertEqual(feature_rows(data,ix,cs).shape[1],43)
    def test_fit_is_invariant_to_row_order_and_weight_scale(self):
        x=np.column_stack([np.arange(10),np.ones(10)]);y=np.linspace(-1,1,10);w=np.arange(1,11,dtype=float)
        m=ridge_fit(x,y,w,.01);other=ridge_fit(x[::-1],y[::-1],w[::-1]*100,.01)
        for key in ('coef','mean','scale'):
            np.testing.assert_allclose(m[key],other[key],atol=1e-12)
