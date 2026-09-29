import unittest
import numpy as np
from src.quality_class_conditional_selector import solve_blocks,predict
from scripts.audit_quality_class_conditional_selector import reference_predict


class ConditionalObjectiveTests(unittest.TestCase):
    def test_global_normalization_matches_joint_block_system(self):
        x=np.array([[0.,1],[1,1],[0,1],[2,1],[3,1]])
        y=np.array([-1.,1,0,1,-1]);w=np.array([1.,1,2,2,4]);cid=np.array([0,0,1,1,1])
        g=np.zeros((5,2,2));b=np.zeros((5,2));counts=np.zeros(5,int)
        for c in (0,1):
            keep=cid==c;q=x[keep];g[c]=q.T@(w[keep,None]*q);b[c]=q.T@(w[keep]*y[keep]);counts[c]=keep.sum()
        result=solve_blocks(g,b,w.sum(),counts,.1)
        joint=np.zeros((len(x),4))
        for i,c in enumerate(cid):joint[i,c*2:c*2+2]=x[i]
        reg=np.diag([.1,0,.1,0]);expected=np.linalg.solve(joint.T@(w[:,None]*joint)/w.sum()+reg,joint.T@(w*y)/w.sum())
        np.testing.assert_allclose(result['coef'][:2].ravel(),expected)
        np.testing.assert_array_equal(result['coef'][2:],0)
        wrong=np.linalg.solve(g[0]/w[cid==0].sum()+np.diag([.1,0]),b[0]/w[cid==0].sum())
        self.assertFalse(np.allclose(result['coef'][0],wrong))

    def test_positive_tie_and_missing_blocks_keep_raw(self):
        raw=np.zeros((2,2),np.uint8);maps=np.ones((5,2,2),np.uint8);maps[1]=2
        prob=np.full((5,2,2),.2,np.float32);data=(raw,maps,[prob]*3,np.full((2,2,2),.5),np.ones((2,2)))
        model=dict(mean=np.zeros(43),scale=np.ones(43),coef=np.zeros((5,43)),active=np.array([False,True,True,False,False]),fallback=np.array(False))
        model['coef'][1,-1]=1;model['coef'][2,-1]=1
        np.testing.assert_array_equal(predict(data,model),raw)
        np.testing.assert_array_equal(reference_predict(data,model),raw)
        model['active'][2]=False
        np.testing.assert_array_equal(predict(data,model),np.ones_like(raw))
        np.testing.assert_array_equal(reference_predict(data,model),np.ones_like(raw))


if __name__=='__main__':unittest.main()
