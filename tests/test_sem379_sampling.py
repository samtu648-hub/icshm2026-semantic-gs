"""Native grid, paired transforms, border and RNG isolation contracts."""
import random
import unittest
import numpy as np
from src.semantic_cable_sampling import CableRegionSampler
from src.quality_mask2former_training import sampled_crop

class SamplingTests(unittest.TestCase):
 def setUp(self):
  self.y=np.zeros((989,1320),np.uint8)
  self.y[:,0]=2;self.y[988,:]=2;self.y[200:800,650]=2
  self.x=np.repeat(self.y[:,:,None],3,axis=2)
 def test_zero_mixture_exact_parent(self):
  a=random.Random(20260805);b=random.Random(20260805);s=CableRegionSampler(sampled_crop,mixture=0)
  for _ in range(12):
   x,y,t=sampled_crop(a,['a'],{'a':self.x},{'a':self.y})
   xx,yy,tt=s(b,['a'],{'a':self.x},{'a':self.y})
   self.assertTrue(np.array_equal(x,xx));self.assertTrue(np.array_equal(y,yy));self.assertEqual(t,{k:tt[k] for k in t})
  self.assertEqual(a.getstate(),b.getstate())
 def test_guided_native_grid_and_borders(self):
  a=random.Random(4);b=random.Random(4);s=CableRegionSampler(sampled_crop,mixture=1)
  for _ in range(40):
   _,_,base=sampled_crop(a,['a'],{'a':self.x},{'a':self.y})
   x,y,t=s(b,['a'],{'a':self.x},{'a':self.y})
   self.assertEqual(t['flip'],base['flip']);self.assertEqual(x.shape,(768,768,3))
   self.assertTrue(np.array_equal(x[:,:,0],y));self.assertEqual(self.y[t['anchor_y'],t['anchor_x']],2)
   self.assertTrue(t['top']<=t['anchor_y']<t['top']+768);self.assertTrue(t['left']<=t['anchor_x']<t['left']+768)
  self.assertEqual(a.getstate(),b.getstate())
 def test_no_cable_keeps_view(self):
  s=CableRegionSampler(sampled_crop,mixture=1);x,y,t=s(random.Random(2),['a'],{'a':self.x},{'a':np.zeros_like(self.y)})
  self.assertEqual(t['source'],'uniform_no_cable');self.assertEqual(t['name'],'a')
if __name__=='__main__':unittest.main()
