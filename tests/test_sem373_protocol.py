"""CPU protocol invariants for the isolated semantic geometry experiment."""
import unittest
from scripts.run_sem373_joint import schedule, GeometryView

class ProtocolTest(unittest.TestCase):
 def test_matched_sampling_complete_epochs(self):
  names=[f'{i:03}.png' for i in range(1,401)]
  a=schedule(names,2000,20260916)
  self.assertEqual(a,schedule(reversed(names),2000,20260916))
  for i in range(0,2000,400):self.assertEqual(set(a[i:i+400]),set(names))
  self.assertEqual(sum(int(n[:3])<=300 for n in a),1500)
 def test_detach_route(self):
  class Value:
   def detach(self):return 'detached'
  class Model:
   get_xyz=Value();get_scaling=Value();get_rotation=Value();get_opacity=Value()
  m=Model();control=GeometryView(m,());joint=GeometryView(m,('get_xyz',))
  self.assertEqual(control.get_xyz,'detached');self.assertIs(joint.get_xyz,m.get_xyz)
  for n in ['get_scaling','get_rotation','get_opacity']:self.assertEqual(getattr(joint,n),'detached')

if __name__=='__main__':unittest.main()
