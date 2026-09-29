"""CPU geometry indexing and fixed role tests, independent of native CUDA."""
import unittest,json
from pathlib import Path
from scripts.run_sem374_structure import schedule
class StructureProtocol(unittest.TestCase):
 def test_complete_balanced_schedule(self):
  names=[f'{x:03}.png' for x in range(1,401)];a=schedule(names,2000,20260916)
  self.assertEqual(a,schedule(reversed(names),2000,20260916))
  self.assertTrue(all(a.count(n)==5 for n in names))
 def test_fixed_recipe_and_roles(self):
  c=json.loads(Path('experiments/runs/SEM-374-STANDARD-STRUCTURE-DEPTH/config.yaml').read_text())
  self.assertEqual(c['structure_classes'],[1,3,4]);self.assertEqual(c['structure_weight'],.05)
  self.assertFalse(c['densification']);self.assertFalse(c['pruning']);self.assertEqual(c['matched_steps'],2000)
 def test_edge_origin_conversion(self):
  for width in [989,1320]:
   for index in [0,1,width-1]:
    colmap=index+.5;grid=2*(colmap-.5)/(width-1)-1
    recovered=(grid+1)*(width-1)/2
    self.assertAlmostEqual(recovered,index)
if __name__=='__main__':unittest.main()
