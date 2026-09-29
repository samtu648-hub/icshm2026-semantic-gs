import unittest
from types import SimpleNamespace
from src.quality_fit_view_expansion import expand_roles, coverage


class ExpansionRolesTests(unittest.TestCase):
    def setUp(self):
        self.roles=[dict(id='f-b',fold='f',fit=['old'],evaluation=['eval'],guard=['guard'])]
        self.cameras={k:SimpleNamespace(center=(x,0,0)) for k,x in [('old',4),('eval',0),('guard',.1),('near',.2),('edge',1),('far',3)]}

    def test_guard_boundary_preserves_original_evaluation_and_training(self):
        new=expand_roles(self.roles,{'f':['far','near','edge']},self.cameras,1)
        self.assertEqual(new[0]['fit'],['old','edge','far'])
        self.assertEqual(new[0]['guard'],['guard','near'])
        self.assertEqual(new[0]['evaluation'],['eval'])
        self.assertEqual(self.roles[0]['fit'],['old'])
        d=coverage(self.roles,new,self.cameras)['f-b']
        self.assertEqual(d['closer_evaluation_views'],1)
        self.assertEqual(d['rows'][0]['nearest_fit_after'],1)

    def test_extra_evaluation_or_duplicate_is_rejected(self):
        for names in [['eval'],['old'],['guard'],['far','far']]:
            with self.assertRaises(ValueError):
                expand_roles(self.roles,{'f':names},self.cameras,1)
