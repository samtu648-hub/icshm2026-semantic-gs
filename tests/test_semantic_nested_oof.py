import unittest
import numpy as np
from src.semantic_nested_oof import (nested_plan, validate_inner,
                                      validate_teacher_metadata, common_sampling_base)


class NestedContracts(unittest.TestCase):
    def setUp(self):
        self.centers = {str(i): np.array([float(i), 0., 0.]) for i in range(12)}
        self.outer = {'fold': 0, 'train': [str(i) for i in range(10)], 'oof': ['10'], 'guard': ['11']}

    def test_deterministic_and_complete(self):
        one = nested_plan([self.outer], self.centers, 20260805, 1.1)
        self.assertEqual(one, nested_plan([self.outer], self.centers, 20260805, 1.1))
        self.assertEqual(sorted(n for i in one[0]['inners'] for n in i['oof']), sorted(self.outer['train']))
        for inner in one[0]['inners']:
            self.assertGreaterEqual(validate_inner(self.outer, inner, self.centers, 1.1), 1.1)

    def test_target_leak_rejected(self):
        inner = {'train': ['1', '10'], 'oof': ['0'], 'guard': [str(i) for i in range(2, 10)]}
        with self.assertRaises(ValueError):
            validate_inner(self.outer, inner, self.centers, .5)

    def test_wrong_teacher_rejected(self):
        inner = nested_plan([self.outer], self.centers, 0, .5)[0]['inners'][0]
        meta = {'experiment_id': 'SEM-263-NESTED-POINT-OOF', 'outer_fold': 1,
                'inner_fold': 0, 'kind': 'deck', **{k: inner[k] for k in ('train', 'oof', 'guard')}}
        with self.assertRaises(ValueError):
            validate_teacher_metadata(meta, self.outer, inner, 'deck')

    def test_inner_neighbour_guard_rejected(self):
        inner = {'train': [str(i) for i in range(1,10)], 'oof': ['0'], 'guard': []}
        with self.assertRaises(ValueError):
            validate_inner(self.outer,inner,self.centers,1.1)

    def test_duplicate_access_rejected(self):
        inner = {'train': [str(i) for i in range(1,10)]+['1'], 'oof': ['0'], 'guard': []}
        with self.assertRaises(ValueError):
            validate_inner(self.outer,inner,self.centers,.5)

    def test_sampler_not_prediction_mutation(self):
        static = np.array([[0, 1, 2, 3, 4]], np.uint8)
        np.testing.assert_array_equal(common_sampling_base(static), [[0, 1, 0, 3, 4]])
        np.testing.assert_array_equal(static, [[0, 1, 2, 3, 4]])


if __name__ == '__main__':
    unittest.main()
