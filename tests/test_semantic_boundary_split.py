import unittest

import numpy as np

from src.semantic_boundary_split import (
    add_signed_support,
    conflict_score,
    inherited_child_values,
    robust_signed_evidence,
    select_matched_pair,
    stable_topk,
)


class SemanticBoundarySplitTests(unittest.TestCase):
    def test_normalization_is_bounded_and_sign_preserving(self):
        got = robust_signed_evidence(np.array([-4.0, -1.0, 0.0, 2.0]))
        self.assertTrue(np.all(np.abs(got) < 1.0))
        self.assertTrue(np.array_equal(np.sign(got), np.array([-1.0, -1.0, 0.0, 1.0])))

    def test_conflict_requires_both_signs(self):
        pos = np.zeros(3)
        neg = np.zeros(3)
        pos, neg = add_signed_support(pos, neg, np.array([0.5, 0.5, -0.5]))
        pos, neg = add_signed_support(pos, neg, np.array([0.5, -0.5, 0.5]))
        got = conflict_score(pos, neg, prior=0.0)
        self.assertEqual(got[0], 0.0)
        self.assertEqual(got[1], 1.0)
        self.assertEqual(got[2], 1.0)

    def test_stable_topk_respects_eligibility_and_id_tie_break(self):
        got = stable_topk(np.array([9.0, 4.0, 4.0, 8.0]), np.array([0, 1, 1, 0]), 2)
        np.testing.assert_array_equal(got, [1, 2])

    def test_pair_has_exact_matched_budget(self):
        pair = select_matched_pair(
            np.array([4.0, 3.0, 2.0, 1.0]),
            np.array([1.0, 4.0, 2.0, 3.0]),
            np.ones(4, dtype=bool),
            2,
        )
        np.testing.assert_array_equal(pair.control, [0, 1])
        np.testing.assert_array_equal(pair.candidate, [1, 3])
        self.assertEqual(pair.overlap, 1)

    def test_sidecar_matches_native_split_then_parent_prune_order(self):
        values = np.array([[0], [1], [2], [3]])
        got = inherited_child_values(values, np.array([3, 1]), children=2)
        np.testing.assert_array_equal(got[:, 0], [0, 2, 3, 1, 3, 1])

    def test_invalid_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            stable_topk(np.ones(2), np.array([True, False]), 2)


if __name__ == "__main__":
    unittest.main()
