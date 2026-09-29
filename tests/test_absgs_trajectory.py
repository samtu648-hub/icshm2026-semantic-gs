import unittest
import numpy as np
from src.absgs_trajectory import Lineage


class LineageTests(unittest.TestCase):
    def test_split_repeat_order_and_parent_removal(self):
        state = Lineage(3)
        child = state.append(np.tile([0, 2], 2), 600, "split")
        self.assertEqual(child.tolist(), [3, 4, 5, 6])
        self.assertEqual(state.births[0][1].tolist(), [0, 2, 0, 2])
        removed = state.prune([True, False, True, False, False, False, False])
        self.assertEqual(removed.tolist(), [0, 2])
        self.assertEqual(state.ids.tolist(), [1, 3, 4, 5, 6])
        newer = state.append([1, 4], 700, "clone")
        self.assertEqual(newer.tolist(), [7, 8])
        self.assertEqual(state.births[-1][2].tolist(), [700, 700])

    def test_empty_append_keeps_ids_and_counter(self):
        state = Lineage(4)
        state.append([], 600, "clone")
        self.assertEqual(state.ids.tolist(), [0, 1, 2, 3])
        self.assertEqual(state.next_id, 4)
        self.assertEqual(state.births, [])

    def test_bad_native_mask_fails(self):
        with self.assertRaises(ValueError):
            Lineage(4).prune([True])


if __name__ == "__main__":
    unittest.main()
