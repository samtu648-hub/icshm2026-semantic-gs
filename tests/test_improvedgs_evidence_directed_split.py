import unittest


class EvidenceDirectedSplitTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional torch runtime unavailable")
        from src.improvedgs_evidence_directed_split import compose_evidence_directed_selection

        self.torch = torch
        self.compose = compose_evidence_directed_selection

    def test_directed_selection_forces_evidence_and_preserves_budget(self):
        native = self.torch.tensor([0, 1, 2, 3, 4])
        valid = self.torch.tensor([False, False, True, False, False, True, True, False])
        priority = self.torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 3.0, 0.0])
        final, forced, kept = self.compose(native, valid, priority, 5)
        self.assertEqual(set(forced.tolist()), {2, 5, 6})
        self.assertEqual(set(final.tolist()), {0, 1, 2, 5, 6})
        self.assertEqual(kept.tolist(), [0, 1])
        self.assertEqual(len(final), 5)

    def test_directed_selection_uses_priority_when_evidence_exceeds_budget(self):
        native = self.torch.tensor([0, 1])
        valid = self.torch.tensor([True, True, True, True])
        priority = self.torch.tensor([1.0, 4.0, 3.0, 2.0])
        final, forced, kept = self.compose(native, valid, priority, 2)
        self.assertEqual(set(forced.tolist()), {1, 2})
        self.assertEqual(set(final.tolist()), {1, 2})
        self.assertEqual(kept.numel(), 0)


if __name__ == "__main__":
    unittest.main()
