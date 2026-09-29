import json
from pathlib import Path
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None
if torch is not None:
    from src.offset_responsibility import dominant_evidence, posterior


@unittest.skipIf(torch is None, "requires optional Torch runtime")
class ResponsibilityTests(unittest.TestCase):
    def test_dominant_assignment_preserves_offset_and_class(self):
        winner = torch.tensor([[0, 1, 1, -1, 2]])
        weight = torch.tensor([[.2, .5, .4, 0., .7]])
        target = torch.tensor([[0, 1, 2, 0, 1]])
        actual = dominant_evidence(winner, weight, target, 3)
        torch.testing.assert_close(actual, torch.tensor([[.2, 0., .2], [0., .5, .9], [0., .7, .7]]))

    def test_unseen_uniform_and_posterior_simplex(self):
        counts = torch.tensor([[0., 0., 0.], [3., 1., 5.]])
        value = posterior(counts)
        torch.testing.assert_close(value, torch.tensor([[.5,.5], [4/6,2/6]]))
        torch.testing.assert_close(value.sum(-1),torch.ones(2))


class ResponsibilitySplitTests(unittest.TestCase):
    def test_registered_fixed_spatial_exclusion(self):
        root = Path(__file__).resolve().parents[1]
        split = json.loads((root/"experiments/splits/sem_005_oof_spatial_v1.json").read_text())
        oof = []
        for fold in split["folds"]:
            self.assertTrue(set(fold["train"]).isdisjoint(fold["oof"]))
            self.assertTrue(set(fold["train"]).isdisjoint(fold["guard"]))
            self.assertTrue(set(fold["train"]).isdisjoint(split["fixed_validation_read_only"]))
            oof.extend(fold["oof"])
        self.assertEqual(len(oof),219)
        self.assertEqual(len(set(oof)),219)


if __name__ == "__main__":
    unittest.main()
