from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # Repository-health deliberately uses torch-free Python.
    torch = None

if torch is not None:
    from src.scaffold_offset_semantic import OffsetSemanticResidualSidecar, active_offset_indices
else:
    OffsetSemanticResidualSidecar = None
    active_offset_indices = None


@unittest.skipIf(torch is None, "torch is intentionally absent from the lightweight repository-health interpreter")
class OffsetSemanticSidecarTests(unittest.TestCase):
    def test_active_offsets_keep_anchor_and_offset_identity(self) -> None:
        visible = torch.tensor([False, True, False, True])
        anchors, offsets = active_offset_indices(visible, torch.tensor([0, 2, 4, 5]), n_offsets=3)
        self.assertTrue(torch.equal(anchors, torch.tensor([1, 1, 3, 3])))
        self.assertTrue(torch.equal(offsets, torch.tensor([0, 2, 1, 2])))

    def test_boolean_active_offsets_keep_anchor_and_offset_identity(self) -> None:
        visible = torch.tensor([False, True, False, True])
        active = torch.tensor([True, False, True, False, True, True])
        anchors, offsets = active_offset_indices(visible, active, n_offsets=3)
        self.assertTrue(torch.equal(anchors, torch.tensor([1, 1, 3, 3])))
        self.assertTrue(torch.equal(offsets, torch.tensor([0, 2, 1, 2])))

    def test_zero_residual_is_exact_anchor_inheritance(self) -> None:
        base = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        sidecar = OffsetSemanticResidualSidecar(base, n_offsets=3, device="cpu")
        visible = torch.tensor([True, False, True, False])
        active = torch.tensor([0, 2, 3, 5])
        expected_anchors, _ = active_offset_indices(visible, active, n_offsets=3)
        self.assertTrue(torch.equal(sidecar.offset_logits(visible, active), base[expected_anchors]))

    def test_residual_can_change_only_background_and_deck(self) -> None:
        base = torch.zeros((2, 5), dtype=torch.float32)
        sidecar = OffsetSemanticResidualSidecar(base, n_offsets=2, device="cpu")
        with torch.no_grad():
            sidecar.offset_background_deck_residual[1, 0] = torch.tensor([3.0, -2.0])
        output = sidecar.offset_logits(torch.tensor([False, True]), torch.tensor([0]))
        self.assertEqual(float(output[0, 0].detach()), 3.0)
        self.assertEqual(float(output[0, 1].detach()), -2.0)
        self.assertTrue(torch.equal(output[0, 2:], torch.zeros(3)))


if __name__ == "__main__":
    unittest.main()
