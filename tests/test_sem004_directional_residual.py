"""Dependency-light tests for the SEM-004 restricted direction residual."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # Keep the lightweight repository-health interpreter dependency-free.
    torch = None

if torch is not None:
    from src.scaffold_directional_residual import DirectionalResidualSidecar, load_direction_anchor_indices


@unittest.skipIf(torch is None, "torch is intentionally absent from the lightweight repository-health interpreter")
class TestDirectionalResidualSidecar(unittest.TestCase):
    def test_zero_residual_is_exact_and_only_selected_coefficients_train(self) -> None:
        logits = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0], [0.5, 0.0, -0.5, 1.0, -1.0]],
            dtype=torch.float32,
        )
        sidecar = DirectionalResidualSidecar(logits, [1], cap=0.75, device="cpu")
        visible = torch.tensor([True, True, False])
        anchors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        adjusted, stats = sidecar.adjusted_visible_logits(anchors, visible, torch.zeros(3))
        self.assertTrue(torch.equal(adjusted, logits[:2]))
        self.assertTrue(torch.equal(stats["delta"], torch.zeros(2)))
        self.assertFalse(sidecar.base_logits.requires_grad)
        self.assertEqual(tuple(sidecar.direction_coefficients.shape), (1, 3))

        adjusted[:, 2].sum().backward()
        self.assertIsNotNone(sidecar.direction_coefficients.grad)
        self.assertTrue(torch.isfinite(sidecar.direction_coefficients.grad).all())

    def test_residual_is_bounded_and_protected_logits_are_directly_unchanged(self) -> None:
        logits = torch.randn(4, 5)
        sidecar = DirectionalResidualSidecar(logits, [0, 2], cap=0.5, device="cpu")
        with torch.no_grad():
            sidecar.direction_coefficients.fill_(100.0)
        visible = torch.ones(4, dtype=torch.bool)
        anchors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
        adjusted, stats = sidecar.adjusted_visible_logits(anchors, visible, torch.zeros(3))
        self.assertLessEqual(float(stats["delta"].detach().abs().max()), 0.5)
        self.assertTrue(torch.equal(adjusted[:, 1], logits[:, 1]))
        self.assertTrue(torch.equal(adjusted[:, 3], logits[:, 3]))
        self.assertTrue(torch.equal(adjusted[:, 4], logits[:, 4]))
        self.assertEqual(float(stats["delta"][1].detach()), 0.0)
        self.assertEqual(float(stats["delta"][3].detach()), 0.0)

    def test_direction_definition_is_anchor_minus_camera(self) -> None:
        logits = torch.zeros(1, 5)
        sidecar = DirectionalResidualSidecar(logits, [0], cap=1.0, device="cpu")
        with torch.no_grad():
            sidecar.direction_coefficients[0] = torch.tensor([1.0, 0.0, 0.0])
        visible = torch.ones(1, dtype=torch.bool)
        adjusted, stats = sidecar.adjusted_visible_logits(
            torch.tensor([[2.0, 0.0, 0.0]]), visible, torch.tensor([[1.0, 0.0, 0.0]])
        )
        expected = torch.tanh(torch.tensor(1.0))
        self.assertTrue(torch.allclose(stats["delta"], expected.reshape(1)))
        self.assertTrue(torch.allclose(adjusted[0, 0], -expected))
        self.assertTrue(torch.allclose(adjusted[0, 2], expected))

    def test_csv_mask_must_be_sorted_unique_and_exact_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anchors.csv"
            path.write_text("anchor_index,support_views\n1,3\n4,5\n", encoding="utf-8")
            self.assertEqual(load_direction_anchor_indices(path, expected_count=2, anchor_count=5), [1, 4])
            with self.assertRaises(ValueError):
                load_direction_anchor_indices(path, expected_count=3, anchor_count=5)


if __name__ == "__main__":
    unittest.main()
