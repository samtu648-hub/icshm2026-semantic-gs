"""CPU-only unit tests for the SEM-003 evidence mathematics."""

from __future__ import annotations

import unittest
import json
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # Keep the repository-health interpreter dependency-free.
    torch = None


ROOT = Path(__file__).resolve().parents[1]


class TestSEM003DirichletEvidence(unittest.TestCase):
    def test_configs_disable_global_gate_and_freeze_train_only_policy(self) -> None:
        diagnostic = json.loads((ROOT / "experiments/runs/SEM-003-DIAG/config.yaml").read_text(encoding="utf-8"))
        control = json.loads((ROOT / "experiments/runs/SEM-003-CTRL/config.yaml").read_text(encoding="utf-8"))
        candidate = json.loads((ROOT / "experiments/runs/SEM-003/config.yaml").read_text(encoding="utf-8"))
        policy = diagnostic["conflict_diagnostic"]["eligibility_policy"]
        self.assertEqual(policy["minimum_support_views"], 3)
        self.assertEqual(policy["maximum_protected_fraction"], 0.05)
        self.assertEqual(control["conflict_diagnostic"]["eligibility_policy"], policy)
        self.assertEqual(candidate["conflict_diagnostic"]["eligibility_policy"], policy)
        self.assertFalse(control["conflict_regularizer"]["enabled"])
        self.assertTrue(candidate["conflict_regularizer"]["enabled"])
        for config in (control, candidate):
            self.assertNotIn("visibility_gate", config)
            self.assertEqual(config["frozen_rgb"]["pose_convention"], "CAMERA_FROM_WORLD")
            self.assertFalse(config["data"]["unlabeled_rgb_in_semantic_supervision"])

    @unittest.skipIf(torch is None, "torch is intentionally absent from the lightweight repository-health interpreter")
    def test_high_strength_conversion_preserves_sem001_probabilities(self) -> None:
        from src.scaffold_dirichlet_evidence import DirichletAnchorSidecar

        logits = torch.tensor([[2.0, -1.0, 0.5, 0.0, -2.0], [-2.0, 1.0, 3.0, 0.2, -0.5]])
        sidecar = DirichletAnchorSidecar(logits, initial_evidence_strength=1_000_000.0, device="cpu")
        self.assertLess(float((sidecar.predictive_probabilities() - torch.softmax(logits, dim=1)).abs().max().detach()), 5e-6)
        self.assertTrue(torch.equal(sidecar.predictive_probabilities().argmax(dim=1), logits.argmax(dim=1)))

    @unittest.skipIf(torch is None, "torch is intentionally absent from the lightweight repository-health interpreter")
    def test_training_only_policy_excludes_protected_evidence(self) -> None:
        from src.scaffold_dirichlet_evidence import ConflictPolicy, conflict_statistics

        evidence = torch.tensor([
            [10.0, 0.0, 10.0, 0.0, 0.0],  # clean bg/cable conflict
            [10.0, 0.0, 10.0, 2.0, 0.0],  # tower evidence means exclusion
            [19.0, 0.0, 1.0, 0.0, 0.0],   # low conflict
            [0.0, 0.0, 0.0, 0.0, 0.0],    # unsupported
        ])
        support = torch.tensor([4, 4, 4, 0])
        policy = ConflictPolicy(3, 0.05, 0.05, 0.05, 0.5)
        result = conflict_statistics(evidence, support, policy)
        mask = result["eligibility"]
        self.assertTrue(bool(mask[0]))
        self.assertFalse(bool(mask[1]))
        self.assertFalse(bool(mask[2]))
        self.assertFalse(bool(mask[3]))

    @unittest.skipIf(torch is None, "torch is intentionally absent from the lightweight repository-health interpreter")
    def test_regularizer_is_zero_without_selected_anchor(self) -> None:
        from src.scaffold_dirichlet_evidence import masked_concentration_regularizer

        strength = torch.tensor([4.0, 5.0], requires_grad=True)
        loss = masked_concentration_regularizer(strength, torch.tensor([False, False]), target_evidence_strength=1.0)
        loss.backward()
        self.assertTrue(torch.equal(strength.grad, torch.zeros_like(strength)))

    @unittest.skipIf(torch is None, "torch is intentionally absent from the lightweight repository-health interpreter")
    def test_regularizer_only_backpropagates_selected_anchor(self) -> None:
        from src.scaffold_dirichlet_evidence import masked_concentration_regularizer

        strength = torch.tensor([4.0, 5.0], requires_grad=True)
        loss = masked_concentration_regularizer(strength, torch.tensor([True, False]), target_evidence_strength=1.0)
        loss.backward()
        self.assertNotEqual(float(strength.grad[0]), 0.0)
        self.assertEqual(float(strength.grad[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
