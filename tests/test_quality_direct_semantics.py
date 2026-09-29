import importlib
import unittest
from types import SimpleNamespace

from src.quality_direct_semantics import (
    build_direct_logit_field,
    direct_logit_sha256,
    render_direct_probabilities,
)


class DirectSemanticTests(unittest.TestCase):
    def torch(self):
        try:
            return importlib.import_module("torch")
        except ImportError:
            self.skipTest("optional torch runtime unavailable")

    def test_field_is_exactly_five_logits_per_gaussian(self):
        torch = self.torch()
        field = build_direct_logit_field(7, device="cpu")
        self.assertEqual(tuple(field.shape), (7, 5))
        self.assertTrue(torch.equal(field, torch.zeros_like(field)))
        self.assertEqual(direct_logit_sha256(field), direct_logit_sha256(field.clone()))

    def test_renderer_conserves_probability_and_assigns_empty_mass_to_background(self):
        torch = self.torch()
        model = SimpleNamespace(get_xyz=torch.zeros((2, 3)))
        weights = torch.tensor([0.25, 0.50])

        def render(_view, _model, _pipe, background, *, override_color):
            alpha = weights.sum()
            value = torch.einsum("n,nc->c", weights, override_color)
            value = value + (1.0 - alpha) * background
            return {"render": value[:, None, None]}

        logits = build_direct_logit_field(2, device="cpu")
        probabilities = render_direct_probabilities(render, object(), model, object(), logits)
        torch.testing.assert_close(probabilities.sum(dim=0), torch.ones((1, 1)))
        self.assertGreater(
            float(probabilities[0, 0, 0].detach()),
            float(probabilities[1, 0, 0].detach()),
        )
        loss = -probabilities[2, 0, 0].log()
        loss.backward()
        self.assertGreater(float(logits.grad.norm()), 0.0)

    def test_topology_mismatch_is_rejected(self):
        torch = self.torch()
        model = SimpleNamespace(get_xyz=torch.zeros((3, 3)))
        logits = build_direct_logit_field(2, device="cpu")
        with self.assertRaisesRegex(ValueError, "topology"):
            render_direct_probabilities(None, object(), model, object(), logits)


if __name__ == "__main__":
    unittest.main()
