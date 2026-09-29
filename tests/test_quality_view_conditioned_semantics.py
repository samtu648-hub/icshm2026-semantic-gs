import importlib
import unittest
from types import SimpleNamespace

from src.quality_view_conditioned_semantics import (
    build_directional_logit_field,
    direction_basis,
    directional_logits,
    directional_logit_sha256,
    render_directional_probabilities,
)


class ViewConditionedSemanticTests(unittest.TestCase):
    def torch(self):
        try:
            return importlib.import_module("torch")
        except ImportError:
            self.skipTest("optional torch runtime unavailable")

    def test_field_has_bias_and_three_direction_coefficients(self):
        torch = self.torch()
        field = build_directional_logit_field(7, device="cpu")
        self.assertEqual(tuple(field.shape), (7, 5, 4))
        self.assertTrue(torch.equal(field, torch.zeros_like(field)))
        self.assertEqual(
            directional_logit_sha256(field), directional_logit_sha256(field.clone())
        )

    def test_direction_basis_and_logits_change_with_camera(self):
        torch = self.torch()
        model = SimpleNamespace(get_xyz=torch.zeros((1, 3)))
        positive = SimpleNamespace(camera_center=torch.tensor([1.0, 0.0, 0.0]))
        negative = SimpleNamespace(camera_center=torch.tensor([-1.0, 0.0, 0.0]))
        self.assertTrue(torch.equal(direction_basis(positive, model.get_xyz), torch.tensor([[1.0, 1.0, 0.0, 0.0]])))
        field = build_directional_logit_field(1, device="cpu")
        with torch.no_grad():
            field[0, 2, 1] = 3.0
        self.assertGreater(
            float(directional_logits(field, positive, model)[0, 2].detach()),
            float(directional_logits(field, negative, model)[0, 2].detach()),
        )

    def test_renderer_conserves_probability_and_trains_direction_terms(self):
        torch = self.torch()
        model = SimpleNamespace(get_xyz=torch.zeros((2, 3)))
        view = SimpleNamespace(camera_center=torch.tensor([1.0, 0.0, 0.0]))
        weights = torch.tensor([0.25, 0.50])

        def render(_view, _model, _pipe, background, *, override_color):
            alpha = weights.sum()
            value = torch.einsum("n,nc->c", weights, override_color)
            value = value + (1.0 - alpha) * background
            return {"render": value[:, None, None]}

        field = build_directional_logit_field(2, device="cpu")
        probabilities = render_directional_probabilities(render, view, model, object(), field)
        torch.testing.assert_close(probabilities.sum(dim=0), torch.ones((1, 1)))
        (-probabilities[2, 0, 0].log()).backward()
        self.assertGreater(float(field.grad[:, :, 0].norm()), 0.0)
        self.assertGreater(float(field.grad[:, :, 1:].norm()), 0.0)

    def test_topology_mismatch_is_rejected(self):
        torch = self.torch()
        model = SimpleNamespace(get_xyz=torch.zeros((3, 3)))
        view = SimpleNamespace(camera_center=torch.zeros(3))
        field = build_directional_logit_field(2, device="cpu")
        with self.assertRaisesRegex(ValueError, "topology"):
            directional_logits(field, view, model)


if __name__ == "__main__":
    unittest.main()
