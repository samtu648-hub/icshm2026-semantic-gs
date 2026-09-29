import importlib.util
from types import SimpleNamespace
import unittest

from src.quality_backbone import ModelIdentity
from src.quality_semantic_fields import PrimitiveFeatureField, PrimitiveSemanticField


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch runtime")
class QualitySemanticFieldTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.renderer = SimpleNamespace(identity=ModelIdentity("pgsr", "a" * 64, "b" * 64),
            field_positions=torch.zeros(2, 3))
        weights = torch.tensor([[[.3, 0.]], [[.2, 0.]]])
        self.renderer.render_channels = lambda camera, values: torch.einsum("nhw,nc->chw", weights, values)

    def test_static_background_residual_and_gradient(self):
        field = PrimitiveSemanticField(self.renderer)
        p = field.render_probabilities(self.renderer, None)
        self.torch.testing.assert_close(p.sum(0), self.torch.ones(1, 2))
        self.assertAlmostEqual(p[0, 0, 0].item(), .6, places=6)
        self.assertEqual(p[0, 0, 1].item(), 1.)
        (-p[1, 0, 0].log()).backward()
        self.assertGreater(field.logits.grad.abs().sum().item(), 0)

    def test_attachment_rejects_other_checkpoint_and_corrupt_shape(self):
        field = PrimitiveSemanticField(self.renderer)
        payload = field.state_dict()
        payload["metadata"]["checkpoint_sha256"] = "c" * 64
        with self.assertRaises(ValueError):
            field.load_state_dict(payload, self.renderer)
        payload = field.state_dict()
        payload["logits"] = self.torch.zeros(1, 5)
        with self.assertRaises(ValueError):
            field.load_state_dict(payload, self.renderer)

    def test_feature_capacity_and_trainable_head(self):
        field = PrimitiveFeatureField(self.renderer)
        self.assertEqual(tuple(field.features.shape), (2, 16))
        p = field.render_probabilities(self.renderer, None)
        self.assertEqual(tuple(p.shape), (5, 1, 2))
        self.torch.testing.assert_close(p.sum(0), self.torch.ones(1, 2))
        (-p[1].log().mean()).backward()
        self.assertIsNotNone(field.features.grad)
        self.assertTrue(all(p.grad is not None for p in field.head.parameters()))
        clone = PrimitiveFeatureField(self.renderer)
        clone.load_state_dict(field.state_dict(), self.renderer)
        self.torch.testing.assert_close(p, clone.render_probabilities(self.renderer, None))
