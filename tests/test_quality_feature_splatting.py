import unittest
import importlib

import numpy as np

from src.camera_adapter import CameraFromWorld
from src.quality_backbone import CameraRequest
from src.quality_feature_splatting import (
    build_feature_splatting_field,
    crop_and_flip,
    crop_camera,
    patch_cosine_loss,
    restore_signed_composite,
    signed_feature_affine,
)


class FeatureSplattingTests(unittest.TestCase):
    def torch(self):
        try:
            return importlib.import_module("torch")
        except ImportError:
            self.skipTest("optional torch runtime unavailable")

    def test_crop_camera_preserves_pixel_projection(self):
        camera = CameraRequest(
            1320,
            989,
            np.array([[900.0, 0, 650.0], [0, 910.0, 480.0], [0, 0, 1.0]]),
            CameraFromWorld(np.eye(3), np.zeros(3)),
        )
        cropped = crop_camera(camera, top=101, left=203, size=768)
        self.assertEqual((cropped.width, cropped.height), (768, 768))
        self.assertEqual(cropped.matrix[0, 2], 447)
        self.assertEqual(cropped.matrix[1, 2], 379)
        self.assertEqual(cropped.matrix[0, 0], camera.matrix[0, 0])

    def test_crop_and_flip_matches_registered_order(self):
        image = np.arange(30).reshape(5, 6)
        sample = {"top": 1, "left": 2, "crop": 3, "flip": True}
        np.testing.assert_array_equal(
            crop_and_flip(image, sample), image[1:4, 2:5][:, ::-1]
        )

    def test_signed_affine_restores_alpha_composite_and_zero_gradient(self):
        torch = self.torch()

        values = torch.zeros((4, 2), requires_grad=True)
        normalized, lower, scale = signed_feature_affine(values)
        weights = torch.tensor([0.2, 0.3, 0.1, 0.0])
        rendered = torch.einsum("n,nc->c", weights, normalized)[:, None, None]
        alpha = weights.sum().reshape(1, 1, 1)
        restored = restore_signed_composite(rendered, alpha, lower, scale)
        self.assertTrue(torch.equal(restored, torch.zeros_like(restored)))
        restored.sum().backward()
        torch.testing.assert_close(values.grad, weights[:, None].expand_as(values))

    def test_feature_field_shape_and_shared_heads(self):
        torch = self.torch()

        torch.manual_seed(7)
        field = build_feature_splatting_field(11, device="cpu")
        self.assertEqual(tuple(field.latent.shape), (11, 32))
        logits, teacher = field.decode(torch.zeros((32, 4, 5)))
        self.assertEqual(tuple(logits.shape), (1, 5, 4, 5))
        self.assertEqual(tuple(teacher.shape), (1, 768, 4, 5))

    def test_patch_cosine_identity(self):
        torch = self.torch()

        value = torch.randn((1, 8, 3, 4))
        self.assertAlmostEqual(float(patch_cosine_loss(value, value)), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
