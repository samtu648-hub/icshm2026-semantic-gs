import unittest
import numpy as np

from src.camera_adapter import CameraFromWorld, Sim3
from src.quality_backbone import CameraRequest, ModelIdentity, NativeRender, array_signature, render_official, validate_attachment
from src.quality_protocol import CLASSES


class StubRenderer:
    """Contract test only; never a candidate-backbone quality result."""
    identity = ModelIdentity("pgsr", "a" * 64, "b" * 64)
    official_to_model = Sim3(1., np.eye(3), np.zeros(3))

    def render_pinhole(self, camera):
        assert camera.model == "PINHOLE" and camera.radial_k == 0
        self.camera = camera
        rgb = np.full((camera.height, camera.width, 3), .5, np.float32)
        p = np.zeros((5, camera.height, camera.width), np.float32)
        p[1] = 1
        return NativeRender(rgb, p)


class QualityBackboneTests(unittest.TestCase):
    def camera(self, **kw):
        return CameraRequest(8, 6, np.array([[10., 0, 4], [0, 10, 3], [0, 0, 1]]), CameraFromWorld(np.eye(3), np.zeros(3)), **kw)

    def test_native_output_and_class_mapping(self):
        r = render_official(StubRenderer(), self.camera())
        self.assertEqual(r["rgb"].dtype, np.uint8)
        np.testing.assert_array_equal(r["rgb"], 128)
        np.testing.assert_array_equal(r["semantic"], 1)
        self.assertEqual(r["probabilities"].shape, (5, 6, 8))

    def test_radial_zero_matches_pinhole(self):
        a = render_official(StubRenderer(), self.camera())
        b = render_official(StubRenderer(), self.camera(model="SIMPLE_RADIAL"))
        np.testing.assert_array_equal(a["semantic"], b["semantic"])
        np.testing.assert_array_equal(a["rgb"], b["rgb"])

    def test_nonzero_radial_preserves_simplex(self):
        r = render_official(StubRenderer(), self.camera(model="SIMPLE_RADIAL", radial_k=.2))
        np.testing.assert_allclose(r["probabilities"].sum(0), 1, atol=1e-6)

    def test_official_similarity_is_applied(self):
        backend = StubRenderer()
        backend.official_to_model = Sim3(2., np.eye(3), np.array([1., 2., 3.]))
        render_official(backend, self.camera())
        np.testing.assert_allclose(backend.camera.pose.center, [1, 2, 3])

    def test_wrong_camera_rejected(self):
        with self.assertRaises(ValueError):
            self.camera(model="FISHEYE")
        with self.assertRaises(ValueError):
            self.camera(radial_k=.1)

    def test_wrong_sidecar_rejected(self):
        identity = StubRenderer.identity
        metadata = dict(schema_version=1, backbone=identity.backbone, checkpoint_sha256=identity.checkpoint_sha256,
                        topology_sha256=identity.topology_sha256, classes=list(CLASSES), geometry_frozen=True, pixel_space="PINHOLE")
        validate_attachment(metadata, identity)
        for key, wrong in [("backbone", "sof"), ("checkpoint_sha256", "c" * 64), ("topology_sha256", "d" * 64), ("classes", ["x"] * 5), ("geometry_frozen", False)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_attachment(dict(metadata, **{key: wrong}), identity)

    def test_invalid_native_probabilities_rejected(self):
        native = StubRenderer().render_pinhole(self.camera())
        native.probabilities[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            native.validate(self.camera())

    def test_signature_includes_shape_and_dtype(self):
        a = np.arange(6, dtype=np.float32)
        original = array_signature({"xyz": a})
        self.assertNotEqual(original, array_signature({"xyz": a.reshape(2, 3)}))
        self.assertNotEqual(original, array_signature({"xyz": a.astype(np.float64)}))
        self.assertNotEqual(original, array_signature({"xyz": a + 1}))


if __name__ == "__main__":
    unittest.main()
