import unittest

import numpy as np

from src.improvedgs_responsibility_clone import alpha_conserving_partition


class ResponsibilityCloneTest(unittest.TestCase):
    def test_alpha_partition_is_exact_and_nontrivial(self):
        opacity = np.asarray([[0.1], [0.4], [0.8]], dtype=np.float64)
        fraction = np.asarray([[0.2], [0.5], [0.7]], dtype=np.float64)
        context, bridge = alpha_conserving_partition(opacity, fraction)
        combined = 1.0 - (1.0 - context) * (1.0 - bridge)
        np.testing.assert_allclose(combined, opacity, atol=1e-12, rtol=0)
        self.assertTrue(np.all(context > 0))
        self.assertTrue(np.all(bridge > 0))

    def test_torch_geometry_preserves_context_and_narrows_bridge(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional torch runtime unavailable")
        from src.improvedgs_responsibility_clone import responsibility_child_geometry

        xyz = torch.tensor([[1.0, 2.0, 3.0]])
        scales = torch.tensor([[4.0, 2.0, 1.0]])
        rotation_matrix = torch.eye(3).unsqueeze(0)
        rotation_raw = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        tangent = torch.tensor([[0.0, 1.0, 0.0]])
        context_xyz, context_scales, context_rotation, bridge_xyz, bridge_scales, _ = responsibility_child_geometry(
            xyz, scales, rotation_matrix, rotation_raw, tangent, transverse_multiplier=0.5
        )
        self.assertTrue(torch.equal(context_xyz, xyz))
        self.assertTrue(torch.equal(context_scales, scales))
        self.assertTrue(torch.equal(context_rotation, rotation_raw))
        self.assertTrue(torch.equal(bridge_xyz, xyz))
        self.assertEqual(float(bridge_scales.max()), 4.0)
        self.assertEqual(sorted(bridge_scales[0].tolist()), [0.5, 1.0, 4.0])


if __name__ == "__main__":
    unittest.main()
