# pyright: reportPossiblyUnboundVariable=false

import math
import sys
import types
import unittest
from unittest import mock


try:
    import torch  # type: ignore[import-not-found]

    from src.standard_long_axis_split import (
        install,
        long_axis_children,
        projected_covariance,
    )

    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, "Optional Torch runtime absent")
class LongAxisSplitTests(unittest.TestCase):
    def test_rotated_long_axis_positions_scales_and_opacity(self):
        xyz = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 1.0, 4.0]])
        scales = torch.tensor([[1.0, 4.0, 2.0], [3.0, 1.0, 2.0]])
        rotations = torch.tensor(
            [
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            ]
        )
        opacity = torch.tensor([[0.8], [0.5]])

        child_xyz, child_scales, child_opacity, axes = long_axis_children(
            xyz, scales, rotations, opacity
        )

        self.assertTrue(torch.equal(axes, torch.tensor([1, 0])))
        expected_offset = torch.tensor([[-5.4, 0.0, 0.0], [4.05, 0.0, 0.0]])
        self.assertTrue(torch.allclose(child_xyz[:2], xyz + expected_offset))
        self.assertTrue(torch.allclose(child_xyz[2:], xyz - expected_offset))
        minor = math.sqrt(1.0 - 0.45**2)
        expected_scales = torch.tensor(
            [[minor, 2.2, 2.0 * minor], [1.65, minor, 2.0 * minor]]
        )
        self.assertTrue(torch.allclose(child_scales[:2], expected_scales))
        self.assertTrue(torch.allclose(child_scales[2:], expected_scales))
        self.assertTrue(
            torch.allclose(child_opacity, torch.tensor([[0.48], [0.3], [0.48], [0.3]]))
        )
        self.assertTrue(torch.isfinite(child_xyz).all())
        self.assertTrue(torch.isfinite(child_scales).all())
        self.assertTrue((child_scales > 0).all())

    def test_projected_coverage_contract(self):
        parent_scales = torch.tensor([[4.0, 1.0, 0.5]])
        rotation = torch.eye(3).unsqueeze(0)
        opacity = torch.tensor([[0.8]])
        child_xyz, child_scales, _, _ = long_axis_children(
            torch.zeros((1, 3)), parent_scales, rotation, opacity
        )
        jacobian = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        parent_cov = projected_covariance(parent_scales, rotation, jacobian)[0]
        child_cov = projected_covariance(child_scales[:1], rotation, jacobian)[0]

        major_ratio = torch.sqrt(child_cov[0, 0] / parent_cov[0, 0]).item()
        minor_ratio = torch.sqrt(child_cov[1, 1] / parent_cov[1, 1]).item()
        pair_outer_extent = (child_xyz[0, 0].abs() + 3.0 * child_scales[0, 0]) / (
            3.0 * parent_scales[0, 0]
        )
        self.assertAlmostEqual(major_ratio, 0.55, places=6)
        self.assertAlmostEqual(minor_ratio, math.sqrt(1.0 - 0.45**2), places=6)
        self.assertAlmostEqual(pair_outer_extent.item(), 1.0, places=6)
        self.assertLess(torch.linalg.det(child_cov), torch.linalg.det(parent_cov))

    def test_disabled_install_keeps_native_method(self):
        class Model:
            def densify_and_split(self):
                return "native"

        original = Model.densify_and_split
        state = install(Model, enabled=False)
        self.assertIs(Model.densify_and_split, original)
        self.assertEqual(state["method"], "native")

    def test_hook_preserves_lifecycle_attributes_optimizer_state_and_rng(self):
        class Model:
            percent_dense = 0.1

            def __init__(self):
                self._xyz = torch.tensor(
                    [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]
                )
                self._scaling = torch.log(
                    torch.tensor([[4.0, 1.0, 0.5], [2.0, 1.0, 0.5], [0.5, 0.5, 0.5]])
                )
                self._opacity = torch.logit(torch.tensor([[0.8], [0.7], [0.6]]))
                self._rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3)
                self._features_dc = torch.arange(6.0).reshape(3, 1, 2)
                self._features_rest = torch.arange(9.0).reshape(3, 1, 3)
                self.tmp_radii = torch.tensor([1.0, 2.0, 3.0])
                self.optimizer_exp_avg = torch.tensor([[1.0], [2.0], [3.0]])

            @property
            def get_xyz(self):
                return self._xyz

            @property
            def get_scaling(self):
                return torch.exp(self._scaling)

            @property
            def get_opacity(self):
                return torch.sigmoid(self._opacity)

            scaling_inverse_activation = staticmethod(torch.log)
            inverse_opacity_activation = staticmethod(torch.logit)

            def densify_and_split(self, grads, threshold, extent, N=2):
                raise AssertionError("Native method should be replaced")

            def densification_postfix(
                self,
                xyz,
                features_dc,
                features_rest,
                opacity,
                scaling,
                rotation,
                tmp_radii,
            ):
                child_count = xyz.shape[0]
                self._xyz = torch.cat((self._xyz, xyz))
                self._features_dc = torch.cat((self._features_dc, features_dc))
                self._features_rest = torch.cat((self._features_rest, features_rest))
                self._opacity = torch.cat((self._opacity, opacity))
                self._scaling = torch.cat((self._scaling, scaling))
                self._rotation = torch.cat((self._rotation, rotation))
                self.tmp_radii = torch.cat((self.tmp_radii, tmp_radii))
                self.optimizer_exp_avg = torch.cat(
                    (self.optimizer_exp_avg, torch.zeros((child_count, 1)))
                )

            def prune_points(self, mask):
                keep = ~mask
                for name in (
                    "_xyz",
                    "_features_dc",
                    "_features_rest",
                    "_opacity",
                    "_scaling",
                    "_rotation",
                    "tmp_radii",
                    "optimizer_exp_avg",
                ):
                    setattr(self, name, getattr(self, name)[keep])

        general_utils = types.ModuleType("utils.general_utils")
        setattr(
            general_utils,
            "build_rotation",
            lambda value: torch.eye(3).repeat(value.shape[0], 1, 1),
        )
        utils = types.ModuleType("utils")
        setattr(utils, "general_utils", general_utils)
        with mock.patch.dict(
            sys.modules,
            {"utils": utils, "utils.general_utils": general_utils},
        ):
            state = install(Model, enabled=True)
            model = Model()
            torch.manual_seed(17)
            model.densify_and_split(torch.tensor([[1.0], [0.0], [0.0]]), 0.5, 10.0)
            actual_rng = torch.get_rng_state()

        torch.manual_seed(17)
        torch.normal(
            mean=torch.zeros((2, 3)),
            std=torch.tensor([[4.0, 1.0, 0.5], [4.0, 1.0, 0.5]]),
        )
        expected_rng = torch.get_rng_state()

        self.assertTrue(torch.equal(actual_rng, expected_rng))
        self.assertEqual(model.get_xyz.shape[0], 4)
        self.assertTrue(torch.equal(model.get_xyz[:2, 0], torch.tensor([10.0, 20.0])))
        self.assertTrue(torch.allclose(model.get_xyz[2:, 0], torch.tensor([5.4, -5.4])))
        self.assertTrue(
            torch.allclose(model.get_opacity[2:], torch.tensor([[0.48], [0.48]]))
        )
        self.assertTrue(
            torch.equal(
                model._features_dc[2:], torch.tensor([[[0.0, 1.0]], [[0.0, 1.0]]])
            )
        )
        self.assertTrue(
            torch.equal(model._rotation[2:], torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2))
        )
        self.assertTrue(
            torch.equal(model.tmp_radii, torch.tensor([2.0, 3.0, 1.0, 1.0]))
        )
        self.assertTrue(
            torch.equal(
                model.optimizer_exp_avg, torch.tensor([[2.0], [3.0], [0.0], [0.0]])
            )
        )
        self.assertEqual(state["calls"], 1)
        self.assertEqual(state["selected_parents"], 1)


if __name__ == "__main__":
    unittest.main()
