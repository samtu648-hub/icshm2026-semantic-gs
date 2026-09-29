from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geometry_prior.camera_io import ColmapImage  # noqa: E402
from semantic.cable_geometry import centerline_metrics, extract_centerline, fit_piecewise_lines  # noqa: E402
from semantic.cable_triangulation import (  # noqa: E402
    CableObservation,
    fundamental_matrix,
    pinhole_matrix,
    pixel_rays_world,
    reproject,
    robust_triangulate,
    symmetric_epipolar_distance,
)


def _image(name: str, center: tuple[float, float, float]) -> ColmapImage:
    center_array = np.asarray(center, dtype=np.float64)
    return ColmapImage(
        image_id=int(Path(name).stem),
        name=name,
        camera_id=1,
        qvec=np.array([1.0, 0.0, 0.0, 0.0]),
        tvec=-center_array,
        points2d=tuple(),
    )


class CableGeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.intrinsic = pinhole_matrix(900.0, 660.0, 494.5)

    def test_camera_from_world_cam2world_and_skeleton_pixel_ray(self) -> None:
        image = _image("001.png", (1.0, -2.0, 0.5))
        self.assertTrue(np.allclose(image.center, [1.0, -2.0, 0.5]))
        point = np.array([1.4, -1.8, 5.5])
        pixel, depth = reproject(point, image, self.intrinsic)
        origins, rays = pixel_rays_world(pixel[None], self.intrinsic, image)
        expected = point - image.center
        expected /= np.linalg.norm(expected)
        self.assertGreater(depth, 0)
        self.assertTrue(np.allclose(origins[0], image.center, atol=1e-12))
        self.assertTrue(np.allclose(rays[0], expected, atol=1e-10))

    def test_epipolar_distance_and_three_view_positive_depth_triangulation(self) -> None:
        images = [_image("001.png", (-1.0, 0.0, 0.0)), _image("002.png", (0.0, 0.0, 0.0)), _image("003.png", (1.0, 0.0, 0.0))]
        point = np.array([0.2, 0.15, 6.0])
        observations = []
        pixels = []
        for image in images:
            pixel, depth = reproject(point, image, self.intrinsic)
            self.assertGreater(depth, 0)
            pixels.append(pixel)
            observations.append(CableObservation(image.name, pixel, 3.0, image, "synthetic"))
        fundamental = fundamental_matrix(images[0], images[2], self.intrinsic)
        distance = symmetric_epipolar_distance(pixels[0][None], pixels[2][None], fundamental)
        self.assertLess(float(distance[0]), 1e-8)
        candidate = robust_triangulate(observations, self.intrinsic, 0.3, 5.0, 1.0, 0.1, 3)
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(len(candidate.observations), 3)
        self.assertTrue(np.allclose(candidate.xyz, point, atol=1e-8))
        self.assertTrue(np.all(candidate.reprojection_errors_px < 1e-7))

    def test_centerline_components_width_endpoints_and_metrics(self) -> None:
        label = np.zeros((80, 120), dtype=np.uint8)
        cv2.line(label, (10, 15), (105, 65), 2, 7)
        label[2, 2] = 2
        config = {
            "close_kernel_px": 3,
            "close_iterations": 1,
            "open_kernel_px": 0,
            "open_iterations": 0,
            "minimum_component_area_px": 16,
            "connectivity": 8,
            "minimum_skeleton_component_px": 8,
            "endpoint_neighbor_count": 1,
            "branch_neighbor_count_minimum": 3,
            "sample_grid_spacing_px": 8,
            "maximum_samples_per_image": 600,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.png"
            Image.fromarray(label, mode="L").save(path)
            result = extract_centerline(path, 2, config)
        self.assertEqual(result.component_count, 1)
        self.assertGreater(np.count_nonzero(result.skeleton), 80)
        self.assertGreaterEqual(len(result.endpoints_xy), 2)
        self.assertGreater(float(np.median(result.sample_widths_px)), 4.0)
        metrics = centerline_metrics(result.skeleton, result.skeleton, (2, 4, 8), 0.02)
        self.assertEqual(metrics["exact_iou"], 1.0)
        self.assertTrue(metrics["covered_at_2px"])

    def test_piecewise_curve_tangent_and_support(self) -> None:
        x = np.linspace(0.0, 1.2, 25)
        points = np.column_stack([x, 2.0 * x, -0.5 * x])
        confidences = np.ones(len(points))
        support = [("001.png", "002.png", "003.png") for _ in points]
        widths = np.full(len(points), 4.0)
        config = {
            "line_inlier_radius_official": 0.06,
            "minimum_inlier_candidates": 6,
            "ransac_iterations_per_curve": 300,
            "minimum_curve_length_official": 0.3,
            "maximum_along_curve_gap_official": 0.3,
            "minimum_occupied_bin_fraction": 0.5,
            "endpoint_quantiles": [0.0, 1.0],
            "maximum_curves": 4,
        }
        curves = fit_piecewise_lines(points, confidences, support, widths, config, 7)
        self.assertEqual(len(curves), 1)
        expected = np.array([1.0, 2.0, -0.5])
        expected /= np.linalg.norm(expected)
        self.assertGreater(abs(float(np.dot(curves[0].tangent, expected))), 0.999999)
        self.assertEqual(curves[0].support_views, ("001.png", "002.png", "003.png"))
        self.assertGreater(curves[0].length, 1.0)

    def test_geometry_split_has_no_validation_or_unlabeled_leakage(self) -> None:
        split = json.loads((ROOT / "experiments/splits/sem_060_cable_geometry_v1.json").read_text(encoding="utf-8"))
        fit, check, guard = map(set, (split["geometry_fit"], split["geometry_check"], split["local_guard_band"]))
        prohibited = set(split["fixed_validation_read_only_names"]) | set(split["parent_guard_never_used"]) | set(split["unlabeled_never_used"])
        self.assertEqual((len(fit), len(check), len(guard)), (177, 40, 2))
        self.assertFalse(fit & check or fit & guard or check & guard)
        self.assertFalse((fit | check | guard) & prohibited)
        self.assertEqual(split["checks"]["nearby_camera_graph_edges_fit_to_check"], 0)


if __name__ == "__main__":
    unittest.main()
