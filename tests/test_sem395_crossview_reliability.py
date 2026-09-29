import inspect
import json
import unittest
from pathlib import Path

import numpy as np

from scripts.run_sem395_crossview_reliability import (
    finish_groups,
    semantic_rejection_count,
)
from src.camera_adapter import CameraFromWorld
from src.quality_backbone import CameraRequest
from src.quality_protocol import CLASSES
from src.semantic_crossview_reliability import (
    PAIR_STAGE_NAMES,
    backproject_pixels,
    bilinear_sample,
    combine_source_evidence,
    depth_proxy,
    evaluate_pair,
    nearest_pixel_index,
    project_world,
    replacement_counts,
    select_diagnostic_plan,
)

ROOT = Path(__file__).resolve().parents[1]


def camera(width=7, height=7, translation=None):
    return CameraRequest(
        width,
        height,
        np.array([[4.0, 0, width / 2], [0, 5.0, height / 2], [0, 0, 1.0]]),
        CameraFromWorld(
            np.eye(3),
            np.zeros(3)
            if translation is None
            else np.asarray(translation, dtype=float),
        ),
    )


def thresholds():
    return {
        "minimum_depth": 0.01,
        "alpha_minimum": 0.85,
        "depth_discontinuity_relative_maximum": 0.05,
        "relative_depth_maximum": 0.04,
        "cycle_reprojection_maximum_px": 1.0,
        "source_confidence_minimum": 0.80,
    }


class GeometryTests(unittest.TestCase):
    def test_edge_origin_round_trip_with_non_square_focal(self):
        view = camera()
        x = np.array([0.0, 2.0, 6.0])
        y = np.array([1.0, 3.0, 5.0])
        z = np.array([2.0, 4.0, 7.0])
        world = backproject_pixels(x, y, z, view)
        actual_x, actual_y, actual_z = project_world(world, view)
        np.testing.assert_allclose(actual_x, x, atol=1e-12)
        np.testing.assert_allclose(actual_y, y, atol=1e-12)
        np.testing.assert_allclose(actual_z, z, atol=1e-12)

    def test_depth_proxy_is_alpha_normalized_inverse_depth_reciprocal(self):
        inverse = np.array([[0.25, 0.0], [1.0, np.nan]], dtype=np.float32)
        alpha = np.array([[0.5, 1.0], [0.25, 1.0]], dtype=np.float32)
        proxy = depth_proxy(inverse, alpha)
        self.assertEqual(float(proxy[0, 0]), 2.0)
        self.assertEqual(float(proxy[1, 0]), 0.25)
        self.assertTrue(np.isnan(proxy[0, 1]))
        self.assertTrue(np.isnan(proxy[1, 1]))

    def test_bilinear_and_nearest_follow_pixel_centres(self):
        values = np.arange(9, dtype=np.float32).reshape(3, 3)
        sampled = bilinear_sample(values, np.array([0.5]), np.array([0.5]))
        np.testing.assert_allclose(sampled, [2.0])
        index = nearest_pixel_index(np.array([0.49, 0.51]), np.array([1.0, 1.0]), 3, 3)
        np.testing.assert_array_equal(index, [3, 4])
        invalid = nearest_pixel_index(
            np.array([np.nan, np.inf, -np.inf]), np.zeros(3), 3, 3
        )
        np.testing.assert_array_equal(invalid, [-1, -1, -1])

    def test_identity_pair_passes_only_complete_three_by_three_neighbourhood(self):
        z = np.full((7, 7), 2.0, dtype=np.float32)
        alpha = np.ones((7, 7), dtype=np.float32)
        probability = np.zeros((5, 7, 7), dtype=np.float32)
        probability[2] = 1.0
        result = evaluate_pair(
            z, alpha, z, alpha, probability, camera(), camera(), thresholds()
        )
        self.assertEqual(result["stage_counts"]["reliable"], 25)
        self.assertEqual(result["stage_counts"]["target_depth_discontinuity"], 24)
        self.assertEqual(sum(result["stage_counts"].values()), 49)
        np.testing.assert_array_equal(result["arrays"]["source_class"], 2)

    def test_forward_occlusion_is_rejected_before_semantics(self):
        target = np.full((7, 7), 2.0, dtype=np.float32)
        source = np.full((7, 7), 1.0, dtype=np.float32)
        alpha = np.ones((7, 7), dtype=np.float32)
        probability = np.zeros((5, 7, 7), dtype=np.float32)
        probability[1] = 1.0
        result = evaluate_pair(
            target, alpha, source, alpha, probability, camera(), camera(), thresholds()
        )
        self.assertEqual(result["stage_counts"]["forward_depth_conflict"], 25)
        self.assertEqual(result["stage_counts"]["reliable"], 0)
        self.assertEqual(
            result["occlusion_counts"]["projected_behind_source_surface"], 25
        )

    def test_source_depth_discontinuity_is_its_own_rejection_stage(self):
        target = np.full((7, 7), 2.0, dtype=np.float32)
        source = target.copy()
        source[3, 3] = 4.0
        alpha = np.ones((7, 7), dtype=np.float32)
        probability = np.zeros((5, 7, 7), dtype=np.float32)
        probability[2] = 1.0
        result = evaluate_pair(
            target, alpha, source, alpha, probability, camera(), camera(), thresholds()
        )
        self.assertEqual(result["stage_counts"]["source_depth_discontinuity"], 9)
        self.assertEqual(result["stage_counts"]["reliable"], 16)

    def test_low_source_confidence_has_disjoint_rejection_stage(self):
        z = np.full((7, 7), 2.0, dtype=np.float32)
        alpha = np.ones((7, 7), dtype=np.float32)
        probability = np.full((5, 7, 7), 0.2, dtype=np.float32)
        result = evaluate_pair(
            z, alpha, z, alpha, probability, camera(), camera(), thresholds()
        )
        self.assertEqual(result["stage_counts"]["source_low_confidence"], 25)
        self.assertEqual(result["stage_counts"]["reliable"], 0)


class EvidenceTests(unittest.TestCase):
    def test_two_sources_require_both_reliability_and_class_agreement(self):
        first = {
            "target_index": np.array([1, 3, 5], np.int32),
            "source_index": np.array([11, 13, 15], np.int32),
            "source_class": np.array([2, 1, 4], np.uint8),
            "source_confidence": np.array([0.9, 0.9, 0.9], np.float16),
            "target_alpha": np.array([1, 1, 1], np.float16),
            "source_alpha": np.array([1, 1, 1], np.float16),
        }
        second = {
            "target_index": np.array([1, 2, 3], np.int32),
            "source_index": np.array([21, 22, 23], np.int32),
            "source_class": np.array([2, 3, 4], np.uint8),
            "source_confidence": np.array([0.9, 0.9, 0.9], np.float16),
            "target_alpha": np.array([1, 1, 1], np.float16),
            "source_alpha": np.array([1, 1, 1], np.float16),
        }
        combined = combine_source_evidence(first, second, total_pixels=8)
        np.testing.assert_array_equal(combined["target_index"], [1])
        np.testing.assert_array_equal(combined["consensus_class"], [2])
        self.assertEqual(combined["counts"]["both_sources_reliable"], 2)
        self.assertEqual(combined["counts"]["source_semantic_disagreement"], 1)

    def test_replacement_counts_separate_corrections_and_damage(self):
        target = np.array([[0, 1, 2, 3]], np.uint8)
        baseline = np.array([[4, 1, 2, 0]], np.uint8)
        counts = replacement_counts(
            target,
            baseline,
            np.array([0, 1, 3], np.int32),
            np.array([0, 4, 3], np.uint8),
        )
        self.assertEqual(counts["corrected_pixels"], 2)
        self.assertEqual(counts["damaged_pixels"], 1)
        self.assertEqual(counts["net_correct_pixels"], 1)

    def test_empty_registered_group_has_no_invalid_metric_delta(self):
        empty = np.zeros((5, 5), dtype=np.int64)
        groups = {
            "target_opacity": {
                "low": {
                    "baseline_confusion": empty.copy(),
                    "counterfactual_confusion": empty.copy(),
                    "corrected": 0,
                    "damaged": 0,
                }
            }
        }
        finished = finish_groups(groups)
        self.assertIsNone(finished["target_opacity"]["low"]["five_class_miou_delta_pp"])

    def test_semantic_rejections_use_sealed_evidence_summaries(self):
        count = semantic_rejection_count(
            {"source_low_confidence": 7},
            [
                {"evidence_counts": {"source_semantic_disagreement": 3}},
                {"evidence_counts": {"source_semantic_disagreement": 5}},
            ],
        )
        self.assertEqual(count, 15)


class ProtocolTests(unittest.TestCase):
    def test_selection_api_cannot_accept_labels_or_scores(self):
        parameters = set(inspect.signature(select_diagnostic_plan).parameters)
        self.assertFalse(
            parameters
            & {"labels", "ground_truth", "prediction", "probability", "errors"}
        )

    def test_selection_is_deterministic_and_covers_segments(self):
        names = [f"{index:03d}.png" for index in range(1, 7)]
        nodes = [
            {"name": name, "center": [float(index), 0.0, 0.0]}
            for index, name in enumerate(names)
        ]
        edges = []
        adjacency = {name: [] for name in names}
        for i, first in enumerate(names):
            for j, second in enumerate(names):
                if i >= j:
                    continue
                edge = {
                    "source": first,
                    "target": second,
                    "shared_point3d": 1000 - 10 * abs(i - j),
                    "center_distance": 0.1 + 0.1 * abs(i - j),
                    "optical_axis_angle_deg": 2.0,
                }
                edges.append(edge)
                adjacency[first].append(edge)
                adjacency[second].append(edge)
        graph = {"nodes": nodes, "edges": edges, "adjacency": adjacency}
        segments = {
            "segments": [
                {"segment_id": "seg-1", "images": names[:3]},
                {"segment_id": "seg-2", "images": names[3:]},
            ]
        }

        def selected():
            return select_diagnostic_plan(
                graph,
                segments,
                names,
                seed=395,
                target_count=2,
                sources_per_target=2,
                source_minimum_shared_points=64,
                source_maximum_axis_angle_deg=45.0,
                source_minimum_center_distance=0.05,
                source_maximum_center_distance=3.0,
                second_source_minimum_separation=0.25,
            )

        first = selected()
        second = selected()
        self.assertEqual(first, second)
        self.assertEqual(first["covered_segments"], ["seg-1", "seg-2"])
        self.assertEqual(first["pair_count"], 4)

    def test_registered_config_fixes_access_and_class_name(self):
        config = json.loads(
            (
                ROOT
                / "experiments/runs/SEM-395-RGB150-CROSSVIEW-RELIABILITY-DIAG/config.yaml"
            ).read_text()
        )
        self.assertEqual(config["access_contract"]["target_count"], 24)
        self.assertEqual(config["access_contract"]["maximum_pairs"], 48)
        self.assertFalse(config["access_contract"]["holdout_label_access"])
        self.assertEqual(config["reliability_filter"]["schemes_tested"], 1)
        self.assertEqual(CLASSES[2], "stay_cable")
        self.assertEqual(len(PAIR_STAGE_NAMES), len(set(PAIR_STAGE_NAMES)))


if __name__ == "__main__":
    unittest.main()
