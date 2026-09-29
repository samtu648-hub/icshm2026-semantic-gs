import unittest

import numpy as np

from src.geometry_prior.camera_io import ColmapCamera, ColmapImage
from src.semantic_multiview_deck import (
    DepthObservation,
    accumulate_source_votes,
    add_deck_with_protection,
    confirm_deck_points,
    deck_points_from_view,
    depth_consistent,
    project_points_to_target_mask,
    strong_neighbours,
)


def camera() -> ColmapCamera:
    return ColmapCamera(1, "PINHOLE", 5, 5, (2.0, 2.0, 2.0, 2.0))


def image(name: str = "a.png") -> ColmapImage:
    return ColmapImage(
        1,
        name,
        1,
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        (),
    )


class SemanticMultiviewDeckTest(unittest.TestCase):
    def test_backproject_confirm_and_project_identity(self) -> None:
        depth = np.full((5, 5), 2.0, dtype=np.float32)
        observation = DepthObservation(depth, np.ones((5, 5), dtype=np.float32))
        deck = np.zeros((5, 5), dtype=bool)
        deck[2, 2] = True
        points = deck_points_from_view(
            deck,
            observation,
            camera=camera(),
            image=image(),
            stride=1,
            minimum_foreground_mass=0.5,
        )
        self.assertEqual(points.shape, (1, 3))
        keep, support = confirm_deck_points(
            points,
            [(deck, observation, camera(), image("b.png"))],
            minimum_confirming_views=1,
            minimum_foreground_mass=0.5,
            relative_tolerance=0.05,
        )
        self.assertTrue(keep[0])
        self.assertEqual(int(support[0]), 1)
        projected, stats = project_points_to_target_mask(
            points[keep],
            target_observation=observation,
            target_camera=camera(),
            target_image=image("target.png"),
            minimum_foreground_mass=0.5,
            relative_tolerance=0.05,
            splat_radius=0,
        )
        self.assertTrue(projected[2, 2])
        self.assertEqual(stats["visible_points"], 1)

    def test_depth_consistency_is_symmetric(self) -> None:
        projected = np.asarray([1.0, 1.0, np.nan])
        observed = np.asarray([1.04, 1.06, 1.0])
        self.assertEqual(
            depth_consistent(projected, observed, relative_tolerance=0.05).tolist(),
            [True, False, False],
        )

    def test_votes_and_protection_only_add_background(self) -> None:
        votes = accumulate_source_votes(
            [
                np.asarray([[True, True], [False, False]]),
                np.asarray([[True, False], [True, False]]),
            ]
        )
        baseline = np.asarray([[0, 2], [0, 1]], dtype=np.uint8)
        eligible = np.asarray([[True, True], [False, True]])
        result, additions = add_deck_with_protection(
            baseline, eligible, votes, minimum_source_votes=2
        )
        np.testing.assert_array_equal(result, np.asarray([[1, 2], [0, 1]], dtype=np.uint8))
        np.testing.assert_array_equal(additions, np.asarray([[True, False], [False, False]]))

    def test_strong_neighbours_respects_allowlist_and_order(self) -> None:
        graph = {
            "adjacency": {
                "a.png": [
                    {"source": "a.png", "target": "c.png", "shared_point3d": 4, "center_distance": 1.0},
                    {"source": "a.png", "target": "b.png", "shared_point3d": 9, "center_distance": 2.0},
                    {"source": "a.png", "target": "d.png", "shared_point3d": 12, "center_distance": 1.0},
                ]
            }
        }
        self.assertEqual(
            strong_neighbours(graph, "a.png", {"a.png", "b.png", "c.png"}, count=2),
            ["b.png", "c.png"],
        )


if __name__ == "__main__":
    unittest.main()
