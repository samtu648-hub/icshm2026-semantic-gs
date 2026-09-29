import unittest

from src.scaffold_appearance import (
    nearest_training_appearance,
    partition_scaffold_camera_records,
)


def camera(name, x, y=0.0, z=0.0):
    return {"img_name": name, "position": [x, y, z]}


class ScaffoldAppearanceTest(unittest.TestCase):
    def test_partition_recovers_test_prefix_and_training_embedding_order(self):
        records = [camera("054", 0), camera("055", 1), camera("001", 2), camera("003", 3)]
        test, train = partition_scaffold_camera_records(records, ["055.png", "054.png"])
        self.assertEqual([item["img_name"] for item in test], ["054", "055"])
        self.assertEqual([item["img_name"] for item in train], ["001", "003"])

    def test_partition_rejects_prefix_different_from_frozen_split(self):
        with self.assertRaises(ValueError):
            partition_scaffold_camera_records(
                [camera("055", 0), camera("054", 1), camera("001", 2)],
                ["054.png", "055.png"],
            )

    def test_nearest_center_returns_training_embedding_index(self):
        selected = nearest_training_appearance(
            camera("054", 4.1),
            [camera("001", 0.0), camera("003", 4.0), camera("004", 9.0)],
        )
        self.assertEqual(selected["appearance_id"], 1)
        self.assertEqual(selected["source_name"], "003.png")
        self.assertAlmostEqual(selected["center_distance"], 0.1)

    def test_nearest_center_tie_uses_lower_embedding_index(self):
        selected = nearest_training_appearance(
            camera("054", 1.0), [camera("001", 0.0), camera("003", 2.0)]
        )
        self.assertEqual(selected["appearance_id"], 0)


if __name__ == "__main__":
    unittest.main()
