import unittest
from collections import OrderedDict
import numpy as np
from src.quality_mask2former import class_masks, semantic_probability, upgrade_official_keys, versioned_state


class Mask2FormerContracts(unittest.TestCase):
    def test_current_schema_retains_version_metadata_and_exact_keys(self):
        current = OrderedDict()
        current._metadata = {"sem_seg_head": {"version": 2}}
        tensor = np.ones(1)
        state = versioned_state({"sem_seg_head.pixel_decoder.weight": tensor}, current)
        self.assertEqual(state._metadata["sem_seg_head"]["version"], 2)
        self.assertIs(state["sem_seg_head.pixel_decoder.weight"], tensor)
        self.assertEqual(len(state), 1)

    def test_author_legacy_query_rename_preserves_tensor(self):
        old = "sem_seg_head.predictor.static_query.weight"
        new = "sem_seg_head.predictor.query_feat.weight"
        value = np.arange(4)
        source = {old: value, "unrelated": value}
        updated, mapping = upgrade_official_keys(source)
        self.assertIs(updated[new], value)
        self.assertIn(old, source)
        self.assertEqual(mapping, {old: new})
        self.assertNotIn(old, updated)
        with self.assertRaises(ValueError):
            upgrade_official_keys({old: value, new: value})

    def test_scores_not_logits_and_argmax_preserved(self):
        scores = np.arange(1, 31, dtype=np.float32).reshape(5, 2, 3)
        p = semantic_probability(scores)
        np.testing.assert_allclose(p, scores / scores.sum(0, keepdims=True))
        np.testing.assert_array_equal(p.argmax(0), scores.argmax(0))
        np.testing.assert_allclose(p.sum(0), 1., atol=2e-6)

    def test_zero_mass_abstains_by_failure_not_fake_background(self):
        x = np.ones((5, 2, 3), np.float32)
        x[:, 1, 2] = 0
        with self.assertRaises(ValueError):
            semantic_probability(x)

    def test_rejects_nonfinite_negative_wrong_classes_and_empty(self):
        for x in [np.full((5, 2, 3), np.nan), np.full((5, 2, 3), np.inf),
                  -np.ones((5, 2, 3)), np.ones((6, 2, 3)), np.ones((5, 0, 1))]:
            with self.subTest(shape=x.shape), self.assertRaises(ValueError):
                semantic_probability(x)

    def test_mask_scores_not_clipped_at_one(self):
        x = np.ones((5, 1, 1), np.float32)
        x[1] = 10
        p = semantic_probability(x)
        self.assertAlmostEqual(float(p[1, 0, 0]), 10/14, places=6)

    def test_float16_input_normalized_in_float32(self):
        p = semantic_probability(np.ones((5, 2, 3), np.float16))
        self.assertEqual(p.dtype, np.float32)

    def test_class_masks_partition_pixels_including_background(self):
        label = np.array([[0, 1, 2], [3, 4, 2]], np.uint8)
        ids, masks = class_masks(label)
        np.testing.assert_array_equal(ids, np.arange(5))
        np.testing.assert_array_equal(masks.sum(0), np.ones((2, 3)))
        np.testing.assert_array_equal(ids[masks.argmax(0)], label)
        self.assertEqual(masks.dtype, bool)

    def test_background_only_and_missing_classes(self):
        ids, masks = class_masks(np.zeros((3, 3), np.uint8))
        np.testing.assert_array_equal(ids, [0])
        self.assertTrue(masks.all())
        ids, masks = class_masks(np.array([[1, 4]], np.uint8))
        np.testing.assert_array_equal(ids, [1, 4])
        self.assertEqual(masks.shape, (2, 1, 2))

    def test_invalid_target_rejected_not_remapped(self):
        for label in [np.array([[255]], np.uint8), np.array([[5]], np.uint8),
                      np.array([[1]], np.int64), np.zeros((0, 1), np.uint8), np.zeros(2, np.uint8)]:
            with self.assertRaises(ValueError):
                class_masks(label)


if __name__ == "__main__":
    unittest.main()
