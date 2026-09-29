from __future__ import annotations

import copy
import unittest

import numpy as np

from src.quality_seed_initialization import (
    FRESH_CLASSIFIER_KEYS,
    tensor_manifest,
    validate_cross_seed_reference,
    validate_same_seed_pair,
)


def state(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    values = {f"body.{index}": np.full((2,), index, np.float32) for index in range(648)}
    values[FRESH_CLASSIFIER_KEYS[0]] = rng.normal(size=(6, 256)).astype(np.float32)
    values[FRESH_CLASSIFIER_KEYS[1]] = rng.normal(size=(6,)).astype(np.float32)
    return values


class SeedInitializationContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.load = {"loaded_tensors": 648, "all_loaded_tensors_exact": True}
        self.old = tensor_manifest(state(20260805))
        self.control = tensor_manifest(state(20260919))

    def test_same_seed_pair_requires_all_state_exact(self) -> None:
        result = validate_same_seed_pair(
            self.control, tensor_manifest(state(20260919)), self.load, self.load
        )
        self.assertTrue(result["all_tensors_exact"])
        self.assertTrue(result["classifier_head_exact"])

    def test_cross_seed_allows_exactly_two_fresh_head_tensors(self) -> None:
        result = validate_cross_seed_reference(
            self.control, self.old, self.load, self.load
        )
        self.assertEqual(result["non_head_tensor_count"], 648)
        self.assertEqual(result["allowed_changed_tensors"], list(FRESH_CLASSIFIER_KEYS))

    def test_cross_seed_non_head_difference_fails(self) -> None:
        changed = copy.deepcopy(self.control)
        changed["body.17"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Non-head tensors differ"):
            validate_cross_seed_reference(changed, self.old, self.load, self.load)

    def test_pretrained_load_report_difference_fails(self) -> None:
        changed_load = {**self.load, "loaded_tensors": 647}
        with self.assertRaisesRegex(ValueError, "pretrained load reports differ"):
            validate_same_seed_pair(self.control, self.control, self.load, changed_load)
        with self.assertRaisesRegex(ValueError, "pretrained load reports differ"):
            validate_cross_seed_reference(
                self.control, self.old, self.load, changed_load
            )

    def test_invalid_fresh_head_shape_or_values_fail(self) -> None:
        for value, message in (
            (np.zeros((5, 256), np.float32), "wrong shape"),
            (np.full((6, 256), np.nan, np.float32), "nonfinite"),
            (np.full((6, 256), np.inf, np.float32), "nonfinite"),
        ):
            with self.subTest(message=message):
                current_state = state(20260919)
                current_state[FRESH_CLASSIFIER_KEYS[0]] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_cross_seed_reference(
                        tensor_manifest(current_state), self.old, self.load, self.load
                    )


if __name__ == "__main__":
    unittest.main()
