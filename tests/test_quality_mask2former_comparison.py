import itertools
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from src.quality_fusion import pipeline
from src.quality_mask2former_comparison import (ARMS, independent_map, validate_arms,
    assert_baseline_counts, class_transitions)
from scripts.evaluate_quality_mask2former_oof import calibrate, score


class ComparisonTests(unittest.TestCase):
    def test_all_off_on_combinations_match_stage_pipeline(self):
        rng = np.random.default_rng(341)
        static = np.tile(np.arange(5, dtype=np.uint8), (7, 3))
        five = rng.random((5, *static.shape), dtype=np.float32)
        five /= five.sum(0)
        cable, deck = rng.random((2, *static.shape), dtype=np.float32)
        for choices in itertools.product((None, .2, .8), repeat=4):
            thresholds = dict(zip(("fiveclass_lane", "cable", "foundation", "deck"), choices))
            self.assertTrue(np.array_equal(independent_map(static, five, cable, deck, thresholds),
                pipeline(static, five, cable, deck, thresholds)["deck"]))

    def test_deck_cannot_fill_deleted_foundation(self):
        static = np.array([[4, 0, 2, 1, 3]], np.uint8)
        five = np.zeros((5, 1, 5), np.float32); five[0] = 1
        result = independent_map(static, five, np.zeros((1, 5), np.float32), np.ones((1, 5), np.float32),
            dict(cable=.75, foundation=.8, deck=.98))
        np.testing.assert_array_equal(result, [[0, 1, 1, 1, 3]])

    def test_partial_prefix_and_unknown_invalid_threshold(self):
        s = np.array([[0, 2]], np.uint8)
        p = np.full((5, 1, 2), .2, np.float32)
        b = np.full((1, 2), .5, np.float32)
        np.testing.assert_array_equal(independent_map(s, p, b, b, {}), s)
        for thresholds in ({"unknown": None}, {"deck": float("nan")}, {"cable": 0}):
            with self.assertRaises(ValueError): independent_map(s, p, b, b, thresholds)

    def test_no_missing_control_arm(self):
        validate_arms(ARMS)
        with self.assertRaises(ValueError): validate_arms(ARMS[1:])

    def test_baseline_recount_and_transition_partition(self):
        m = {"confusion": np.eye(5, dtype=int).tolist()}
        evaluation = {f"{a}/{v}": m for a in ARMS[:2] for v in ("raw", "fixed")}
        paired = {f"{v}_{n}": m for v in ("raw", "full") for n in ("3k", "6k")}
        assert_baseline_counts(evaluation, paired)
        with self.assertRaises(ValueError):
            assert_baseline_counts(dict(evaluation, **{"b2_6k/raw": {"confusion": []}}), paired)
        cube = class_transitions(np.array([0, 4, 2], np.uint8), np.array([1, 0, 2], np.uint8),
            np.array([1, 4, 2], np.uint8))
        self.assertEqual(cube.sum(), 2)
        self.assertEqual(cube[0, 1, 1], 1)
        self.assertEqual(cube[4, 0, 4], 1)

    def test_full_grid_calibration_then_native_scoring_independent_replay(self):
        rng = np.random.default_rng(341)
        static = rng.integers(0, 5, (8, 10), dtype=np.uint8)
        target = rng.integers(0, 5, static.shape, dtype=np.uint8)
        probs = {arm: rng.random((5, *static.shape), dtype=np.float32) for arm in ARMS}
        for value in probs.values(): value /= value.sum(0)
        probs.update(stay_cable=rng.random(static.shape, dtype=np.float32), deck=rng.random(static.shape, dtype=np.float32))
        root = Path(__file__).resolve().parents[1]
        grid = json.loads((root / "experiments/runs/SEM-331-QUALITY-EXPERTS/fusion_config.yaml").read_text())
        cfg = {"comparison": {"fixed_thresholds": dict(fiveclass_lane=.8, cable=.75, foundation=.8, deck=.98)}}
        class Labels:
            def read(self, name):
                if name != "test.png": raise ValueError("Unregistered target")
                return target
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            output, replay = Path(tmp) / "original", Path(tmp) / "replay"
            with patch("scripts.evaluate_quality_mask2former_oof.load_inputs", return_value=(static, probs)):
                index = {"test.png": ("oof-0", {})}
                t, curves = calibrate(index, ["test.png"], Labels(), grid)
                rt, rc = calibrate(index, ["test.png"], Labels(), grid, replay=True)
                self.assertEqual(t, rt); self.assertEqual(curves, rc)
                primary = score(index, ["test.png"], Labels(), cfg, t, output, "calibration")
                repeated = score(index, ["test.png"], Labels(), cfg, rt, replay, "calibration", output)
                self.assertEqual(primary, repeated)
                self.assertEqual(len(list(output.rglob("*.png"))), 17)


if __name__ == "__main__":
    unittest.main()
