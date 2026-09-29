import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from src.quality_paired_calibration import select_recall_preserving, independent_map, LabelRelease
from src.quality_semantic_audit import metrics
from src.quality_fusion import pipeline
from scripts.run_quality_paired_calibration import calculate_curves, evaluate, KEYS


def score(tp, fp):
    matrix = np.eye(5, dtype=np.int64)*100
    matrix[0, 0] = 200-fp
    matrix[0, 4] = fp
    matrix[4, 4] = tp
    matrix[4, 0] = 100-tp
    return metrics(matrix)


class PairedCalibrationTests(unittest.TestCase):
    def test_relative_floor_rejects_more_precise_lower_recall(self):
        rows = [dict(threshold=t, metrics=score(tp, fp)) for t, tp, fp in
            [(None, 100, 80), (.5, 90, 20), (.8, 80, 0)]]
        self.assertGreater(rows[2]['metrics']['mIoU_with_background'], rows[1]['metrics']['mIoU_with_background'])
        self.assertEqual(select_recall_preserving(rows, score(90, 40))['threshold'], .5)

    def test_off_is_real_candidate_and_tie_preference(self):
        rows = [dict(threshold=t, metrics=score(100, 0)) for t in (.8, None, .2)]
        self.assertIsNone(select_recall_preserving(rows, score(90, 40))['threshold'])

    def test_higher_threshold_breaks_remaining_tie(self):
        rows = [dict(threshold=None, metrics=score(100, 100))]+[
            dict(threshold=t, metrics=score(100, 0)) for t in (.2, .8)]
        self.assertEqual(select_recall_preserving(rows, score(90, 40))['threshold'], .8)

    def test_different_targets_or_undefined_recall_refused(self):
        row = dict(threshold=None, metrics=score(100, 80))
        row['metrics']['confusion'][0][0] += 1
        with self.assertRaises(ValueError):
            select_recall_preserving([row], score(90, 40))
        matrix = np.eye(5, dtype=np.int64); matrix[4, 4] = 0
        with self.assertRaises(ValueError):
            select_recall_preserving([], metrics(matrix))

    def test_no_admissible_off_refused(self):
        for rows in ([], [dict(threshold=.8, metrics=score(90, 20))],
                [dict(threshold=None, metrics=score(80, 20))]):
            with self.assertRaises(ValueError):
                select_recall_preserving(rows, score(90, 40))

    def test_direct_construct_all_classes_and_threshold_boundary(self):
        static = np.array([[0, 1, 2, 3, 4, 4, 4]], np.uint8)
        p4 = np.array([[.9, .9, .9, .9, .1, .5, .8]], np.float32)
        cable = np.array([[.75, 0, .1, 0, 0, 0, 0]], np.float32)
        deck = np.ones_like(cable)
        expected = [[2, 1, 1, 3, 0, 4, 4]]
        np.testing.assert_array_equal(independent_map(static, p4, cable, deck, .5), expected)
        np.testing.assert_array_equal(independent_map(static, p4, cable, deck, None), [[2, 1, 1, 3, 4, 4, 4]])

    def test_random_direct_matches_native_pipeline(self):
        rng = np.random.default_rng(123)
        static = rng.integers(0, 5, (16, 21), dtype=np.uint8)
        five = rng.random((5, 16, 21)).astype(np.float32); five /= five.sum(0)
        cable, deck = rng.random((2, 16, 21)).astype(np.float32)
        for t in (None, .1, .8):
            direct = independent_map(static, five[4], cable, deck, t)
            native = pipeline(static, five, cable, deck, dict(fiveclass_lane=.8, cable=.75, foundation=t, deck=.98))['deck']
            np.testing.assert_array_equal(direct, native)

    def test_empty_background_and_invalid_inputs(self):
        static = np.zeros((2, 3), np.uint8); prob = np.zeros((2, 3), np.float32)
        np.testing.assert_array_equal(independent_map(static, prob, prob, prob, .8), static)
        for p in (np.full_like(prob, np.nan), np.full_like(prob, 1.1), prob[:, :-1]):
            with self.assertRaises(ValueError):
                independent_map(static, p, prob, prob, .8)
        for threshold in (0, 1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                independent_map(static, prob, prob, prob, threshold)

    def test_check_release_requires_immutable_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); guard = LabelRelease(folder)
            guard.require('calibration')
            for role in ('check', 'development'):
                with self.assertRaises(ValueError): guard.require(role)
            (folder/'calibration_sealed.json').write_text('{}')
            guard.seal(); guard.require('check')
            with self.assertRaises(ValueError): guard.seal()
            (folder/'calibration_sealed.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError): guard.require('check')

    def test_synthetic_full_curves_export_and_fresh_reconstruction(self):
        static = np.array([[4, 1, 2, 3, 4]], np.uint8)
        target = np.array([[0, 1, 2, 3, 4]], np.uint8)
        five = np.zeros((5, 1, 5), np.float32)
        five[4] = [.01, .1, .1, .1, .7]; five[0] = 1-five[4]
        probs = {'fiveclass': five, 'sixk': five, 'stay_cable': np.ones((1, 5), np.float32), 'deck': np.zeros((1, 5), np.float32)}
        class Source:
            cfg = {'threshold_grid': [None, .5, .8], 'fixed_thresholds': dict(fiveclass_lane=.8, cable=.75, foundation=.8, deck=.98)}
            roles = {'calibration': ['001.png'], 'check': ['002.png']}
            index = {name: ('oof-0', {}) for name in ('001.png', '002.png')}
            def labels(self, role):
                self.release.require(role)
                class Reader:
                    def read(inner, name):
                        if name not in self.roles[role]: raise ValueError('Wrong label')
                        return target
                return Reader()
        from src.quality_fusion import confusion
        with tempfile.TemporaryDirectory() as tmp, patch('scripts.run_quality_paired_calibration.load_inputs', return_value=(static, probs)):
            folder = Path(tmp); s = Source(); s.release = LabelRelease(folder)
            curves = calculate_curves(s)
            self.assertEqual(curves, calculate_curves(s, independent=True))
            reference = curves['3k'][2]['metrics']
            selected = {k: select_recall_preserving(v, reference)['threshold'] for k, v in curves.items()}
            self.assertEqual(selected, {'3k': .5, '6k': .5})
            s.parent = {'evaluation': {r: {'metrics': {f'full_{e}': reference for e in ('3k', '6k')}} for r in s.roles}}
            (folder/'calibration_sealed.json').write_text(json.dumps(selected)); s.release.seal()
            actual = evaluate(s, selected, folder)
            self.assertEqual(actual, evaluate(s, selected, folder, audit=True))
            self.assertEqual(set(actual['check']['metrics']), set(KEYS))
            self.assertEqual(actual['check']['changes']['policy_6k']['corrected'], 1)
            with np.load(folder/'check/evidence_6k/002.npz') as z:
                self.assertEqual(z['flat_index'].tolist(), [4])


if __name__ == '__main__':
    unittest.main()
