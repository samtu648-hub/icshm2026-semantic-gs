import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from src.quality_foundation_policy import (replace_foundation, changes, historical_gate,
    select_historical, probability_strata, FrozenFoundationInputs)
from src.quality_semantic_audit import metrics
from scripts.run_quality_foundation_study import calibrate_policy, evaluate_policy, audit_policy
from src.quality_protocol import sha256

ROOT = Path(__file__).resolve().parents[1]
POLICY = "experiments/runs/SEM-150-FOUNDATION-GATE-OOF/config.yaml"


class FoundationPolicyTests(unittest.TestCase):
    def setUp(self):
        self.before = np.array([[0, 1, 2, 3, 4, 4, 4]], np.uint8)
        self.parent = np.array([[1, 1, 2, 3, 0, 0, 4]], np.uint8)
        self.p = np.array([[.9, .9, .9, .9, .2, .5, .9]], np.float32)
        self.policy = json.loads((ROOT / POLICY).read_text())

    def test_only_original_foundation_can_change(self):
        out = replace_foundation(self.parent, self.before, self.p, .5)
        np.testing.assert_array_equal(out, [[1, 1, 2, 3, 0, 4, 4]])
        np.testing.assert_array_equal(self.parent, [[1, 1, 2, 3, 0, 0, 4]])

    def test_off_restores_eligible_not_other_predictions(self):
        np.testing.assert_array_equal(replace_foundation(self.parent, self.before, self.p, None), [[1, 1, 2, 3, 4, 4, 4]])

    def test_exact_threshold_kept_and_protected(self):
        out = replace_foundation(self.parent, self.before, self.p, .5)
        self.assertEqual(out[0, 5], 4)
        for c in (1, 2, 3):
            np.testing.assert_array_equal(out == c, self.parent == c)

    def test_invalid_probability_refused(self):
        for value in (np.nan, np.inf, -.1, 1.1):
            p = self.p.copy(); p[0, 0] = value
            with self.assertRaises(ValueError):
                replace_foundation(self.parent, self.before, p, .5)
        with self.assertRaises(ValueError):
            replace_foundation(self.parent, self.before, self.p[:, :-1], .5)

    def test_invalid_threshold_refused(self):
        for threshold in (0, 1, np.nan, np.inf, -.1):
            with self.assertRaises(ValueError):
                replace_foundation(self.parent, self.before, self.p, threshold)

    def test_invalid_class_and_provenance_refused(self):
        bad = self.parent.copy(); bad[0, -1] = 2
        with self.assertRaises(ValueError):
            replace_foundation(bad, self.before, self.p, .5)
        bad[0, -1] = 5
        with self.assertRaises(ValueError):
            replace_foundation(bad, self.before, self.p, .5)

    def test_modification_accounting(self):
        base = np.array([[0, 4, 0, 2]], np.uint8)
        actual = np.array([[4, 0, 4, 2]], np.uint8)
        target = np.array([[4, 4, 1, 2]], np.uint8)
        self.assertEqual(changes(base, actual, target), dict(changed=3, corrected=1, harmed=1, wrong_to_wrong=1))

    def test_recall_constraint_rejects_higher_iou(self):
        b = np.eye(5, dtype=np.int64)*100
        b[0, 4] = 100
        high_iou = b.copy(); high_iou[0, 4] = 5; high_iou[4, 4] = 94; high_iou[4, 0] = 6
        good = b.copy(); good[0, 4] = 25; good[4, 4] = 96; good[4, 0] = 4
        rows = []
        for t, m in ((.8, high_iou), (.2, good)):
            value = metrics(m)
            rows.append(dict(threshold=t, metrics=value, checks=historical_gate(value, metrics(b), self.policy['oof_gates']['calibration'])))
        self.assertGreater(rows[0]['metrics']['per_class']['foundation']['IoU'], rows[1]['metrics']['per_class']['foundation']['IoU'])
        self.assertEqual(select_historical(rows)['threshold'], .2)

    def test_no_qualifier_and_empty_checks(self):
        self.assertIsNone(select_historical([]))
        self.assertIsNone(select_historical([{'checks': {'recall': False}}, {'checks': {}}]))

    def test_tie_prefers_lower_threshold(self):
        m = metrics(np.eye(5, dtype=np.int64))
        self.assertEqual(select_historical([dict(threshold=t, metrics=m, checks={'pass': True}) for t in (.5, .2)])['threshold'], .2)

    def test_strata_cover_only_eligible(self):
        prob = np.zeros((5, *self.p.shape), np.float32)
        prob[4] = self.p; prob[0] = 1-self.p
        target = np.array([[4, 1, 2, 3, 0, 4, 2]], np.uint8)
        rows = probability_strata(prob, self.before, target, [0, .5, 1], .8)
        self.assertEqual(sum(r['count'] for r in rows.values()), 3)
        for row in rows.values():
            self.assertEqual(sum(row['histogram']), row['count'])
            self.assertEqual(sum(row['argmax_counts']), row['count'])
        self.assertEqual(rows['true_foundation']['rejected'], 1)

    def test_unsealed_holdout_cannot_open(self):
        obj = object.__new__(FrozenFoundationInputs)
        obj.allowed_roles = {'calibration'}
        for role in ('check', 'development'):
            with self.assertRaises(ValueError):
                obj.labels(role)
            with self.assertRaises(ValueError):
                obj.read(role, '001.png')

    def test_synthetic_policy_seal_export_independent_replay(self):
        target = np.array([[0, 1, 2, 3, 4]], np.uint8)
        before = np.array([[4, 1, 2, 3, 4]], np.uint8)
        prob = np.zeros((5, 1, 5), np.float32)
        prob[4, 0] = [.01, .01, .01, .01, .5]
        prob[0] = 1-prob[4]
        parent = np.array([[0, 1, 2, 3, 0]], np.uint8)
        class Labels:
            accessed = set()
            def read(inner, name):
                inner.accessed.add(name)
                return target
        class Source:
            root = ROOT
            roles = {'calibration': ['001.png']}
            config = {'parent_result_sha256': 'synthetic', 'historical_policy': POLICY}
            parent = {'thresholds': {'foundation': .8, 'deck': .98}, 'evaluation': {'calibration': {'metrics': metrics(np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[1,0,0,0,0]], np.int64))}}}
            allowed_roles = {'calibration'}
            def labels(inner, role):
                return Labels()
            def read(inner, role, name):
                return {'cable': before, 'deck': parent}, prob, 'oof-0'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            source = Source()
            seal = calibrate_policy(source, self.policy, path)
            self.assertEqual(seal['selected_threshold'], .1)
            self.assertFalse(seal['development_labels_read_in_this_run'])
            evaluation = evaluate_policy(source, seal['selected_threshold'], path)
            record = {'selected_threshold': .1, 'calibration_sealed_sha256': sha256(path/'calibration_sealed.json'), 'evaluation': evaluation,
                'artifacts': {str(p.relative_to(path)): sha256(p) for p in path.rglob('*') if p.is_file()}}
            (path/'result.json').write_text(json.dumps(record))
            self.assertEqual(audit_policy(source, path, record)['views_replayed'], 1)


if __name__ == '__main__':
    unittest.main()
