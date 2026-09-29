import importlib.util
import unittest
import tempfile
from pathlib import Path
import numpy as np
from src.quality_crossview_evidence import (STATES, source_sample, source_status,
    evidence_states, counted, compare_evidence, snapshot_real_sources)
from src.geometry_prior.camera_io import ColmapCamera, ColmapImage
from src.semantic_dense_geometry import normalized_grid


class CrossviewEvidenceTests(unittest.TestCase):
    def test_snapshot_ignores_virtual_modules_but_required_missing_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); required = root/'required.py'; required.write_text('pass\n')
            real = root/'actual.py'; real.write_text('value = 1\n')
            result = snapshot_real_sources(root, root/'snap', ['required.py'], ['_ops.py', None, 1, str(real)])
            self.assertEqual(set(result), {'required.py', 'actual.py'})
            self.assertEqual((root/'snap/actual.py').read_bytes(), real.read_bytes())
            with self.assertRaises(FileNotFoundError):
                snapshot_real_sources(root, root/'bad', ['absent.py'], [])

    def sample(self, labels, valid=True, reliable=True):
        labels = np.asarray(labels, np.uint8)
        return {'label': labels, 'valid': np.full(labels.shape, valid, bool), 'reliable': np.full(labels.shape, reliable, bool)}

    def test_all_nine_evidence_categories(self):
        a, b = self.sample([0, 1, 4, 2, 1]), self.sample([0, 1, 4, 2, 4])
        np.testing.assert_array_equal(evidence_states(a, b, True, True), [5, 4, 6, 7, 8])
        np.testing.assert_array_equal(evidence_states(a, b, False, False), [0]*5)
        np.testing.assert_array_equal(evidence_states(a, b, True, False), [1]*5)
        b['valid'][0] = False; b['reliable'][1] = False
        np.testing.assert_array_equal(evidence_states(a, b, True, True), [2, 3, 6, 7, 8])
        self.assertEqual(len(STATES), 9)

    def test_second_without_first_and_shape_mismatch_refused(self):
        with self.assertRaises(ValueError): evidence_states(self.sample([1]), self.sample([1]), False, True)
        with self.assertRaises(ValueError): evidence_states(self.sample([1]), self.sample([1, 2]), True, True)

    def test_one_reliable_deck_source_is_not_pair_support(self):
        a, b = self.sample([1]), self.sample([1], reliable=False)
        self.assertEqual(evidence_states(a, b, True, True).item(), 3)
        b = self.sample([0]); self.assertEqual(evidence_states(a, b, True, True).item(), 8)

    def test_source_status_exclusive_and_unknown(self):
        a = self.sample([0, 1, 2, 3, 4, 255, 1])
        a['valid'][5] = False; a['reliable'][6] = False
        np.testing.assert_array_equal(source_status(a, True), [3, 4, 5, 6, 7, 1, 2])
        np.testing.assert_array_equal(source_status(a, False), [0]*7)

    def test_histograms_include_false_evidence_and_empty_denominator(self):
        result = counted([4, 4, 4, 3], np.array([1, 0, 2, 1], np.uint8), 9)
        np.testing.assert_array_equal(result[4], [1, 1, 1, 0, 0])
        self.assertEqual(counted([], np.empty(0, np.uint8), 9).sum(), 0)
        for states, gt in [([9], [1]), ([4], [5]), ([-1], [1]), ([1, 2], [1])]:
            with self.assertRaises(ValueError): counted(states, gt, 9)

    def setup_native(self):
        camera = ColmapCamera(1, 'SIMPLE_RADIAL', 32, 24, (30., 16., 12., 0.))
        pose = ColmapImage(1, 't.png', 1, np.array([1., 0, 0, 0]), np.zeros(3), ())
        sp = ColmapImage(2, 's.png', 1, pose.qvec, np.array([1., 0, 0]), ())
        grid = normalized_grid(8, 8)
        source = {'pose': sp, 'camera': camera, 'labels': np.ones((24, 32), np.uint8),
                  'maps': [(grid, np.ones((8, 8))), (grid, np.ones((8, 8)))]}
        points = np.array([[[10.5, 10.5], [16.5, 12.5], [20.5, 15.5]]])
        filters = {'overlap_each_direction_minimum': .5, 'cycle_maximum_native_pixels': 2., 'epipolar_maximum_native_pixels': 2.}
        return points, source, pose, camera, filters

    def test_detailed_native_evidence_reproduces_support(self):
        args = self.setup_native()
        a = source_sample(*args, details=True)
        self.assertTrue(a['reliable'].all()); self.assertTrue((a['label'] == 1).all())
        np.testing.assert_allclose(a['forward_confidence'], 1)
        np.testing.assert_allclose(a['reverse_confidence'], 1)
        np.testing.assert_allclose(a['cycle'], 0, atol=1e-12)
        np.testing.assert_allclose(a['epipolar'], 0, atol=1e-12)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'optional Torch independent replay')
    def test_independent_all_native_details_and_dtype_contract(self):
        args = self.setup_native()
        a = source_sample(*args, details=True)
        b = source_sample(*args, independent=True, details=True)
        compare_evidence(a, b)
        b['label'][0, 0] = 0
        with self.assertRaises(ValueError): compare_evidence(a, b)
        b = {**a, 'xy': a['xy'].astype(np.float32)}
        with self.assertRaises(ValueError): compare_evidence(a, b)

    def test_missing_source_and_empty_points(self):
        points, source, pose, camera, filters = self.setup_native()
        for src, pts in ((None, points), (source, points[:, :0])):
            out = source_sample(pts, src, pose, camera, filters, details=True)
            self.assertFalse(out['reliable'].any()); self.assertFalse(out['valid'].any())
            self.assertTrue(np.isnan(out['forward_confidence']).all())

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'optional Torch independent replay')
    def test_invalid_native_border_padding_is_not_source_evidence(self):
        points, source, pose, camera, filters = self.setup_native()
        points = np.array([[[.5, .5], [31.5, 23.5], [16.5, 12.5]]])
        a = source_sample(points, source, pose, camera, filters, details=True)
        b = source_sample(points, source, pose, camera, filters, independent=True, details=True)
        compare_evidence(a, b)
        np.testing.assert_array_equal(a['valid'], [[False, False, True]])
        self.assertTrue(np.isnan(a['xy'][0, :2]).all())
        np.testing.assert_array_equal(evidence_states(a, a, True, True), [[2, 2, 4]])

    def test_both_confidences_required_and_outside_labels_abstains(self):
        points, source, pose, camera, filters = self.setup_native()
        source['maps'][1] = (source['maps'][1][0], np.full((8, 8), .49))
        self.assertFalse(source_sample(points, source, pose, camera, filters, details=True)['reliable'].any())
        source['maps'][0] = (np.full((8, 8, 2), 2.), np.ones((8, 8)))
        out = source_sample(points, source, pose, camera, filters, details=True)
        self.assertFalse(out['valid'].any()); self.assertTrue((out['label'] == 255).all())


if __name__ == '__main__': unittest.main()
