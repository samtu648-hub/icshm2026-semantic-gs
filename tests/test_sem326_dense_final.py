import unittest
try:
    import numpy as np
    from src.semantic_dense_final import inverse_radial, review_official_output
    from src.semantic_dense_review import native_prediction
    from src.semantic_dense_geometry import normalized_grid
    from src.geometry_prior.camera_io import ColmapCamera, ColmapImage
except ImportError:
    np = None


@unittest.skipIf(np is None, "optional native geometry unavailable")
class FinalDenseTests(unittest.TestCase):
    def setUp(self):
        self.ca = ColmapCamera(1, "SIMPLE_RADIAL", 32, 24, (30., 16., 12., 0.))
        self.a = ColmapImage(1, "novel-a", 1, np.array([1., 0, 0, 0]), np.zeros(3), ())
        self.b = ColmapImage(2, "bank-b", 1, self.a.qvec, np.array([1., 0, 0]), ())
        grid = normalized_grid(8, 8)
        self.maps = [(grid, np.ones((8,8))), (grid, np.ones((8,8)))]
        self.filters = {"overlap_each_direction_minimum": .5, "cycle_maximum_native_pixels": 2., "epipolar_maximum_native_pixels": 2.}

    def test_zero_distortion_matches_unchanged_pinhole_reviewer(self):
        old = np.full((24,32), 4, np.uint8); source = np.zeros_like(old)
        a, sa = native_prediction(old, source, self.maps, self.a, self.b, self.ca, self.ca, self.filters)
        b, sb = review_official_output(old, source, self.maps, self.a, self.b, self.ca, self.ca, self.filters)
        np.testing.assert_array_equal(a,b); self.assertEqual(sa,sb)

    def test_radial_inverse_known_forward_and_independent(self):
        ca = ColmapCamera(1,"SIMPLE_RADIAL",32,24,(30.,16.,12.,.01))
        normalized = np.array([[[-.3,.2],[.4,-.1]]])
        distorted = normalized * (1 + .01 * (normalized ** 2).sum(-1,keepdims=True)) * 30 + [16,12]
        a = inverse_radial(distorted,ca); b = inverse_radial(distorted,ca,True)
        np.testing.assert_allclose(a, normalized * 30 + [16,12], atol=1e-10,rtol=0)
        np.testing.assert_allclose(a,b,atol=1e-10,rtol=0)

    def test_distorted_protection_bypass_and_independent(self):
        try: import torch
        except ImportError: self.skipTest("optional torch unavailable")
        ca = ColmapCamera(1,"SIMPLE_RADIAL",32,24,(30.,16.,12.,.01))
        old = (np.arange(32*24).reshape(24,32)%5).astype(np.uint8)
        source = np.zeros_like(old)
        a, sa = review_official_output(old,source,self.maps,self.a,self.b,ca,self.ca,self.filters)
        b, sb = review_official_output(old,source,self.maps,self.a,self.b,ca,self.ca,self.filters,True)
        np.testing.assert_array_equal(a,b); self.assertEqual(sa,sb)
        self.assertTrue(sa["changed_pixels"]>0)
        for k in (1,2,3): np.testing.assert_array_equal(a==k, old==k)
        bypass,_ = review_official_output(old,source,self.maps,self.a,self.b,ca,self.ca,self.filters,enabled=False)
        np.testing.assert_array_equal(bypass,old)

    def test_renderer_changes_only_extra_pinhole_export(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        old = (root/"scripts/render_submission_sem161.py").read_text()
        new = (root/"scripts/render_sem326_base.py").read_text()
        old = old.replace('"""Full frozen SEM-161 bundle: external COLMAP camera metadata only."""',
                          '"""SEM326 probe: unchanged frozen bundle render, additionally export PINHOLE RGB."""')
        old = old.replace('outputs = {"rgb": rgb, "semantic": stages["SEM-161"]}',
                          'outputs = {"rgb": rgb, "semantic": stages["SEM-161"], "pinhole_rgb": pinhole}')
        self.assertEqual(old,new)

    def test_venv_interpreter_symlink_is_not_dereferenced(self):
        import tempfile
        from pathlib import Path
        import sys
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0,str(root/"scripts"))
        from render_sem326_external import argument_path
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory)/"python"
            executable.symlink_to(sys.executable)
            self.assertEqual(argument_path("render_python",executable),executable.absolute())
            self.assertNotEqual(argument_path("render_python",executable),executable.resolve())


if __name__ == "__main__": unittest.main()
