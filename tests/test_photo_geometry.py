import unittest
from types import SimpleNamespace
import numpy as np
from src.photo_geometry import resized_to_colmap, project, epipolar_error, triangulate_track


class PhotoGeometryTests(unittest.TestCase):
    def setUp(self):
        self.cams={str(i):SimpleNamespace(matrix=np.array([[1000.,0,660],[0,1000.,494.5],[0,0,1.]]),
            pose=SimpleNamespace(rotation=np.eye(3),translation=np.array([-x,0.,0.])))
            for i,x in enumerate([-1.,-.35,.4,1.1])}
        self.xyz=np.array([.1,.2,10.])
        self.obs=[{'camera':n,'xy':project(c,self.xyz)[0].tolist()} for n,c in self.cams.items()]

    def test_pixel_centres_and_anisotropic_resize(self):
        p=resized_to_colmap([[0,0],[1023,767]])
        np.testing.assert_allclose(p[0],[.5*1320/1024,.5*989/768])
        np.testing.assert_allclose(p[1],[1320-.5*1320/1024,989-.5*989/768])

    def test_valid_triple_and_separate_check(self):
        q=triangulate_track(self.obs,self.cams)
        self.assertTrue(q['accepted'],q)
        np.testing.assert_allclose(q['xyz'],self.xyz,atol=1e-10)
        self.assertEqual(len(q['build_indices']),3)
        self.assertEqual(len(q['check_indices']),1)

    def test_duplicate_camera(self):
        self.assertEqual(triangulate_track(self.obs+[self.obs[0]],self.cams)['reason'],'duplicate_camera_conflict')

    def test_negative_depth(self):
        obs=[{'camera':n,'xy':project(c,-self.xyz)[0].tolist()} for n,c in self.cams.items()]
        self.assertEqual(triangulate_track(obs,self.cams)['reason'],'nonpositive_depth')

    def test_low_parallax(self):
        obs=[{'camera':n,'xy':project(c,self.xyz*1000)[0].tolist()} for n,c in self.cams.items()]
        self.assertEqual(triangulate_track(obs,self.cams)['reason'],'low_parallax')

    def test_extra_camera_catches_wrong_match(self):
        original=triangulate_track(self.obs,self.cams)
        i=original['check_indices'][0]
        self.obs[i]['xy'][1]+=20
        q=triangulate_track(self.obs,self.cams)
        self.assertFalse(q['accepted'])
        self.assertIn(q['reason'],['extra_view_reprojection','build_reprojection'])

    def test_epipolar_wrong_match(self):
        a,b=np.array([self.obs[0]['xy']]),np.array([self.obs[1]['xy']])
        self.assertLess(epipolar_error(a,b,self.cams['0'],self.cams['1'])[0],1e-9)
        b[:,1]+=10
        self.assertGreater(epipolar_error(a,b,self.cams['0'],self.cams['1'])[0],9.9)


if __name__=='__main__':unittest.main()
