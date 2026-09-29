import unittest
import numpy as np
try:
    from src.camera_adapter import CameraFromWorld, Sim3
    from src.quality_backbone import CameraRequest
    from src.quality_external_camera import official_labels, check_protected, projection_error
    AVAILABLE=True
except ImportError:
    AVAILABLE=False


@unittest.skipUnless(AVAILABLE,'Optional camera dependencies absent')
class ExternalCameraTests(unittest.TestCase):
    def camera(self):
        return CameraRequest(16,12,np.array([[12.,0,8],[0,12,6],[0,0,1]]),
                             CameraFromWorld(np.eye(3),np.array([.2,.3,1.])), 'SIMPLE_RADIAL', .09)

    def test_nonidentity_sim3_projection(self):
        rotation=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        transform=Sim3(2.3,rotation,np.array([4.,-2,.6]))
        points=np.array([[1.,2,3],[-1,.2,4],[0,0,2]])
        self.assertLess(projection_error(self.camera(),transform,points),1e-12)

    def test_nearest_preserves_protected_class_sets(self):
        raw=(np.arange(192).reshape(12,16)%5).astype(np.uint8)
        selected=raw.copy();selected[(raw==3)&(np.indices(raw.shape)[1]>8)]=4
        a,b=[official_labels(x,self.camera()) for x in [raw,selected]]
        check_protected(a,b)
        for k in [0,1,2]:
            np.testing.assert_array_equal(a==k,b==k)
        self.assertEqual(a.dtype,np.uint8)

    def test_forbidden_mutation_rejected(self):
        with self.assertRaises(ValueError):
            check_protected(np.array([[2]],np.uint8),np.array([[4]],np.uint8))
