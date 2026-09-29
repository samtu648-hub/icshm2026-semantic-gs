import unittest
import importlib.util
from unittest.mock import patch
import numpy as np
from src.quality_crossview_migration import (require_source, SourceLabels, migrated_prediction,
    evidence_trace, verify_trace)
from src.geometry_prior.camera_io import ColmapCamera, ColmapImage
from src.semantic_dense_geometry import normalized_grid


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.folds=[{'fold':0,'train':['s.png'],'oof':['t.png'],'guard':['g.png']},
                    {'fold':1,'train':['t.png'],'oof':['s.png'],'guard':['g.png']}]
        self.camera=ColmapCamera(1,'SIMPLE_RADIAL',32,24,(30.,16.,12.,0.))
        self.pose=ColmapImage(1,'t.png',1,np.array([1.,0,0,0]),np.zeros(3),())
        self.sp=ColmapImage(2,'s.png',1,self.pose.qvec,np.array([1.,0,0]),())
        grid=normalized_grid(8,8)
        self.maps=[(grid,np.ones((8,8))),(grid,np.ones((8,8)))]
        self.filters={'overlap_each_direction_minimum':.5,'cycle_maximum_native_pixels':2.,'epipolar_maximum_native_pixels':2.}
        self.base=np.full((24,32),4,np.uint8)
        self.base[2,3]=1;self.base[2,4]=2;self.base[2,5]=3;self.base[2,6]=0

    def src(self,name,label):
        return {'name':name,'pose':self.sp,'camera':self.camera,'labels':np.full_like(self.base,label),'maps':self.maps}

    def run_review(self,first,second=None,independent=False):
        sources=[s['name'] for s in (first,second) if s]
        target={'accessible_sources':sources,'selected_source':sources[0] if sources else None}
        return migrated_prediction(self.base,target,first,second,self.pose,self.camera,self.filters,independent)

    def test_target_guard_and_wrong_fold_source_refused(self):
        require_source(self.folds,0,'t.png','s.png')
        for target,source in [('t.png','t.png'),('t.png','g.png'),('s.png','t.png'),('t.png','x.png')]:
            with self.assertRaises(ValueError):require_source(self.folds,0,target,source)

    def test_source_historical_target_in_other_fold_is_allowed(self):
        require_source(self.folds,1,'s.png','t.png')

    def test_role_validation_before_cached_label_access(self):
        obj=object.__new__(SourceLabels);obj.folds=self.folds
        obj.cache={(0,'t.png'):self.base};obj.readers={};obj.access=[]
        with self.assertRaises(ValueError):obj.read(0,'t.png','t.png')
        self.assertEqual(obj.access,[])

    def test_no_sources_is_exact_abstention(self):
        dense,veto,stats=self.run_review(None)
        np.testing.assert_array_equal(dense,self.base);np.testing.assert_array_equal(veto,self.base)
        self.assertEqual(stats['veto']['old_deletions'],0)

    def test_background_deletion_and_second_foundation_veto(self):
        first,second=self.src('a.png',0),self.src('b.png',4)
        dense,veto,stats=self.run_review(first,second)
        self.assertGreater(stats['dense']['changed_pixels'],0)
        np.testing.assert_array_equal(veto,self.base)
        self.assertEqual(stats['dense']['changed_pixels'],stats['veto']['foundation_veto_pixels'])
        for c in (0,1,2,3):self.assertTrue((veto[self.base==c]==c).all())

    def test_source_nonbackground_does_not_delete(self):
        for label in (1,2,3,4):
            dense,veto,_=self.run_review(self.src('a.png',label))
            np.testing.assert_array_equal(dense,self.base);np.testing.assert_array_equal(veto,self.base)

    def test_nonfoundation_second_source_does_not_veto(self):
        for label in (0,1,2,3):
            dense,veto,_=self.run_review(self.src('a.png',0),self.src('b.png',label))
            np.testing.assert_array_equal(dense,veto)

    def test_unreliable_maps_abstain_and_no_background_added(self):
        self.maps[0]=(self.maps[0][0],np.full((8,8),.49))
        dense,veto,_=self.run_review(self.src('a.png',0),self.src('b.png',4))
        np.testing.assert_array_equal(dense,self.base);np.testing.assert_array_equal(veto,self.base)

    def test_source_order_and_empty_camera_mismatch_refused(self):
        with self.assertRaises(ValueError):
            migrated_prediction(self.base,{'accessible_sources':['b.png'],'selected_source':'b.png'},
                self.src('a.png',0),None,self.pose,self.camera,self.filters)
        with self.assertRaises(ValueError):
            migrated_prediction(self.base[:1],{'accessible_sources':[],'selected_source':None},
                None,None,self.pose,self.camera,self.filters)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'optional Torch runtime required for independent replay')
    def test_independent_native_outputs_and_trace(self):
        first,second=self.src('a.png',0),self.src('b.png',4)
        d,v,s=self.run_review(first,second)
        dd,vv,ss=self.run_review(first,second,True)
        np.testing.assert_array_equal(d,dd);np.testing.assert_array_equal(v,vv);self.assertEqual(s,ss)
        a=evidence_trace(self.base,d,v,first,second,self.pose,self.camera,self.filters)
        b=evidence_trace(self.base,dd,vv,first,second,self.pose,self.camera,self.filters,True)
        verify_trace(a,b)
        self.assertTrue((a['first_label']==0).all());self.assertTrue((a['second_label']==4).all())
        changed={k:x.copy() for k,x in b.items()};changed['veto_class'][0]=0
        with self.assertRaises(ValueError):verify_trace(a,changed)

    def test_out_of_bounds_source_evidence_abstains(self):
        self.maps[0]=(np.full((8,8,2),2.),np.ones((8,8)))
        dense,veto,_=self.run_review(self.src('a.png',0))
        np.testing.assert_array_equal(dense,self.base);np.testing.assert_array_equal(veto,self.base)

    def test_empty_trace_without_sources(self):
        trace=evidence_trace(self.base,self.base,self.base,None,None,self.pose,self.camera,self.filters)
        self.assertEqual(len(trace['flat_index']),0)
        verify_trace(trace,trace)


if __name__=='__main__':unittest.main()
