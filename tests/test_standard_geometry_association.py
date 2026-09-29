import unittest
from types import SimpleNamespace
import numpy as np
from src.standard_geometry_split import associate
from src.photo_geometry import project


class AssociationTests(unittest.TestCase):
    def setUp(self):
        self.cams={str(i):SimpleNamespace(matrix=np.array([[1000.,0,660],[0,1000.,494.5],[0,0,1.]]),
            pose=SimpleNamespace(rotation=np.eye(3),translation=np.array([-i,0.,0.]))) for i in range(4)}
        self.xyz=np.array([[i*10.,0,20.] for i in range(8)])
        self.scales=np.ones((8,3));self.rot=np.tile(np.eye(3),(8,1,1))
        self.proposals=[{'id':i,'xyz':(x+[.1,0,0]).tolist(),'confidence':.9,
            'observations':[{'camera':n,'xy':project(c,x+[.1,0,0])[0].tolist()} for n,c in self.cams.items()]} for i,x in enumerate(self.xyz)]

    def call(self,used=()):return associate(self.xyz,self.scales,self.rot,self.proposals,self.cams,set(used))

    def test_local_unique_and_quarter_budget(self):
        chosen,_=self.call();self.assertEqual(len(chosen),2)
        self.assertEqual(len({a['parent_row'] for a in chosen}),2)
        self.assertEqual(len({a['proposal_id'] for a in chosen}),2)
        self.assertEqual(chosen,self.call()[0])

    def test_used_proposals_excluded(self):self.assertEqual(self.call(range(8))[0],[])

    def test_bad_projection_rejected_despite_nearby_xyz(self):
        for p in self.proposals:
            for o in p['observations']:o['xy'][1]+=1000
        chosen,audit=self.call();self.assertFalse(chosen);self.assertIn(2,audit['reject_code'])

    def test_remote_point_not_teleported(self):
        for p in self.proposals:p['xyz'][2]+=100
        chosen,audit=self.call();self.assertFalse(chosen);self.assertEqual(set(audit['reject_code']),{1})


if __name__=='__main__':unittest.main()
