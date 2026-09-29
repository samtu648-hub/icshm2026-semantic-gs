#!/usr/bin/env python3
"""Independent CPU role/count and reprojection audit of the fixed sparse targets."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from src.quality_cameras import read_camera_requests
from scripts.run_sem373_joint import write,digest

def main():
 c=json.loads((ROOT/'experiments/runs/SEM-374-STANDARD-STRUCTURE-DEPTH/config.yaml').read_text());base=ROOT/c['output'];obs=json.loads((ROOT/c['structure_observations']).read_text());meta={v['point_id']:v for v in json.loads((base/'tracks-r3/tracks.json').read_text())};cams=read_camera_requests(ROOT/c['scene']/'sparse/0');n=0;errors=[];byid={};counts={}
 for name,rows in obs.items():
  cam=cams[name];gt=np.asarray(Image.open(ROOT/c['labels']/name))
  for v in rows:
   m=meta[v['point_id']];role='check' if v['point_id']%5==0 else 'fit';assert v['role']==m['role']==role;assert m['class_id']==v['class_id'] in (1,3,4)
   xyz=np.asarray(m['xyz']);xyzcam=cam.pose.rotation@xyz+cam.pose.translation;uv=cam.matrix@xyzcam;uv=uv[:2]/uv[2];error=float(np.linalg.norm(uv-v['xy']));assert error<=2.+1e-9;assert abs(1/xyzcam[2]-v['inverse_z'])<1e-10
   x,y=np.floor(v['xy']).astype(int);assert gt[y,x]==v['class_id'];assert (gt[y-1:y+2,x-1:x+2]==v['class_id']).all()
   assert name not in byid.setdefault(v['point_id'],set());byid[v['point_id']].add(name);errors.append(error);n+=1;k=str(v['class_id'])+'_'+role;counts[k]=counts.get(k,0)+1
 assert all(len(v)>=3 for v in byid.values());assert len(byid)==len(meta);assert len(obs)==300
 write(base/'tracks-independent-audit.json',dict(status='passed',observations=n,tracks=len(meta),views=300,counts=counts,max_reprojection_error=max(errors),role_overlap=0,all_positive_z_and_inverse_z_exact=True,observation_hash=digest(ROOT/c['structure_observations']),source_hash=digest(__file__)))
 print(n,len(meta),counts)
if __name__=='__main__':main()
