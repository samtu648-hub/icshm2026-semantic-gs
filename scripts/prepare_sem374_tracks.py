#!/usr/bin/env python3
"""Bind human-labelled multiview SfM observations, without new reconstruction."""
import argparse,json,sys,math,time
from pathlib import Path
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scripts.prepare_rgb_colmap_scene import parse_colmap_images,parse_single_simple_radial_camera,undistort_observations
from scripts.run_sem373_joint import digest,write

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--attempt',default='r1');a=p.parse_args();c=json.loads((ROOT/'experiments/runs/SEM-374-STANDARD-STRUCTURE-DEPTH/config.yaml').read_text());out=ROOT/c['output']/('tracks-'+a.attempt);out.mkdir(exist_ok=False);start=time.monotonic()
 records=parse_colmap_images(a.dataset_root/'camera_parameters/images.txt');_,w,h,f,cx,cy,k=parse_single_simple_radial_camera(a.dataset_root/'camera_parameters/cameras.txt');K=np.array([[f,0,cx],[0,f,cy],[0,0,1.]])
 undistort_observations(records,K,k);points={}
 source=ROOT/c['scene']/'sparse/0/points3D.txt'
 for line in source.read_text().splitlines():
  if line and not line.startswith('#'):
   v=line.split();points[int(v[0])]=np.array(v[1:4],float)
 split=json.loads((ROOT/c['split_path']).read_text());labeled=set(split['semantic_train']);tracks=defaultdict(list);labels_hash={};rejected=Counter()
 for rec in records:
  if rec.name not in labeled:continue
  lp=ROOT/c['labels']/rec.name;labels_hash[rec.name]=digest(lp);lab=np.asarray(Image.open(lp));dist={cl:distance_transform_edt(lab==cl) for cl in c['structure_classes']}
  for ob in rec.observations:
   if ob.point_id not in points:continue
   xyz=points[ob.point_id];cam=rec.rotation@xyz+rec.translation;uv=np.array(ob.undistorted_xy);x,y=np.floor(uv).astype(int)
   if not(0<=x<w and 0<=y<h) or cam[2]<=0:continue
   cl=int(lab[y,x]);pred=(K@cam)[:2]/cam[2];err=float(np.linalg.norm(pred-uv));valid=cl in dist and dist[cl][y,x]>c['label_boundary_margin_px'] and err<=c['track_max_reprojection_px']
   tracks[ob.point_id].append(dict(name=rec.name,xy=uv.tolist(),z=float(cam[2]),cl=cl,valid=bool(valid),error=err,center=(-rec.rotation.T@rec.translation).tolist()))
 byview={n:[] for n in sorted(labeled)};meta=[]
 for pid,obs in sorted(tracks.items()):
  classes={v['cl'] for v in obs}
  if len(classes)!=1 or next(iter(classes)) not in c['structure_classes']:rejected['class_inconsistent_or_excluded']+=1;continue
  # One observation per independent camera, choose lowest reprojection error deterministically.
  unique={}
  for v in obs:
   if v['valid'] and (v['name'] not in unique or v['error']<unique[v['name']]['error']):unique[v['name']]=v
  good=list(unique.values())
  if len(good)<c['track_min_views']:rejected['too_few_valid_views']+=1;continue
  dirs=np.array([v['center'] for v in good])-points[pid];dirs/=np.linalg.norm(dirs,axis=1,keepdims=True);angle=float(np.degrees(np.arccos(np.clip((dirs@dirs.T).min(),-1,1))))
  if angle<c['track_min_angle_degrees']:rejected['weak_baseline']+=1;continue
  role='check' if pid%5==0 else 'fit';meta.append(dict(point_id=pid,xyz=points[pid].tolist(),class_id=good[0]['cl'],views=len(good),max_angle=angle,role=role))
  for v in good:byview[v['name']].append(dict(point_id=pid,xy=v['xy'],inverse_z=1/v['z'],class_id=v['cl'],role=role,reprojection_error=v['error']))
 assert meta and all(any(v['class_id']==cl for v in meta) for cl in c['structure_classes'])
 write(out/'observations.json',byview);write(out/'tracks.json',meta)
 r=dict(status='completed',experiment_id=c['experiment_id'],seconds=time.monotonic()-start,source_sha256=digest(__file__),points_sha256=digest(source),official_images_sha256=digest(a.dataset_root/'camera_parameters/images.txt'),official_cameras_sha256=digest(a.dataset_root/'camera_parameters/cameras.txt'),label_hashes=labels_hash,track_counts=dict(Counter(str(v['class_id'])+'_'+v['role'] for v in meta)),observation_counts=dict(Counter(str(v['class_id'])+'_'+v['role'] for vs in byview.values() for v in vs)),views_with_fit=sum(any(v['role']=='fit' for v in vs) for vs in byview.values()),rejected=dict(rejected),observations_sha256=digest(out/'observations.json'))
 write(out/'result.json',r);(out/'source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({k:v for k,v in r.items() if k!='label_hashes'},indent=2))
if __name__=='__main__':main()
