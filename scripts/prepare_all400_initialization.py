#!/usr/bin/env python3
"""Build an all-400-view DLT initialization without changing cameras or images."""
import argparse, json, shutil, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_rgb_colmap_scene import (find_image_paths, parse_colmap_images,
    parse_single_simple_radial_camera, undistort_observations, triangulate_tracks,
    projection_matrix, write_colmap_text)
from src.quality_protocol import sha256, write_new_json

REGIONS={
 '159-shore':('159.png',[(20,720,290,870),(920,720,1310,860)]),
 '159-near':('159.png',[(20,890,240,985)]),
 '160-shore':('160.png',[(20,740,285,885),(920,740,1310,875)]),
 '160-near':('160.png',[(20,905,230,985)]),
 '159-far-occlusion':('159.png',[(720,260,1100,610)]),
 '146-tower-cable':('146.png',[(370,100,950,760)]),
 '057-remote':('057.png',[(580,170,1250,620)]),
 '064-remote':('064.png',[(520,150,1250,620)]),
}

def write_ply(path, points):
    xyz=np.asarray([p['xyz'] for p in points],np.float32)
    rgb=np.asarray([p['rgb'] for p in points],np.uint8)
    dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('nx','<f4'),('ny','<f4'),('nz','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    out=np.zeros(len(points),dtype=dtype);out['x'],out['y'],out['z']=xyz.T;out['red'],out['green'],out['blue']=rgb.T
    header=(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode()
    with path.open('xb') as f:f.write(header);out.tofile(f)

def q(values):
    return {str(x):float(np.quantile(values,x)) for x in (0,.1,.5,.9,1)} if values else {}

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--baseline-scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-mean-reprojection-error-px',type=float,default=3.0);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 source=find_image_paths(a.dataset_root); records=parse_colmap_images(a.dataset_root/'camera_parameters/images.txt');names=sorted(source)
 if len(names)!=400 or set(names)!={r.name for r in records}:raise ValueError('Expected exactly the released 400-image camera universe')
 _,w,h,f,cx,cy,k=parse_single_simple_radial_camera(a.dataset_root/'camera_parameters/cameras.txt');K=np.array([[f,0,cx],[0,f,cy],[0,0,1.]],float)
 undistort_observations(records,K,k)
 points,color_jobs=triangulate_tracks(records,K,a.max_mean_reprojection_error_px)
 shutil.copytree(a.baseline_scene,a.output,symlinks=True)
 sparse=a.output/'sparse/0'
 for x in ('cameras.txt','images.txt','points3D.txt','points3D.ply'):(sparse/x).unlink(missing_ok=True)
 write_colmap_text(sparse,records,w,h,f,cx,cy,points)
 # RGB colors use the same first-observation sampling convention as the existing preparation code.
 import cv2
 for name,jobs in color_jobs.items():
  im=cv2.imread(str(a.output/'images'/name),cv2.IMREAD_COLOR)
  for idx,u,v,_ in jobs:
   if 0<=u<w and 0<=v<h:
    b,g,r=im[v,u];points[idx]['rgb']=(int(r),int(g),int(b))
 (sparse/'points3D.txt').unlink();write_colmap_text(sparse,records,w,h,f,cx,cy,points);write_ply(sparse/'points3D.ply',points)
 byid=defaultdict(list)
 for r in records:
  for o in r.observations:
   if o.point_id>=0:byid[o.point_id].append((r,o))
 accepted={x['point_id']:x for x in points};lens=[];errs=[];angles=[];region={key:[] for key in REGIONS}
 for pid,pt in accepted.items():
  tr=byid[pid]; xyz=np.asarray(pt['xyz']); centers=np.asarray([-r.rotation.T@r.translation for r,_ in tr]);rays=xyz-centers;rays/=np.linalg.norm(rays,axis=1,keepdims=True)
  dots=np.clip(rays@rays.T,-1,1);angles.append(float(np.degrees(np.arccos(dots.min()))));lens.append(len(tr));errs.append(float(pt['mean_reprojection_error_px']))
  for key,(view,boxes) in REGIONS.items():
   for r,o in tr:
    if r.name!=view:continue
    pr=projection_matrix(r,K)@np.append(xyz,1.);u,v=pr[:2]/pr[2]
    if any(x0<=u<x1 and y0<=v<y1 for x0,y0,x1,y1 in boxes):region[key].append((len(tr),float(pt['mean_reprojection_error_px']),angles[-1]));break
 diagnostics=dict(point_count=len(points),track_length_quantiles=q(lens),mean_reprojection_error_px_quantiles=q(errs),maximum_triangulation_angle_deg_quantiles=q(angles),tracks_total=len(byid),accepted_track_fraction=float(len(points)/len(byid)),regions={k:dict(projected_points=len(v),track_length_quantiles=q([x[0] for x in v]),mean_reprojection_error_px_quantiles=q([x[1] for x in v]),maximum_triangulation_angle_deg_quantiles=q([x[2] for x in v])) for k,v in region.items()})
 manifest=dict(schema_version=1,experiment_id='RGB-137-STANDARD-ALL400-INIT400',source_coordinate_frame='official_colmap_reference',camera_policy='official CAMERA_FROM_WORLD poses unchanged; SIMPLE_RADIAL full-frame cv2.undistort to same-K PINHOLE, no crop',initialization=dict(method='same DLT retriangulation of released images.txt shared POINT3D_ID tracks',training_views=400,source_observations=sum(len(x.observations) for x in records),max_mean_reprojection_error_px=a.max_mean_reprojection_error_px,additional_filtering='none',official_points3D='not released'),baseline_scene=str(a.baseline_scene),baseline_initial_ply_sha256=sha256(a.baseline_scene/'sparse/0/points3D.ply'),candidate_initial_ply_sha256=sha256(sparse/'points3D.ply'),source_camera_hashes={x:sha256(a.dataset_root/'camera_parameters'/x) for x in ('cameras.txt','images.txt')},diagnostics=diagnostics)
 write_new_json(a.output/'initialization_manifest.json',manifest);print(json.dumps(manifest,ensure_ascii=False))
if __name__=='__main__':main()
