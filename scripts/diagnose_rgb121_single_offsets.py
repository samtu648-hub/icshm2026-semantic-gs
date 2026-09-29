#!/usr/bin/env python3
"""Decompose the already fixed safe set; no new candidate or threshold search."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from src.quality_scaffold import ScaffoldQualityRenderer
from src.quality_cameras import read_camera_requests
from src.quality_native_gaussian import make_native_view
from src.scaffold_semantic_sidecar import _render_color_channels
from src.quality_protocol import sha256
c=json.loads((ROOT/'experiments/runs/RGB-121-LOCAL-ISOLATION-DIAG/config.yaml').read_text());o=ROOT/c['output'];s=json.loads((o/'selection.json').read_text());out=o/'single';out.mkdir(exist_ok=False);start=time.monotonic()
parent=ROOT/c['parent_run'];pr=json.loads((parent/'result.json').read_text());artifact=Path(sys.argv[1]);r=ScaffoldQualityRenderer(artifact/'r1/third_party/scaffold-gs',o/'model',30000,train_camera_count=400);sig=r.rgb_signature();cams=read_camera_requests(parent/'scene/sparse/0')
from gaussian_renderer import generate_neural_gaussians,prefilter_voxel
bg=torch.zeros(3,device='cuda');rows=[];torch.set_num_threads(4)
with torch.inference_mode():
 for n in c['views']:
  v=make_native_view(cams[n]);vis=prefilter_voxel(v,r.model,r.pipe,bg);xyz,rgb,op,sc,rot,_,active=generate_neural_gaussians(v,r.model,vis,is_training=True);ids=(torch.nonzero(vis).flatten()[:,None]*r.model.n_offsets+torch.arange(r.model.n_offsets,device='cuda')).flatten()[active]
  base=np.asarray(Image.open(o/'images'/('base-'+n)));photo=np.asarray(Image.open(parent/'scene/images'/n));err=np.abs(base.astype(float)-photo).mean(2)
  for sid in s['accepted']['union']:
   opacity=op.clone();opacity[ids==sid]=0;im=_render_color_channels((xyz,opacity,sc,rot,active),rgb,v,bg);im=np.rint(im.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype('uint8');delta=np.abs(im.astype(float)-photo).mean(2)-err;diff=np.abs(im.astype(float)-base).max(2);m=diff>0
   row=dict(view=n,id=sid,changed_pixels=int(m.sum()),photo_mae_delta=float(delta.mean()),changed_pixel_mae_delta=float(delta[m].mean()) if m.any() else 0.,roi={})
   for k,spec in c['rois'].items():
    if spec['view']!=n:continue
    roi=np.zeros(delta.shape,bool)
    for x0,y0,x1,y1 in spec['boxes']:roi[y0:y1,x0:x1]=True
    row['roi'][k]=dict(photo_mae_delta=float(delta[roi].mean()),changed_pixels=int((m&roi).sum()))
   rows.append(row)
   if n in ['159.png','160.png','174.png']:
    Image.fromarray(im).save(out/f'{sid}-{n}')
  print(n,flush=True)
assert sig==r.rgb_signature()
assert all(sha256(parent/k)==h for k,h in pr['artifacts'].items())
(out/'result.json').write_text(json.dumps(dict(status='completed_fixed_set_single_offset_decomposition',experiment_id=c['experiment_id'],selection_sha256=sha256(o/'selection.json'),script_sha256=sha256(Path(__file__)),parent_and_memory_unchanged=True,rows=rows,seconds=time.monotonic()-start),indent=2)+'\n')
