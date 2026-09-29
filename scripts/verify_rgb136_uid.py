#!/usr/bin/env python3
"""Verify saved native no-appearance model is exactly UID-independent."""
import argparse,fcntl,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_scaffold import ScaffoldQualityRenderer
from src.quality_cameras import read_camera_requests
from src.quality_native_gaussian import make_native_view
from src.scaffold_semantic_sidecar import render_frozen_rgb
from src.quality_protocol import sha256,write_new_json,verify_hashes
p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--iteration',type=int,required=True);a=p.parse_args();o=a.run.resolve();rec=json.loads((o/'result.json').read_text());assert rec['status'] in ['completed_smoke','completed_all400'];verify_hashes(o,rec['artifacts'])
with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 import torch
 torch.set_num_threads(4)
 r=ScaffoldQualityRenderer(a.artifact_root/'r1/third_party/scaffold-gs',o/'model',a.iteration,train_camera_count=400);assert r.model.appearance_dim==0 and r.model.embedding_appearance is None;sig=r.rgb_signature();cams=read_camera_requests(o/'scene/sparse/0');rows=[]
 with torch.inference_mode():
  for n in ['159.png','160.png','174.png']:
   ref=None
   for uid in [0,158,399,999]:
    v=make_native_view(cams[n]);v.uid=uid;im=render_frozen_rgb(v,r.model,r.pipe)
    if ref is None:ref=im.clone()
    else:assert torch.equal(ref,im),(n,uid)
    rows.append(dict(view=n,uid=uid,exact=True))
 assert sig==r.rgb_signature();verify_hashes(o,rec['artifacts']);write_new_json(o/'uid_audit.json',dict(status='passed',script_sha256=sha256(Path(__file__)),iteration=a.iteration,appearance_dim=0,embedding_absent=True,signature=sig,cases=rows))
print('UID independence: 12 native renders exactly equal per camera; checkpoint unchanged')
