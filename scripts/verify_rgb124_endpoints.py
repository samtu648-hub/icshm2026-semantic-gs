#!/usr/bin/env python3
"""Fresh endpoint reload, all400 pixel closure and saved official trajectory."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from src.quality_scaffold import ScaffoldQualityRenderer
from src.quality_cameras import read_camera_requests
from src.quality_backbone import CameraRequest
from src.camera_adapter import CameraFromWorld
from src.quality_protocol import sha256
p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);a=p.parse_args();c=json.loads((ROOT/'experiments/runs/RGB-124-MATCHED-DENSIFICATION-PF/config.yaml').read_text());o=ROOT/c['output'];parent=ROOT/c['parent_run'];cams=read_camera_requests(parent/'scene/sparse/0');trajectory=json.loads((ROOT/'outputs/quality/diagnostics/RGB-122-r1/trajectory.json').read_text());results=[]
for arm in c['arms']:
 r=ScaffoldQualityRenderer(a.artifact_root/'r1/third_party/scaffold-gs',o/arm/'model',7500,train_camera_count=400);sig=r.rgb_signature()
 for n,cam in cams.items():
  im=np.rint(r.render_pinhole(cam).rgb*255).astype('uint8');assert np.array_equal(im,np.asarray(Image.open(o/arm/'images'/n))),(arm,n)
 for t in trajectory:
  cam=CameraRequest(t['width'],t['height'],np.array(t['matrix']),CameraFromWorld(np.array(t['rotation']),np.array(t['translation'])));im=np.rint(r.render_pinhole(cam).rgb*255).astype('uint8');Image.fromarray(im).save(o/arm/'images'/t['view'])
 assert r.rgb_signature()==sig
 results.append(dict(arm=arm,fresh_png_exact=400,novel_cameras=8,signature=sig))
 del r
frozen=json.loads((ROOT/'experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json').read_text())['artifacts']
for x in frozen:assert sha256(a.artifact_root/x['path'])==x['sha256']
(o/'reload_audit.json').write_text(json.dumps(dict(status='passed',results=results,frozen18_exact=True,trajectory_source_sha256=sha256(ROOT/'outputs/quality/diagnostics/RGB-122-r1/trajectory.json')),indent=2)+'\n')
print('fresh800 exact; trajectory16; frozen18 exact')
