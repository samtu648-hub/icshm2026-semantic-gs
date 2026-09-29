#!/usr/bin/env python3
"""Independent saved-PNG, selection, copy and frozen-source audit for RGB121."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
c=json.loads((ROOT/'experiments/runs/RGB-121-LOCAL-ISOLATION-DIAG/config.yaml').read_text());o=ROOT/c['output'];r=json.loads((o/'result.json').read_text());s=json.loads((o/'selection.json').read_text());b=json.loads((o/'binding.json').read_text());parent=ROOT/c['parent_run'];batch=ROOT/c['semantic_batch'];checks=0
for row in r['rows']:
 n=row['view'];base=np.asarray(Image.open(o/'images'/('base-'+n))).astype('int16');im=np.asarray(Image.open(o/'images'/(row['group']+'-'+n))).astype('int16');photo=np.asarray(Image.open(parent/'scene/images'/n)).astype('int16');lab=np.asarray(Image.open(batch/'labels'/n));delta=np.abs(im-base);masks={'frame':np.ones(lab.shape,bool),'protected':binary_dilation((lab>0)&(lab<5),iterations=4)}
 for k,spec in c['rois'].items():
  if spec['view']==n:
   mask=np.zeros(lab.shape,bool)
   for x0,y0,x1,y1 in spec['boxes']:mask[y0:y1,x0:x1]=True
   masks[k]=mask
 for k,m in masks.items():
  v=dict(pixels=int(m.sum()),changed_pixels=int((delta.max(2)[m]>0).sum()),delta_rgb_mae=float(delta[m].mean()),max_rgb_delta=float(delta[m].max()),photo_mae_before=float(np.abs(base-photo)[m].mean()),photo_mae_after=float(np.abs(im-photo)[m].mean()))
  assert all(abs(v[key]-row['metrics'][k][key])<1e-10 for key in v),(n,k);checks+=1
for candidate in s['candidates']:
 safe=all(x['max_Talpha']<=c['protection_max_contribution'] for x in candidate['protection']);assert safe==(candidate['id'] in s['accepted']['union']);assert candidate['id']==candidate['anchor_id']*10+candidate['offset_id']
for p,h in b['parent_artifacts'].items():assert sha(parent/p)==h
for p,h in b['copy_artifacts'].items():assert sha(o/'model'/p)==h
frozen=json.loads((ROOT/'experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json').read_text())['artifacts']
for x in frozen:assert sha(Path(sys.argv[1])/x['path'])==x['sha256']
# Exact closure against all previously exported baselines.
for n in c['views']:assert np.array_equal(np.asarray(Image.open(o/'images'/('base-'+n))),np.asarray(Image.open(batch/'export/images'/n)))
a=dict(status='passed',independent_metric_regions=checks,png_ablations=len(r['rows']),baseline_exact=len(c['views']),frozen_artifacts_exact=len(frozen),selection_and_stable_id_mapping_exact=True,parent_copy_exact=True)
(o/'audit.json').write_text(json.dumps(a,indent=2)+'\n');print(json.dumps(a))
