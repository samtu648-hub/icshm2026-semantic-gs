#!/usr/bin/env python3
"""Fixed pre-existing ROI accounting, with explicit corrected/harmed pixels."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from src.quality_protocol import write_new_json,sha256,CLASSES
from src.quality_semantic_audit import metrics

def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);a=p.parse_args();b=a.batch.resolve();base=ROOT/'outputs/quality/backbone_comparison';labelroot=ROOT/'outputs/quality/all_data/SEM-370-r1/labels';cfgpaths=['experiments/runs/RGB-132-THREE-BACKBONE-DIAG/config.yaml','experiments/runs/QUAL-005-PHOTO-INPUT-DIAG/config.yaml','experiments/runs/QUAL-004-CURRENT-BOTTLENECK/config.yaml']
 regions=read(ROOT/cfgpaths[0])['regions'].copy()
 for kind,key,path in [('difficult','rois',cfgpaths[1]),('structure','crops',cfgpaths[2])]:
  for stem,box in read(ROOT/path)[key].items():regions[kind+'-'+stem]=dict(view=stem+'.png',boxes=[box])
 paths={'old_scaffold':base/'scaffold','standard':base/'standard','mip':base/'mip','new_scaffold':b};rows=[]
 for region,spec in regions.items():
  n=spec['view'];gt=np.asarray(Image.open(labelroot/n));mask=np.zeros(gt.shape,bool)
  for x0,y0,x1,y1 in spec['boxes']:mask[y0:y1,x0:x1]=True
  preds={arm:np.asarray(Image.open(path/'inference/predictions'/n)) for arm,path in paths.items()};new=preds['new_scaffold'];arm_metrics={};changes={}
  for arm,pred in preds.items():
   cm=np.bincount((gt[mask].astype(np.int64)*5+pred[mask]),minlength=25).reshape(5,5);arm_metrics[arm]=metrics(cm)
   if arm=='new_scaffold':continue
   corrected=(pred!=gt)&(new==gt)&mask;harmed=(pred==gt)&(new!=gt)&mask;changed=(pred!=new)&mask
   changes[arm]=dict(changed=int(changed.sum()),corrected=int(corrected.sum()),harmed=int(harmed.sum()),still_wrong_changed=int((changed&(pred!=gt)&(new!=gt)).sum()),per_class={c:dict(corrected=int((corrected&(gt==i)).sum()),harmed=int((harmed&(gt==i)).sum())) for i,c in enumerate(CLASSES)})
  rows.append(dict(region=region,view=n,boxes=spec['boxes'],pixels=int(mask.sum()),metrics=arm_metrics,new_vs=changes))
 write_new_json(b/'region_audit.json',dict(status='completed',experiment_id='SEM-375-SCAFFOLD-NOAPPEARANCE-ALL300',source_roi_configs_sha256={p:sha256(ROOT/p) for p in cfgpaths},script_sha256=sha256(Path(__file__)),regions=rows))
if __name__=='__main__':main()
