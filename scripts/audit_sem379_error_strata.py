#!/usr/bin/env python3
"""Exposure and optional raw prediction changes by fixed input-detail strata."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter,minimum_filter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256
OUT=ROOT/'outputs/quality/SEM-379/r1';BASE=ROOT/'outputs/quality/SEM-378/r1';SEM=ROOT/'outputs/quality/all_data/SEM-370-r1';PHOTO=ROOT/'outputs/quality/all_data/RGB-120-r2/scene/images'
def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',action='store_true');a=p.parse_args()
 old=[json.loads(l)['sample'] for l in (BASE/'mask2former/training.jsonl').read_text().splitlines()];new=[json.loads(l) for l in (OUT/'sampling-plan.jsonl').read_text().splitlines()];pairs={n:[] for n in sorted({t['name'] for t in old})}
 for b,c in zip(old,new):pairs[b['name']].append((b,c))
 totals={};perview=[]
 for i,(n,ps) in enumerate(pairs.items()):
  gt=np.asarray(Image.open(SEM/'labels'/n));baseline=np.asarray(Image.open(BASE/'inference/predictions'/n));fn=(gt==2)&(baseline!=2);detail=[]
  for path in [PHOTO,BASE/'export/images']:
   g=np.asarray(Image.open(path/n).convert('RGB')).astype(np.float32).mean(2);detail.append((maximum_filter(g,size=5)-minimum_filter(g,size=5))>=12)
  pd,rd=detail;groups={'photo_detail_render_weak':pd&~rd,'both_have_local_detail':pd&rd,'both_low_detail':~pd&~rd,'render_only_detail':~pd&rd};cov=[]
  for col in [0,1]:
   d=np.zeros((990,1321),np.int32)
   for pair in ps:
    t=pair[col];y,x=t['top'],t['left'];d[y,x]+=1;d[y+768,x]-=1;d[y,x+768]-=1;d[y+768,x+768]+=1
   cov.append(d.cumsum(0).cumsum(1)[:989,:1320])
  pred=np.asarray(Image.open(OUT/'inference/predictions'/n)) if a.candidate else None
  row={'name':n,'groups':{}}
  for k,m in groups.items():
   m=m&fn;v={'baseline_fn_pixels':int(m.sum()),'baseline_exposure':int(cov[0][m].sum()),'candidate_exposure':int(cov[1][m].sum()),'baseline_unseen':int((m&(cov[0]==0)).sum()),'candidate_unseen':int((m&(cov[1]==0)).sum())}
   if pred is not None:
    v['fn_fixed']=int((m&(pred==2)).sum());v['fn_remaining']=int((m&(pred!=2)).sum())
    full=groups[k]&(gt==2);v['true_cable_lost']=int((full&(baseline==2)&(pred!=2)).sum())
   row['groups'][k]=v
   if k not in totals:totals[k]={key:0 for key in v}
   for key,value in v.items():totals[k][key]+=value
  perview.append(row)
  if i%50==0:print('strata',i,flush=True)
 suffix='candidate' if a.candidate else 'exposure'
 result=dict(status='completed',experiment_id='SEM-379-CABLE-REGION-SAMPLING',scope='all300 training-fit baseline errors; local detail strata are not physical visibility labels',strata=totals,per_view=perview,script_sha256=sha256(Path(__file__)))
 (OUT/f'error-strata-{suffix}.json').write_text(json.dumps(result,indent=2)+'\n');print(totals)
if __name__=='__main__':main()
