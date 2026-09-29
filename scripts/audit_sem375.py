#!/usr/bin/env python3
"""Independent saved-pixel accounting for the RGB136 matched semantic fit."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from src.quality_protocol import sha256,write_new_json,CLASSES,verify_hashes

def read(p):return json.loads(p.read_text())
def counts(gt,pred):
 assert gt.shape==pred.shape==(989,1320) and gt.max()<5 and pred.max()<5
 # Separate class masks; no shared evaluation helper or encoded bincount.
 return np.asarray([[int(np.logical_and(gt==t,pred==p).sum()) for p in range(5)] for t in range(5)],dtype=np.int64)
def scores(cm):
 out={};ious=[]
 for i,c in enumerate(CLASSES):
  tp=int(cm[i,i]);fp=int(cm[:,i].sum())-tp;fn=int(cm[i].sum())-tp
  iou=tp/(tp+fp+fn) if tp+fp+fn else None
  out[c]=dict(tp=tp,fp=fp,fn=fn,IoU=iou,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None)
  if iou is not None:ious.append(iou)
 return dict(confusion=cm.tolist(),per_class=out,mIoU_with_background=sum(ious)/len(ious))
def main():
 p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);a=p.parse_args();batch=a.batch.resolve();start=time.monotonic()
 old=ROOT/'outputs/quality/backbone_comparison';sem=ROOT/'outputs/quality/all_data/SEM-370-r1';train=read(batch/'mask2former/result.json');assert train['status']=='completed'
 logs=[json.loads(x) for x in (batch/'mask2former/training.jsonl').read_text().splitlines()];reference=[json.loads(x) for x in (old/'standard/mask2former/training.jsonl').read_text().splitlines()]
 assert [r['step'] for r in logs]==list(range(1,6001));assert [r['sample'] for r in logs]==[r['sample'] for r in reference]
 assert (batch/'mask2former/resolved_model.yaml').read_bytes()==(old/'standard/mask2former/resolved_model.yaml').read_bytes()
 assert train['initial_model_state_sha256']==read(old/'standard/mask2former/result.json')['initial_model_state_sha256']
 split=read(ROOT/'outputs/quality/all_data/RGB-120-r2/split.json');names=sorted(split['semantic_train']);arms={'old_scaffold':old/'scaffold','standard':old/'standard','mip':old/'mip','new_scaffold':batch};results={};hashes={}
 for arm,path in arms.items():
  rec=read(path/'inference/result.json');assert rec['status']=='completed';total=np.zeros((5,5),np.int64);rows={x['name']:x for x in rec['per_view']};per=[]
  for n in names:
   gt=np.asarray(Image.open(sem/'labels'/n));pr=np.asarray(Image.open(path/'inference/predictions'/n));cm=counts(gt,pr);assert cm.tolist()==rows[n]['confusion'];total+=cm
   score=scores(cm);assert abs(score['mIoU_with_background']-rows[n]['mIoU_with_background'])<1e-12;per.append(dict(name=n,**score))
  score=scores(total);assert score['confusion']==rec['metrics']['confusion'];assert abs(score['mIoU_with_background']-rec['metrics']['mIoU_with_background'])<1e-12
  for c in CLASSES:assert score['per_class'][c]==rec['metrics']['per_class'][c]
  results[arm]=dict(metrics=score,per_view=per);hashes[arm]=sha256(path/'inference/result.json');print(arm,score['mIoU_with_background'],flush=True)
 new={r['name']:r for r in results['new_scaffold']['per_view']};paired={}
 for arm in ['old_scaffold','standard','mip']:
  deltas=[dict(name=r['name'],mIoU_delta_pp=100*(new[r['name']]['mIoU_with_background']-r['mIoU_with_background'])) for r in results[arm]['per_view']]
  paired[arm]=dict(global_mIoU_delta_pp=100*(results['new_scaffold']['metrics']['mIoU_with_background']-results[arm]['metrics']['mIoU_with_background']),improved=sum(r['mIoU_delta_pp']>0 for r in deltas),degraded=sum(r['mIoU_delta_pp']<0 for r in deltas),per_view=deltas)
 import torch
 checkpoint=[]
 for n in ['model-3000.pt','model-6000.pt']:
  state=torch.load(batch/'mask2former'/n,map_location='cpu',weights_only=True)['model'];assert all(torch.isfinite(v).all() for v in state.values());checkpoint.append(dict(name=n,tensors=len(state),sha256=sha256(batch/'mask2former'/n)));del state
 for old_path,identities in read(batch/'old_semantic_identities.json').items():verify_hashes(ROOT/old_path,identities)
 export=read(batch/'export/result.json');verify_hashes(batch/'export/images',export['image_hashes']);verify_hashes(batch/'mask2former',train['artifacts'])
 cfg=read(ROOT/'experiments/runs/SEM-375-SCAFFOLD-NOAPPEARANCE-ALL300/config.yaml')
 write_new_json(batch/'audit.json',dict(status='completed',experiment_id=cfg['experiment_id'],binding=cfg,config_sha256=sha256(ROOT/'experiments/runs'/cfg['experiment_id']/'config.yaml'),source_results_sha256=hashes,script_sha256=sha256(Path(__file__)),sampling6000_matches_sem371=True,resolved_recipe_matches_sem371=True,initial_state_matches_sem371=True,checkpoint_finite=checkpoint,independent_confusion_checks=1200,old_three_semantic_models_unchanged=True,rgb544_hashes_exact=True,arms=results,paired=paired,seconds=time.monotonic()-start))
if __name__=='__main__':main()
