#!/usr/bin/env python3
"""Independent 2x2 pixel accounting without training or parameter selection."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from src.quality_protocol import sha256,write_new_json,verify_hashes,CLASSES
from scripts.audit_sem375 import counts,scores

def read(p):return json.loads(p.read_text())
def arm_paths(b):
 standard=ROOT/'outputs/quality/backbone_comparison/standard';scaffold=ROOT/'outputs/quality/SEM-375/r1'
 return {'standard_sem371':(standard,standard),'standard_sem375':(standard,b/'standard_sem375'),'scaffold_sem371':(scaffold,b/'scaffold_sem371'),'scaffold_sem375':(scaffold,scaffold)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);a=p.parse_args();cfgpath=ROOT/'experiments/runs/SEM-376-CROSS-RGB-INFERENCE/config.yaml';cfg=read(cfgpath);b=ROOT/cfg['output_path'];paths=arm_paths(b);split=read(ROOT/cfg['split_path']);labelroot=ROOT/'outputs/quality/all_data/SEM-370-r1/labels';start=time.monotonic();arms={};source_hashes={}
 for arm,(rgb,sem) in paths.items():
  rec=read(sem/'inference/result.json');assert rec['status']=='completed';rows={x['name']:x for x in rec['per_view']};total=np.zeros((5,5),np.int64);view={}
  for n in sorted(split['semantic_train']):
   gt=np.asarray(Image.open(labelroot/n));pred=np.asarray(Image.open(sem/'inference/predictions'/n));cm=counts(gt,pred);assert cm.tolist()==rows[n]['confusion'];score=scores(cm);assert abs(score['mIoU_with_background']-rows[n]['mIoU_with_background'])<1e-12;total+=cm;view[n]=score
  score=scores(total);assert score['confusion']==rec['metrics']['confusion'];assert abs(score['mIoU_with_background']-rec['metrics']['mIoU_with_background'])<1e-12;arms[arm]=dict(metrics=score,per_view=view);source_hashes[arm]=sha256(sem/'inference/result.json');print(arm,score['mIoU_with_background'],flush=True)
 pairs={}
 def pair(label,before,after):
  ds=[dict(name=n,delta_pp=100*(arms[after]['per_view'][n]['mIoU_with_background']-arms[before]['per_view'][n]['mIoU_with_background'])) for n in arms[before]['per_view']]
  pairs[label]=dict(before=before,after=after,delta_pp=100*(arms[after]['metrics']['mIoU_with_background']-arms[before]['metrics']['mIoU_with_background']),improved=sum(x['delta_pp']>0 for x in ds),degraded=sum(x['delta_pp']<0 for x in ds),per_view=ds,class_iou_delta_pp={c:100*(arms[after]['metrics']['per_class'][c]['IoU']-arms[before]['metrics']['per_class'][c]['IoU']) for c in CLASSES})
 pair('swap_semantic_on_standard','standard_sem371','standard_sem375');pair('swap_semantic_on_scaffold','scaffold_sem371','scaffold_sem375');pair('swap_rgb_for_sem371','standard_sem371','scaffold_sem371');pair('swap_rgb_for_sem375','standard_sem375','scaffold_sem375')
 roicfg=read(ROOT/'experiments/runs/RGB-132-THREE-BACKBONE-DIAG/config.yaml');regions=roicfg['regions'].copy()
 for kind,key,path in [('difficult','rois','QUAL-005-PHOTO-INPUT-DIAG'),('structure','crops','QUAL-004-CURRENT-BOTTLENECK')]:
  for stem,box in read(ROOT/'experiments/runs'/path/'config.yaml')[key].items():regions[kind+'-'+stem]=dict(view=stem+'.png',boxes=[box])
 region_rows=[]
 for name,spec in regions.items():
  n=spec['view'];gt=np.asarray(Image.open(labelroot/n));mask=np.zeros(gt.shape,bool)
  for x0,y0,x1,y1 in spec['boxes']:mask[y0:y1,x0:x1]=True
  preds={arm:np.asarray(Image.open(sem/'inference/predictions'/n)) for arm,(rgb,sem) in paths.items()};ms={arm:scores(np.bincount(gt[mask].astype(np.int64)*5+pred[mask],minlength=25).reshape(5,5)) for arm,pred in preds.items()};changes={}
  for key,pairdata in pairs.items():
   before=preds[pairdata['before']];after=preds[pairdata['after']];correct=(before!=gt)&(after==gt)&mask;harm=(before==gt)&(after!=gt)&mask
   changes[key]=dict(corrected=int(correct.sum()),harmed=int(harm.sum()),per_class={c:dict(corrected=int((correct&(gt==i)).sum()),harmed=int((harm&(gt==i)).sum())) for i,c in enumerate(CLASSES)})
  region_rows.append(dict(region=name,view=n,boxes=spec['boxes'],metrics=ms,changes=changes))
 for arm in cfg['arms']:
  fresh=read(b/arm/'fresh/result.json');assert fresh['fresh_probability_exact']==9 and fresh['fresh_png_exact']==9
  rec=read(b/arm/'inference/result.json');verify_hashes(b/arm/'inference/predictions',rec['prediction_hashes'])
 for rgb,_ in paths.values():verify_hashes(rgb/'export/images',read(rgb/'export/result.json')['image_hashes'])
 verify_hashes(ROOT,read(b/'input_hashes.json'))
 for path,h in read(ROOT/'outputs/quality/scaffold_noappearance/old_model_identities.json').items():verify_hashes(ROOT/path,h)
 new=ROOT/'outputs/quality/scaffold_noappearance/full-r1';verify_hashes(new,read(new/'result.json')['artifacts'])
 frozen={x['path']:x['sha256'] for x in read(ROOT/'experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json')['artifacts']};verify_hashes(a.artifact_root,frozen)
 write_new_json(b/'audit.json',dict(status='completed',experiment_id=cfg['experiment_id'],binding=cfg,config_sha256=sha256(cfgpath),arms=arms,pairs=pairs,interaction_pp=pairs['swap_semantic_on_scaffold']['delta_pp']-pairs['swap_semantic_on_standard']['delta_pp'],regions=region_rows,independent_confusion_checks=1200,fresh_probability_exact=18,frozen18_exact=True,rgb_models_unchanged=True,rgb_inputs_unchanged=True,semantic_checkpoints_unchanged=True,training_steps=0,source_results_sha256=source_hashes,script_sha256=sha256(Path(__file__)),seconds=time.monotonic()-start))
if __name__=='__main__':main()
