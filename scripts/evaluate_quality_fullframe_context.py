#!/usr/bin/env python3
"""SEM346 single fixed 2->0 counterfactual; calibration only, no output policy."""
import argparse,json,sys,shutil,time
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,verify_hashes,write_new_json
from src.quality_static_training import TrainingLabels
from src.quality_fusion import confusion
from src.quality_semantic_audit import metrics
from src.quality_cable_diagnostic import largest_component
CONFIG='experiments/runs/SEM-346-FULLFRAME-CONTEXT-DIAG/config.yaml'
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['artifact-root','batch','output']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);c=read(ROOT/CONFIG);r={k:c[k] for k in ['experiment_id','git_commit','seed','dataset_manifest','split_id']};r.update(status='started',config_sha256=sha256(ROOT/CONFIG),code_sha256=sha256(Path(__file__)),threshold_selection=False,new_policy_outputs=False,check_labels_decoded=False,automatic_adoption=False)
 try:
  verify_hashes(ROOT,c['input_hashes']);queue=read(a.batch/'result.json');assert queue['status']=='completed_fullframe108_and_exact_fresh_replay';r['queue_sha256']=sha256(a.batch/'result.json')
  parent=read(ROOT/c['comparison']/'result.json');pc=read(ROOT/'experiments/runs/SEM-340-MASK2FORMER-PREFLIGHT/config.yaml');names=[n for v in c['views'].values() for n in v];labels=TrainingLabels(a.artifact_root/'r2/sem_001/labels',names,read(ROOT/pc['label_manifest']),[1320,989]);groups={};per=[];examples=[]
  for fold in range(3):
   run=a.batch/f'full-oof-{fold}';source=read(run/'result.json');replay=read(a.batch/f'replay-oof-{fold}/result.json')
   if not replay['replay_exact'] or replay['replay_source_sha256']!=sha256(run/'result.json'):raise ValueError('Missing exact fresh replay')
   verify_hashes(run,source['artifacts']);acc={k:np.zeros((5,5),np.int64) for k in ['raw','whole_frame','veto_counterfactual']};counts=np.zeros(5,np.int64)
   for name in c['views'][f'oof-{fold}']:
    rel='calibration/mask2former_6k/raw/'+name;verify_hashes(ROOT/c['comparison'],{rel:parent['artifacts'][rel]})
    with Image.open(ROOT/c['comparison']/rel) as im:raw=np.array(im)
    with np.load(run/(Path(name).stem+'.npz'),allow_pickle=False) as z:full=z['probabilities'].argmax(0).astype(np.uint8)
    target=labels.read(name);mask=(raw==2)&(full==0);truth=np.bincount(target[mask],minlength=5);veto=raw.copy();veto[mask]=0
    cm={k:confusion(target,x) for k,x in [('raw',raw),('whole_frame',full),('veto_counterfactual',veto)]};expected=cm['raw'].copy();expected[:,2]-=truth;expected[:,0]+=truth
    if not np.array_equal(expected,cm['veto_counterfactual']):raise ValueError('Independent counterfactual algebra mismatch')
    if any(not np.array_equal(raw==i,veto==i) for i in [1,3,4]):raise ValueError('Protected classes changed')
    for k in acc:acc[k]+=cm[k]
    counts+=truth;per.append({'name':name,'fold':fold,'veto_truth_counts':truth.tolist(),'confusions':{k:x.tolist() for k,x in cm.items()}})
    for kind,m in [('corrected_background',mask&(target==0)),('lost_true_stay_cable',mask&(target==2))]:
     item=largest_component(m)
     if item:examples.append({'name':name,'fold':fold,'kind':kind,**item})
   groups[str(fold)]={'metrics':{k:metrics(x) for k,x in acc.items()},'veto_truth_counts':counts.tolist()}
  total={k:sum(np.array(g['metrics'][k]['confusion']) for g in groups.values()) for k in ['raw','whole_frame','veto_counterfactual']};base=parent['evaluation']['calibration']['metrics']['mask2former_6k/raw']
  if metrics(total['raw'])!=base:raise ValueError('Raw aggregate differs from historical calibration')
  r.update(status='completed_fullframe_context_calibration_diagnostic',views=len(names),per_fold=groups,metrics={k:metrics(x) for k,x in total.items()},veto_truth_counts=np.sum([g['veto_truth_counts'] for g in groups.values()],axis=0).tolist(),examples=sum([sorted([e for e in examples if e['kind']==kind],key=lambda x:(-x['pixels'],x['name']))[:3] for kind in ['corrected_background','lost_true_stay_cable']],[]))
  from src.quality_mask2former_training import Context
  Context(ROOT,a.artifact_root,'oof-0').finish();r.update(rgb_exports_verified=219,frozen_verified=18)
  write_new_json(a.output/'per_view.json',per);r['per_view_sha256']=sha256(a.output/'per_view.json')
 except BaseException as exc:r.update(status='failed_fullframe_calibration_diagnostic',error=f'{type(exc).__name__}: {exc}');raise
 finally:write_new_json(a.output/'result.json',r)
if __name__=='__main__':main()
