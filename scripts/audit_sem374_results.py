#!/usr/bin/env python3
"""CPU endpoint and summary attestation for SEM374, no repeated GPU work."""
import json,sys,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from scripts.run_sem373_joint import digest,write,tensor_hash,schedule

def read(p):return json.loads(p.read_text())
def main():
 c=read(ROOT/'experiments/runs/SEM-374-STANDARD-STRUCTURE-DEPTH/config.yaml');base=ROOT/c['output'];out=base/'final-audit.json';assert not out.exists();checks={}
 pre=read(base/'preflight-r4/result.json');assert pre['status']=='completed';data=read(base/'tracks-independent-audit.json');assert data['status']=='passed' and data['observation_hash']==digest(ROOT/c['structure_observations']);checks['geometry_and_gradient_preflight']=True
 controls={a:read(base/(a+'-r2/result.json')) for a in ['control','joint']};assert all(x['status']=='completed' and x['successful_steps']==2000 and x['label_visits']==1500 for x in controls.values());assert controls['control']['initial_state']==controls['joint']['initial_state'];assert controls['control']['sampling_sha256']==controls['joint']['sampling_sha256'];checks['matched_model_both_adams_and_sampling']=True
 split=read(ROOT/c['split_path']);expected=schedule(split['rgb_train'],2000,c['seed']+1);parent=torch.load(ROOT/c['semantic_checkpoint'],map_location='cpu',weights_only=False)
 for n,v in parent['parameters'].items():assert tensor_hash(v)==controls['control']['initial_state']['model'][n]
 assert tensor_hash(parent['logits'])==controls['control']['initial_state']['logits']
 for opt in ['rgb_optimizer','semantic_optimizer']:
  for ident,vals in parent[opt]['state'].items():
   for k,v in vals.items():
    h=tensor_hash(v) if torch.is_tensor(v) else v;assert h==controls['control']['initial_state'][opt][str(ident)][k]
 for arm,r in controls.items():
  path=base/(arm+'-r2');assert read(path/'sampling.json')==expected and digest(path/'endpoint.pt')==r['endpoint_sha256'];assert digest(path/'source.py')==r['source_script_sha256'];assert digest(ROOT/'scripts/run_sem374_structure.py')==r['source_script_sha256']
  for rel,h in {**r['inputs'],**r['structure_inputs']}.items():assert digest(ROOT/rel)==h
  q=torch.load(path/'endpoint.pt',map_location='cpu',weights_only=False);assert q['step']==2000
  for n,v in q['parameters'].items():
   assert torch.isfinite(v).all();assert tensor_hash(v)==r['final_state']['model'][n]
   if n in ['_scaling','_rotation','_opacity']:assert torch.equal(v,parent['parameters'][n])
  assert tensor_hash(q['logits'])==r['final_state']['logits'];assert len(q['parameters']['_xyz'])==652298
  for group,key in [('rgb_optimizer','rgb_optimizer'),('semantic_optimizer','semantic_optimizer')]:
   for ident,vals in q[group]['state'].items():
    for k,v in vals.items():
     h=tensor_hash(v) if torch.is_tensor(v) else v;assert h==r['final_state'][key][str(ident)][k]
  logs=[json.loads(l) for l in (path/'training.jsonl').read_text().splitlines()];assert len(logs)==2000 and [v['name'] for v in logs]==expected;assert [v['step'] for v in logs]==list(range(1,2001));assert sum(v['semantic_nll'] is not None for v in logs)==1500
  assert all(np.isfinite(v['rgb_loss']) and np.isfinite(v['structure_loss']) for v in logs)
  del q
 checks['finite_endpoints_frozen_geometry_complete_training']=True
 evals={a:read(base/('eval-'+a+'-r2/result.json')) for a in ['baseline','control','joint']}
 assert len({json.dumps(r['trajectory_hashes'],sort_keys=True) for r in evals.values()})==1
 for arm,r in evals.items():
  assert r['status']=='completed';rows=r['per_view'];assert len(rows)==400 and {v['name'] for v in rows}==set(split['rgb_train']);assert {v['name'] for v in rows if 'semantic' in v}==set(split['semantic_train'])
  for metric in ['PSNR','SSIM','LPIPS','MAE']:assert abs(float(np.mean([v[metric] for v in rows]))-r['rgb_metrics'][metric])<1e-12
  cm=np.sum([v['semantic']['confusion'] for v in rows if 'semantic' in v],axis=0);assert cm.tolist()==r['semantic_metrics']['confusion']
 identity=read(ROOT/'outputs/quality/SEM-373/data_identity.json')
 assert all(digest(ROOT/c['scene']/'images'/n)==h for n,h in identity['rgb_sha256'].items())
 assert all(digest(ROOT/c['labels']/n)==h for n,h in identity['label_png_sha256'].items())
 checks['all400_photos_all300_labels_unchanged']=True
 checks['all400_all300_roles_and_aggregates']=True
 counts=data['counts']
 for arm,path in [('baseline',base/'depth-baseline-r2/structure-depth.json'),('control',base/'eval-control-r2/structure-depth.json'),('joint',base/'eval-joint-r2/structure-depth.json')]:
  sums=collections.Counter()
  for row in read(path):sums[str(row['class_id'])+'_'+row['role']]+=row['count']
  assert dict(sums)==counts
 checks['identical_fit_check_depth_observations']=True
 bm=read(base/'browser-r1/result.json');assert bm['audited_fixed_pairs']==27;checks['independent_fixed9_pixel_counts_and_rgb']=True
 write(out,dict(status='passed',experiment_id=c['experiment_id'],checks=checks,source_hashes={str(p.relative_to(ROOT)):digest(p) for p in ROOT.glob('scripts/*sem374*.py')},old_baseline_reused_readonly=True,failed_control_preserved=dict(steps=865,result='control-r1/result.json')))
 print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
