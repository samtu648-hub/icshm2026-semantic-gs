#!/usr/bin/env python3
"""Bounded identity/role/result checks; no repeated GPU inference or training."""
import json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_sem373_joint import digest,write,schedule

def read(p):return json.loads(p.read_text())
def main():
 c=read(ROOT/'experiments/runs/SEM-373-STANDARD-JOINT-XYZ/config.yaml');base=ROOT/c['output'];out=base/'final-audit.json';assert not out.exists();r=dict(experiment_id=c['experiment_id'],checks={})
 split=read(ROOT/c['split_path']);train={a:read(base/(a+'-r1')/'result.json') for a in ['semantic','control','joint']};ev={a:read(base/('eval-'+a+'-r1')/'result.json') for a in ['baseline','control','joint']}
 attest=read(base/'joint-r1/completion-attestation.json')
 assert attest['verified_matched_initial_model_both_optimizers_and_sampling'] and attest['additional_training_steps']==0
 assert train['joint']['status']=='failed_preserved' and attest['original_result_sha256']==digest(base/'joint-r1/result.json')
 assert all(d['status']=='completed' for d in [train['semantic'],train['control'],*ev.values()])
 for arm,d in train.items():
  seq=read(base/(arm+'-r1')/'sampling.json');rows=[json.loads(x) for x in (base/(arm+'-r1')/'training.jsonl').read_text().splitlines()];sem=arm=='semantic';n=c['semantic_steps'] if sem else c['matched_steps'];names=split['semantic_train'] if sem else split['rgb_train'];expected=schedule(names,n,c['seed']+(0 if sem else 1))
  assert seq==expected==[x['name'] for x in rows] and [x['step'] for x in rows]==list(range(1,n+1)) and d['successful_steps']==n
  assert set(Counter(seq).values())=={20 if sem else 5}
  assert all((x['semantic_nll'] is not None)==(x['name'] in split['semantic_train']) for x in rows)
  assert digest(base/(arm+'-r1')/'endpoint.pt')==d['endpoint_sha256']
  assert digest(base/(arm+'-r1')/'source.py')==d['source_script_sha256']
  for rel,h in d['inputs'].items():assert digest(ROOT/rel)==h
  r['checks'][arm]=dict(successful_steps=n,complete_epochs=20 if sem else 5,semantic_visits=sum(x['semantic_nll'] is not None for x in rows),sampling_and_checkpoint_exact=True,source_snapshot_exact=True)
 assert train['control']['initial_state']==train['joint']['initial_state']
 for n in ['_scaling','_rotation','_opacity']:
  assert len({v['initial_state']['model'][n] for v in train.values()}|{v['final_state']['model'][n] for v in train.values()})==1
 assert train['semantic']['initial_state']['model']==train['semantic']['final_state']['model']
 assert train['semantic']['initial_state']['rgb_optimizer']==train['semantic']['final_state']['rgb_optimizer']==train['control']['initial_state']['rgb_optimizer']
 r['checks']['matched_model_and_both_optimizer_states_exact']=True
 r['checks']['frozen_semantic_rgb_parameters_and_paired_shapes_opacity_exact']=True
 for arm,d in ev.items():
  assert len(d['per_view'])==400 and {x['name'] for x in d['per_view']}==set(split['rgb_train'])
  rows=[x for x in d['per_view'] if 'semantic' in x];assert len(rows)==300 and {x['name'] for x in rows}==set(split['semantic_train'])
  total=[[sum(x['semantic']['confusion'][i][j] for x in rows) for j in range(5)] for i in range(5)];assert total==d['semantic_metrics']['confusion'] and sum(map(sum,total))==300*1320*989
  for k,v in d['rgb_metrics'].items():assert abs(sum(x[k] for x in d['per_view'])/400-v)<1e-12
  r['checks'][arm+'_all400_all300_aggregation']=True
 assert ev['baseline']['trajectory_hashes']==ev['control']['trajectory_hashes']==ev['joint']['trajectory_hashes']
 original=ROOT/c['parent_export'];new=base/'eval-baseline-r1/rgb';names=split['rgb_train']+ev['baseline']['trajectory_names'];assert len(names)==544
 assert all(digest(new/n)==digest(original/n) for n in names)
 r['checks']['original_rgb130_all544_png_files_exact']=True
 # Original input identities sealed during CPU preparation; raw data checked once here.
 identity=read(base/'data_identity.json')
 assert all(digest(ROOT/c['scene']/'images'/n)==h for n,h in identity['rgb_sha256'].items())
 assert all(digest(ROOT/c['labels']/n)==h for n,h in identity['label_png_sha256'].items())
 r['checks']['all400_photos_all300_manual_labels_unchanged']=True
 prior=read(base/'provenance.json')
 revisions={x['path']:x for x in read(base/'source-revisions.json')}
 for rel,h in prior['dirty_source_hashes'].items():
  if rel in revisions:
   assert revisions[rel]['before']==h and digest(ROOT/rel)==revisions[rel]['after']
  else:assert digest(ROOT/rel)==h
 r['checks']['declared_source_hashes_verified_with_explicit_audit_only_revision']=True
 r['checks']['joint_completed_endpoint_recovered_without_extra_training']=attest
 browser=read(base/'browser-r1/result.json');assert browser['audited_fixed_pairs']==27
 r['checks']['fixed9_independent_pixel_audit_pairs']=27
 r['source_hashes']={str(p.relative_to(ROOT)):digest(p) for p in sorted(ROOT.glob('scripts/*sem373*.py'))}
 r['failed_attempts']=[dict(attempt=n,status=read(base/n/'result.json')['status'],error=read(base/n/'result.json')['error']) for n in ['preflight-r1','joint-r1']]
 stages=[read(p) for p in base.glob('*/result.json') if p.parent.name in ['preflight-r1','preflight-r2','semantic-r1','trained-gradient-r1','control-r1','joint-r1','eval-baseline-r1','eval-control-r1','eval-joint-r1','geometry-counterfactual-r1']]
 r['resources']=dict(stage_wall_seconds_excluding_recorded_lock_wait=sum(d['seconds'] for d in stages),gpu_stage_lock_wait_seconds=sum(d.get('lock_wait_seconds',0) for d in stages),allocated_peak_mib=max(d.get('peak_allocated_mib',0) for d in stages),training_loop_seconds=sum(d['training_seconds'] for d in train.values()),output_bytes=sum(p.stat().st_size for p in base.rglob('*') if p.is_file()),note='stage wall includes CPU loading/encoding; not pure GPU kernel timing; failed preflight included')
 r['status']='passed';write(out,r);print(json.dumps(r['checks'],indent=2))

if __name__=='__main__':main()
