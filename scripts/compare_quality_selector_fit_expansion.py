#!/usr/bin/env python3
"""SEM361 matched evaluation and original-fit-sample comparison; no labels, fitting or selection."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,verify_hashes,write_new_json

def read(p):return json.loads(Path(p).read_text())
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    c=read(ROOT/'experiments/runs/SEM-361-SELECTOR-FIT-VIEW-EXPANSION/config.yaml');old=ROOT/c['paired_parent_batch'];base=read(old/'result.json');new=read(a.batch/'result.json')
    for batch,result in [(old,base),(a.batch,new)]:
        seal=read(batch/'predictions_sealed.json');verify_hashes(batch,seal['files'])
        if result['status']!='completed_fixed_development_evaluation':raise ValueError('Incomplete comparison')
    audit=[]
    for role in c['roles']:
        prev=read(old/role['id']/'fit.json');now=read(a.batch/role['id']/'fit.json')
        original_role=prev['role']
        if now['role']!=role or any(original_role[k]!=role[k] for k in ('id','fold','evaluation')):
            raise ValueError('Evaluation role or head identity drift')
        if role['fit'][:len(original_role['fit'])]!=original_role['fit'] or not set(original_role['guard'])<=set(role['guard']):
            raise ValueError('Original fit/guard lost')
        if now['fit_labels_accessed']!=sorted(role['fit']) or now['sampling'][:len(original_role['fit'])]!=prev['sampling']:
            raise ValueError('Original samples or expanded fit whitelist differ')
        audit.append(dict(head=role['id'],old_training_views=len(original_role['fit']),training_views=len(role['fit']),sample_rows=sum(x['rows'] for x in now['sampling']),parent_model_sha256=prev['model_sha256'],candidate_model_sha256=now['model_sha256']))
    original_config=read(ROOT/c['parent_config'])
    if c['learner']!=original_config['learner'] or c['sources']!=original_config['sources']:
        raise ValueError('Non-fit training recipe differs')
    previous_rows={r['name']:r for r in read(old/'per_view.json')}
    current_rows=read(a.batch/'per_view.json')
    if sorted(r['name'] for r in current_rows)!=sorted(previous_rows):raise ValueError('Evaluation coverage differs')
    for row in current_rows:
        oldrow=previous_rows[row['name']]
        if row['fold']!=oldrow['fold'] or row['head']!=oldrow['head']:raise ValueError('Evaluation role differs')
        for arm in ('raw','foundation005','historical_fused'):
            if row['confusions'][arm]!=oldrow['confusions'][arm]:raise ValueError('Comparator pixels differ')
    def compare(a,b):
        x=a['metrics']['selector'];y=b['metrics']['selector'];delta={}
        for cid,name in enumerate(x['per_class']):
            oldm=x['per_class'][name];newm=y['per_class'][name]
            delta[name]={k+'_pp':100*(newm[k]-oldm[k]) if newm[k] is not None and oldm[k] is not None else None for k in ('IoU','precision','recall')}
            delta[name].update(tp_change=newm['tp']-oldm['tp'],fp_change=newm['fp']-oldm['fp'],fn_change=newm['fn']-oldm['fn'])
        return dict(delta_miou_pp=100*(y['mIoU_with_background']-x['mIoU_with_background']),per_class=delta,corrected_change=int(sum(b['corrected_by_class'])-sum(a['corrected_by_class'])),harmed_change=int(sum(b['harmed_by_class'])-sum(a['harmed_by_class'])))
    result=dict(experiment_id=c['experiment_id'],code_sha256=sha256(Path(__file__)),config_sha256=sha256(ROOT/'experiments/runs/SEM-361-SELECTOR-FIT-VIEW-EXPANSION/config.yaml'),parent_result_sha256=sha256(old/'result.json'),candidate_result_sha256=sha256(a.batch/'result.json'),fit_sample_role_audit=audit,matched_comparator_count_checks=324,pooled=compare(base['pooled'],new['pooled']),folds={f:compare(base['folds'][f],new['folds'][f]) for f in c['views']},heads={r['id']:compare(base['heads'][r['id']],new['heads'][r['id']]) for r in c['roles']},boundary='historically adaptive original108 development; no blind or full-chain claim')
    write_new_json(a.output,result);print(json.dumps(result['pooled'],indent=2))
if __name__=='__main__':main()
