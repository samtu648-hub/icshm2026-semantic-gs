#!/usr/bin/env python3
"""SEM356 matched saved-count comparison; no labels, fitting or selection."""
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
    c=read(ROOT/'experiments/runs/SEM-356-SELECTOR-WEIGHT-ABLATION/config.yaml');old=ROOT/c['paired_parent_batch'];base=read(old/'result.json');new=read(a.batch/'result.json')
    for batch,result in [(old,base),(a.batch,new)]:
        seal=read(batch/'predictions_sealed.json');verify_hashes(batch,seal['files'])
        if result['status']!='completed_fixed_development_evaluation':raise ValueError('Incomplete comparison')
    audit=[]
    for role in c['roles']:
        prev=read(old/role['id']/'fit.json');now=read(a.batch/role['id']/'fit.json')
        for key in ('role','fit_labels_accessed','fit_gt_counts','sampling','fallback'):
            if prev[key]!=now[key]:raise ValueError('Matched fit controls differ: '+key)
        audit.append(dict(head=role['id'],training_views=len(role['fit']),sample_rows=sum(x['rows'] for x in now['sampling']),parent_model_sha256=prev['model_sha256'],candidate_model_sha256=now['model_sha256']))
    previous_rows={r['name']:r for r in read(old/'per_view.json')}
    for row in read(a.batch/'per_view.json'):
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
    result=dict(experiment_id=c['experiment_id'],code_sha256=sha256(Path(__file__)),config_sha256=sha256(ROOT/'experiments/runs/SEM-356-SELECTOR-WEIGHT-ABLATION/config.yaml'),parent_result_sha256=sha256(old/'result.json'),candidate_result_sha256=sha256(a.batch/'result.json'),fit_sample_role_audit=audit,matched_comparator_count_checks=324,pooled=compare(base['pooled'],new['pooled']),folds={f:compare(base['folds'][f],new['folds'][f]) for f in c['views']},heads={r['id']:compare(base['heads'][r['id']],new['heads'][r['id']]) for r in c['roles']},boundary='historically adaptive original108 development; no blind or full-chain claim')
    write_new_json(a.output,result);print(json.dumps(result['pooled'],indent=2))
if __name__=='__main__':main()
