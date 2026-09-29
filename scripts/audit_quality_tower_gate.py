#!/usr/bin/env python3
"""SEM353 independent integer counts and exact rational screening, no label reads."""
import argparse,json,sys
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,verify_hashes,write_new_json
CONFIG='experiments/runs/SEM-353-TOWER-GATE-CALIBRATION/config.yaml'


def scores(cm):
    classes=[]
    for k in range(5):
        tp=cm[k][k];fp=sum(row[k] for row in cm)-tp;fn=sum(cm[k])-tp
        classes.append({'IoU':Fraction(tp,tp+fp+fn),'precision':Fraction(tp,tp+fp) if tp+fp else None,'recall':Fraction(tp,tp+fn) if tp+fn else None,'tp':tp,'fp':fp,'fn':fn})
    return sum(d['IoU'] for d in classes)/5,classes


def aggregate(rows,baseline,tag):
    cm=[[0]*5 for _ in range(5)];truth=[0]*5
    for row in rows:
        counts=row['candidates'][tag]
        for g in range(5):
            truth[g]+=counts[g]
            for pred in range(5):cm[g][pred]+=row['baseline_confusions'][baseline][g][pred]
            cm[g][2]-=counts[g];cm[g][3]+=counts[g]
    if any(v<0 for row in cm for v in row):raise ValueError('Negative reconstructed count')
    return cm,truth


def check(cm,record):
    if cm!=record['confusion']:raise ValueError('Independent confusion differs')
    mean,classes=scores(cm)
    if abs(float(mean)-record['mIoU_with_background'])>1e-14:raise ValueError('mIoU differs')
    for name,d in zip(['background','deck','stay_cable','tower','foundation'],classes):
        for field,x in d.items():
            y=record['per_class'][name][field]
            if x is None:
                if y is not None:raise ValueError('Undefined metric differs')
            elif y is None or abs(float(x)-y)>1e-14:raise ValueError('Per-class metric differs')
    return mean,classes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['run','replay','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);read=lambda p:json.loads(p.read_text());c=read(ROOT/CONFIG)
    result={k:c[k] for k in ['experiment_id','git_commit','seed','dataset_manifest','split_id']}
    result.update(status='started',config_sha256=sha256(ROOT/CONFIG),code_sha256=sha256(Path(__file__)),new_labels_decoded=0)
    try:
        verify_hashes(ROOT,c['input_hashes']);verify_hashes(ROOT,c['code_sha256'])
        r=read(a.run/'result.json');replay=read(a.replay/'result.json');rows=read(a.run/'per_view.json')
        if any(x['status']!='completed_fixed108_tower_calibration_no_selection' or x['config_sha256']!=result['config_sha256'] for x in [r,replay]) or not replay['replay_exact'] or replay['replay_source_sha256']!=sha256(a.run/'result.json'):raise ValueError('Unbound pixel replay')
        if sha256(a.run/'per_view.json')!=r['per_view_sha256'] or rows!=read(a.replay/'per_view.json'):raise ValueError('Per-view replay differs')
        if [x['name'] for x in rows]!=[n for ns in c['views'].values() for n in ns] or len(rows)!=108:raise ValueError('Wrong role')
        checks=0;passed=[]
        for tag in c['candidates']:
            flags={}
            for fold,names in c['views'].items():
                subset=[x for x in rows if x['fold']==fold]
                if [x['name'] for x in subset]!=names:raise ValueError('Wrong fold')
                for base in ['raw','selected']:
                    cm,truth=aggregate(subset,base,tag);mean,classes=check(cm,r['per_fold'][fold]['metrics'][base][tag])
                    old,oldc=scores(aggregate(subset,base,'off')[0]);flags[fold+'/'+base]=mean>=old and classes[2]['IoU']>=oldc[2]['IoU'] and classes[3]['IoU']>=oldc[3]['IoU']
                    if truth!=r['per_fold'][fold]['truth_counts'][tag]:raise ValueError('Fold truth differs')
                    checks+=1
            strict={}
            for base in ['raw','selected']:
                cm,truth=aggregate(rows,base,tag);_,classes=check(cm,r['metrics'][base][tag]);strict[base]=classes[3]['IoU']>scores(aggregate(rows,base,'off')[0])[1][3]['IoU'];checks+=1
            eligible=r['eligibility'][tag]
            expect={'per_fold_control_pass':flags,'pooled_tower_strict_gain':strict,'all_pass':all(flags.values()) and all(strict.values()),'nonzero':bool(sum(truth)),'truth_counts':truth,'corrected':truth[3],'harmed':truth[2],'wrong_to_wrong':sum(truth[k] for k in [0,1,4]),'recall_loss_requires_user_review':bool(truth[2])}
            if eligible!=expect:raise ValueError('Exact screening differs')
            if expect['all_pass'] and expect['nonzero']:passed.append(tag)
        if passed!=r['nonzero_candidates_passed'] or r['selected_margin'] is not None or r['automatic_adoption']:raise ValueError('Selection/promotion invalid')
        if r['decision']!=('pending_quality_review' if passed else 'keep_off_close_finite_tower_margin_family'):raise ValueError('Stop rule violated')
        result.update(status='passed_independent_tower_integer_fraction_audit',views=108,fold_and_pooled_control_checks=checks,nonzero_candidates_passed=passed,
                      decision=r['decision'],run_result_sha256=sha256(a.run/'result.json'),replay_result_sha256=sha256(a.replay/'result.json'),per_view_sha256=r['per_view_sha256'])
    except BaseException as exc:result.update(status='failed_tower_integer_audit',error=f'{type(exc).__name__}: {exc}');raise
    finally:write_new_json(a.output/'result.json',result)


if __name__=='__main__':main()
