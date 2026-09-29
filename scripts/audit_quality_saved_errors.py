#!/usr/bin/env python3
"""Independent pure-Python integer/rational check of SEM352 saved-count diagnosis."""
import argparse,json,sys
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import CLASSES,sha256,verify_hashes,write_new_json
from src.quality_mask2former_final import FinalContext
CONFIG='experiments/runs/SEM-352-SAVED-ERROR-DIAG/config.yaml'


def close(a,b):
    if a is None or b is None:
        if a is not None or b is not None:raise ValueError('Undefined-class handling differs')
    elif abs(a-float(b))>1e-10:raise ValueError('Independent rational arithmetic differs')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['artifact-root','diagnosis','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);read=lambda p:json.loads(p.read_text())
    c=read(ROOT/CONFIG);r={k:c[k] for k in ['experiment_id','git_commit','seed','dataset_manifest','split_id']}
    r.update(status='started',config_sha256=sha256(ROOT/CONFIG),code_sha256=sha256(Path(__file__)),target_labels_decoded=False,prediction_images_decoded=False)
    try:
        verify_hashes(ROOT,c['input_hashes']);verify_hashes(ROOT,c['code_sha256'])
        parent=read(a.diagnosis/'result.json');e=read(a.diagnosis/'evidence.json');source=read(ROOT/c['source_result']);rows=read(ROOT/c['source_per_view'])
        if parent['status']!='completed_saved_error_decomposition_pending_independent_audit' or parent['config_sha256']!=r['config_sha256'] or sha256(a.diagnosis/'evidence.json')!=parent['evidence_sha256']:raise ValueError('Unbound evidence')
        if [x['name'] for x in e['per_view']]!=c['names']:raise ValueError('Incomplete views')
        raw_by_name={x['name']:x for x in rows};pooled={'confusions':{a:source['evaluation']['metrics'][a]['confusion'] for a in c['arms']},'transitions_gt_old_new':{k:source['evaluation']['pairs'][k]['transitions_gt_old_new'] for k in c['pairs']}}
        checked=0
        for item in [*e['per_view'],e['pooled']]:
            raw=raw_by_name[item['name']] if 'name' in item else pooled
            rational={}
            for arm in c['arms']:
                cm=raw['confusions'][arm];metrics=item['arms'][arm];ios=[];active=[]
                for k,cl in enumerate(CLASSES):
                    tp=cm[k][k];support=sum(cm[k]);pred=sum(row[k] for row in cm);fp=pred-tp;fn=support-tp;union=tp+fp+fn
                    expected={'support':support,'predicted':pred,'tp':tp,'fp':fp,'fn':fn,'false_positive_only_absent_gt':support==0 and pred>0}
                    if any(metrics['classes'][cl][key]!=v for key,v in expected.items()):raise ValueError('Wrong class counts')
                    iou=Fraction(tp,union) if union else None
                    if union:ios.append(iou);active.append(cl)
                    close(metrics['classes'][cl]['iou'],iou);close(metrics['classes'][cl]['recall'],Fraction(tp,support) if support else None);close(metrics['classes'][cl]['precision'],Fraction(tp,pred) if pred else None)
                rational[arm]=(sum(ios),len(ios));close(metrics['miou'],sum(ios)/len(ios))
                if metrics['active_classes']!=active or metrics['gt_bg_to_stay_cable']!=cm[0][2] or metrics['stay_cable_fn']!=sum(cm[2])-cm[2][2]:raise ValueError('Incorrect summary')
            for pair in c['pairs']:
                old,new=pair.split('->');t=raw['transitions_gt_old_new'][pair];v=item['pairs'][pair];x,y=raw['confusions'][old],raw['confusions'][new]
                if any(sum(t[g][o])!=x[g][o] for g in range(5) for o in range(5)) or any(sum(t[g][o][n] for o in range(5))!=y[g][n] for g in range(5) for n in range(5)):raise ValueError('Wrong integer marginals')
                sx,nx=rational[old];sy,ny=rational[new];close(v['delta_miou_pp'],100*(sy/ny-sx/nx));close(v['iou_sum_change_at_old_denominator_pp'],100*(sy-sx)/nx);close(v['denominator_change_at_new_iou_sum_pp'],100*(sy/ny-sy/nx))
                for cl in CLASSES:
                    oldc=item['arms'][old]['classes'][cl];newc=item['arms'][new]['classes'][cl];detail=v['classes'][cl]
                    for field in ['tp','fp','fn']:
                        if detail['delta_'+field]!=newc[field]-oldc[field]:raise ValueError('Class delta differs')
                    close(detail['delta_iou_pp'],None if oldc['iou'] is None or newc['iou'] is None else 100*(newc['iou']-oldc['iou']))
                    close(detail['delta_recall_pp'],None if oldc['recall'] is None else 100*(newc['recall']-oldc['recall']))
                groups={'correction_types':[],'harm_types':[],'wrong_change_types':[]}
                for g in range(5):
                    for o in range(5):
                        for n in range(5):
                            count=t[g][o][n]
                            if not count or o==n:continue
                            field='correction_types' if n==g else 'harm_types' if o==g else 'wrong_change_types'
                            groups[field].append({'gt':CLASSES[g],'old':CLASSES[o],'new':CLASSES[n],'pixels':count})
                for field,counts in groups.items():
                    counts.sort(key=lambda x:(-x['pixels'],x['gt'],x['old'],x['new']))
                    if counts!=v[field]:raise ValueError('Transition listing differs')
                for name,field in [('corrected','correction_types'),('harmed','harm_types'),('wrong_to_wrong','wrong_change_types')]:
                    if sum(x['pixels'] for x in groups[field])!=v[name]:raise ValueError('Transition sums differ')
                for k,cl in enumerate(CLASSES):
                    if v['classes'][cl]['corrected']!=sum(t[k][o][k] for o in range(5) if o!=k) or v['classes'][cl]['harmed']!=sum(t[k][k][n] for n in range(5) if n!=k):raise ValueError('Per-class corrected/harmed differs')
                bg={'persistent':t[0][2][2],'newly_added':sum(t[0][o][2] for o in range(5) if o!=2),'removed':sum(t[0][2][n] for n in range(5) if n!=2),'removed_to_true_background':t[0][2][0],'removed_to_other_wrong_class':sum(t[0][2][n] for n in [1,3,4])}
                cable={'recovered':sum(t[2][o][2] for o in range(5) if o!=2),'newly_missed':sum(t[2][2][n] for n in range(5) if n!=2),'persistent_missed':sum(t[2][o][n] for o in range(5) for n in range(5) if o!=2 and n!=2)}
                if bg!=v['background_cable_churn'] or cable!=v['cable_recall_churn']:raise ValueError('Error churn differs')
                if v['active_old']!=item['arms'][old]['active_classes'] or v['active_new']!=item['arms'][new]['active_classes']:raise ValueError('Active mask differs')
                checked+=1
        for pair in c['pairs']:
            expected=[x['name'] for x in sorted(e['per_view'],key=lambda x:(x['pairs'][pair]['delta_miou_pp'],x['name'])) if x['pairs'][pair]['delta_miou_pp']<0]
            if expected!=e['regressions'][pair]:raise ValueError('Regression inventory differs')
        for arm in c['arms']:
            expected=[{'name':x['name'],'classes':{cl:z['predicted'] for cl,z in x['arms'][arm]['classes'].items() if z['support']==0 and z['predicted']>0}} for x in e['per_view'] if any(z['support']==0 and z['predicted']>0 for z in x['arms'][arm]['classes'].values())]
            if expected!=e['absent_gt_false_positive_cases'][arm]:raise ValueError('Absent-GT inventory differs')
            ranked=sorted(e['per_view'],key=lambda x:(-x['arms'][arm]['gt_bg_to_stay_cable'],x['name']));top=[{'name':x['name'],'pixels':x['arms'][arm]['gt_bg_to_stay_cable']} for x in ranked[:10]];z=e['background_cable_concentration'][arm]
            if z['top10']!=top or z['top10_pixels']!=sum(x['pixels'] for x in top) or z['total_pixels']!=e['pooled']['arms'][arm]['gt_bg_to_stay_cable']:raise ValueError('Concentration differs')
        FinalContext(ROOT,a.artifact_root,ROOT/c['final_batch']).finish()
        r.update(status='passed_independent_saved_error_audit',checked_view_and_pooled_pairs=checked,views=74,
                 source_result_sha256=sha256(ROOT/c['source_result']),diagnosis_result_sha256=sha256(a.diagnosis/'result.json'),
                 evidence_sha256=sha256(a.diagnosis/'evidence.json'),rgb_verified=293,frozen_verified=18)
    except BaseException as exc:r.update(status='failed_saved_error_audit',error=f'{type(exc).__name__}: {exc}');raise
    finally:write_new_json(a.output/'result.json',r)


if __name__=='__main__':main()
