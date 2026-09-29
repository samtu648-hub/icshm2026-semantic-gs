#!/usr/bin/env python3
"""SEM352: algebraic diagnosis from sealed counts only; never decode GT/images."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import CLASSES,sha256,verify_hashes,write_new_json
from src.quality_mask2former_final import FinalContext
CONFIG='experiments/runs/SEM-352-SAVED-ERROR-DIAG/config.yaml'


def validated(value,shape):
    a=np.asarray(value)
    if a.shape!=shape or not np.issubdtype(a.dtype,np.integer) or np.any(a<0):
        raise ValueError('Invalid nonnegative integer count array')
    return a.astype(np.int64)


def describe(matrix):
    cm=validated(matrix,(5,5));classes={};ious=[];active=[]
    for k,name in enumerate(CLASSES):
        tp=int(cm[k,k]);support=int(cm[k].sum());predicted=int(cm[:,k].sum());fp=predicted-tp;fn=support-tp;union=tp+fp+fn
        iou=Fraction(tp,union) if union else None
        if iou is not None:ious.append(iou);active.append(name)
        classes[name]={'support':support,'predicted':predicted,'tp':tp,'fp':fp,'fn':fn,
                       'iou':float(iou) if iou is not None else None,
                       'recall':tp/support if support else None,'precision':tp/predicted if predicted else None,
                       'false_positive_only_absent_gt':support==0 and predicted>0}
    if not active:raise ValueError('Empty image count matrix')
    return {'classes':classes,'active_classes':active,'miou':float(sum(ious)/len(ious)),
            'gt_bg_to_stay_cable':int(cm[0,2]),'stay_cable_fn':classes['stay_cable']['fn']}


def sum_iou(cm):
    result=Fraction(0);active=0
    for k in range(5):
        tp=int(cm[k,k]);den=int(cm[k].sum()+cm[:,k].sum()-tp)
        if den:result+=Fraction(tp,den);active+=1
    return result,active


def compare(old,new,triple):
    x=validated(old,(5,5));y=validated(new,(5,5));t=validated(triple,(5,5,5))
    if not np.array_equal(t.sum(2),x) or not np.array_equal(t.sum(1),y):raise ValueError('Triple marginals differ')
    dx,dy=describe(x),describe(y);sx,nx=sum_iou(x);sy,ny=sum_iou(y)
    delta=sy/ny-sx/nx
    # Ordered exact algebra, not causal attribution or an alternative leaderboard:
    # change IoU sum at old denominator, then change denominator at new IoU sum.
    class_term=(sy-sx)/nx;denom_term=sy*(Fraction(1,ny)-Fraction(1,nx))
    if delta!=class_term+denom_term:raise ValueError('Nonexact decomposition')
    per_class={};corrections=[];harms=[];other=[]
    for k,name in enumerate(CLASSES):
        oldc,newc=dx['classes'][name],dy['classes'][name]
        per_class[name]={'delta_tp':newc['tp']-oldc['tp'],'delta_fp':newc['fp']-oldc['fp'],
                         'delta_fn':newc['fn']-oldc['fn'],
                         'delta_iou_pp':None if oldc['iou'] is None or newc['iou'] is None else 100*(newc['iou']-oldc['iou']),
                         'delta_recall_pp':None if oldc['recall'] is None else 100*(newc['recall']-oldc['recall']),
                         'corrected':sum(int(t[k,o,k]) for o in range(5) if o!=k),
                         'harmed':sum(int(t[k,k,n]) for n in range(5) if n!=k)}
    for g in range(5):
        for o in range(5):
            for n in range(5):
                count=int(t[g,o,n])
                if not count or o==n:continue
                dest=corrections if n==g else harms if o==g else other
                dest.append({'gt':CLASSES[g],'old':CLASSES[o],'new':CLASSES[n],'pixels':count})
    for seq in [corrections,harms,other]:seq.sort(key=lambda r:(-r['pixels'],r['gt'],r['old'],r['new']))
    bg={'persistent':int(t[0,2,2]),'newly_added':int(t[0,:,2].sum()-t[0,2,2]),
        'removed':int(t[0,2,:].sum()-t[0,2,2]),'removed_to_true_background':int(t[0,2,0]),
        'removed_to_other_wrong_class':int(sum(t[0,2,k] for k in [1,3,4]))}
    cable={'recovered':int(t[2,:,2].sum()-t[2,2,2]),'newly_missed':int(t[2,2,:].sum()-t[2,2,2]),
           'persistent_missed':int(sum(t[2,o,n] for o in range(5) for n in range(5) if o!=2 and n!=2))}
    assert bg['newly_added']-bg['removed']==int(y[0,2]-x[0,2])
    assert cable['newly_missed']-cable['recovered']==dy['stay_cable_fn']-dx['stay_cable_fn']
    return {'delta_miou_pp':float(delta*100),'active_old':dx['active_classes'],'active_new':dy['active_classes'],
            'iou_sum_change_at_old_denominator_pp':float(class_term*100),'denominator_change_at_new_iou_sum_pp':float(denom_term*100),
            'classes':per_class,'background_cable_churn':bg,'cable_recall_churn':cable,
            'corrected':sum(r['pixels'] for r in corrections),'harmed':sum(r['pixels'] for r in harms),
            'wrong_to_wrong':sum(r['pixels'] for r in other),'correction_types':corrections,'harm_types':harms,'wrong_change_types':other}


def diagnose(rows,source,c):
    if [r['name'] for r in rows]!=c['names']:raise ValueError('Fixed view list differs')
    arms=c['arms'];pairs=c['pairs'];per=[]
    total_cm={arm:np.zeros((5,5),np.int64) for arm in arms};total_t={pair:np.zeros((5,5,5),np.int64) for pair in pairs}
    for row in rows:
        cms={a:validated(row['confusions'][a],(5,5)) for a in arms}
        if any(not np.array_equal(cm.sum(1),cms[arms[0]].sum(1)) for cm in cms.values()):raise ValueError('GT supports differ')
        if any(int(cm.sum())!=1320*989 for cm in cms.values()):raise ValueError('Pixel count differs')
        item={'name':row['name'],'arms':{a:describe(cm) for a,cm in cms.items()},'pairs':{}}
        for pair in pairs:
            old,new=pair.split('->');t=validated(row['transitions_gt_old_new'][pair],(5,5,5))
            item['pairs'][pair]=compare(cms[old],cms[new],t);total_t[pair]+=t
        for a in arms:total_cm[a]+=cms[a]
        per.append(item)
    pooled={'arms':{a:describe(cm) for a,cm in total_cm.items()},'pairs':{}}
    for a,cm in total_cm.items():
        if cm.tolist()!=source['evaluation']['metrics'][a]['confusion']:raise ValueError('Pooled confusion differs')
        if abs(pooled['arms'][a]['miou']-source['evaluation']['metrics'][a]['mIoU_with_background'])>1e-14:raise ValueError('Pooled score differs')
    for pair,t in total_t.items():
        old,new=pair.split('->');v=compare(total_cm[old],total_cm[new],t);ref=source['evaluation']['pairs'][pair]
        if t.tolist()!=ref['transitions_gt_old_new'] or any(v[k]!=ref[k] for k in ['corrected','harmed','wrong_to_wrong']):raise ValueError('Pooled transition differs')
        pooled['pairs'][pair]=v
    regressions={}
    for pair in pairs:
        regressions[pair]=[r['name'] for r in sorted(per,key=lambda r:(r['pairs'][pair]['delta_miou_pp'],r['name'])) if r['pairs'][pair]['delta_miou_pp']<0]
    absent={a:[{'name':r['name'],'classes':{cl:v['predicted'] for cl,v in r['arms'][a]['classes'].items() if v['false_positive_only_absent_gt']}}
                 for r in per if any(v['false_positive_only_absent_gt'] for v in r['arms'][a]['classes'].values())] for a in arms}
    concentration={}
    for a in arms:
        ranked=sorted(per,key=lambda r:(-r['arms'][a]['gt_bg_to_stay_cable'],r['name']));total=pooled['arms'][a]['gt_bg_to_stay_cable']
        concentration[a]={'top10':[{"name":r['name'],'pixels':r['arms'][a]['gt_bg_to_stay_cable']} for r in ranked[:10]],
                          'top10_pixels':sum(r['arms'][a]['gt_bg_to_stay_cable'] for r in ranked[:10]),'total_pixels':total}
    return {'pooled':pooled,'per_view':per,'regressions':regressions,'absent_gt_false_positive_cases':absent,'background_cable_concentration':concentration}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);c=json.loads((ROOT/CONFIG).read_text());start=time.monotonic()
    r={k:c[k] for k in ['experiment_id','git_commit','seed','dataset_manifest','split_id']}
    r.update(status='started',config_sha256=sha256(ROOT/CONFIG),code_sha256=c['code_sha256'],target_labels_decoded=False,
             prediction_images_decoded=False,new_training=False,new_inference=False,new_policy=False)
    try:
        verify_hashes(ROOT,c['input_hashes']);verify_hashes(ROOT,c['code_sha256'])
        ctx=FinalContext(ROOT,a.artifact_root,ROOT/c['final_batch'])
        source=json.loads((ROOT/c['source_result']).read_text());rows=json.loads((ROOT/c['source_per_view']).read_text())
        if source['status']!='passed_independent_fixed74_reconstruction_and_scoring':raise ValueError('Unaccepted source')
        evidence=diagnose(rows,source,c);write_new_json(a.output/'evidence.json',evidence)
        ctx.finish();verify_hashes(ROOT,c['input_hashes']);verify_hashes(ROOT,c['code_sha256'])
        r.update(status='completed_saved_error_decomposition_pending_independent_audit',rgb_verified=293,frozen_verified=18,
                 evidence_sha256=sha256(a.output/'evidence.json'),views=len(rows))
    except BaseException as exc:r.update(status='failed_saved_error_diagnostic',error=f'{type(exc).__name__}: {exc}');raise
    finally:r['seconds']=time.monotonic()-start;write_new_json(a.output/'result.json',r)


if __name__=='__main__':main()
