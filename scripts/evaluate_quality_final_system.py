#!/usr/bin/env python3
"""SEM350 fixed historical74 comparison: seal all predictions before labels."""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,verify_hashes,write_new_json
from src.quality_mask2former_final import FinalContext
from src.quality_fusion import pipeline,confusion
from src.quality_raw_foundation_gate import apply_gate
from src.quality_semantic_audit import metrics,independent_confusion
from src.quality_static_training import TrainingLabels
CONFIG='experiments/runs/SEM-350-FIXED-FINAL-COMPARISON/config.yaml'
ARMS=['old_sem161','new_b2','new_raw','new_fused','new_selected']


def read(p):return json.loads(p.read_text())
def png(p):
    with Image.open(p) as im:return np.array(im)


def sources(c,a):
    result={}
    for key in ['fit','static','deck','new_b2']:
        base=ROOT/c[key];result[key]=(base,read(base/'result.json'))
    base=a.artifact_root/c['old_sem161'];result['old_sem161']=(base,read(base/'prediction_manifest.json'))
    return result


def artifact(base,record,relative,kind):
    verify_hashes(base,{relative:record['artifacts'][relative]})
    if kind=='png':return png(base/relative)
    with np.load(base/relative,allow_pickle=False) as z:return z['probabilities']


def construct(name,s,c,independent=False):
    base,record=s['fit'];five=artifact(base,record,'probabilities/'+Path(name).stem+'.npz','npz')
    raw=artifact(base,record,'predictions/'+name,'png')
    assert np.array_equal(five.argmax(0),raw)
    base,record=s['static'];static=artifact(base,record,'predictions/'+name,'png')
    base,record=s['deck'];deck=artifact(base,record,'probabilities/'+Path(name).stem+'.npz','npz')[1]
    if not independent:
        fused=pipeline(static,five,np.zeros_like(deck),deck,c['thresholds'])['deck']
        selected=apply_gate(raw,fused,five,c['margin'])
    else:
        fused=static.copy();lane=(static==0)|(static==2)
        fused[lane]=np.where(five[2][lane]>=.35,2,0)
        pre=fused.copy();fused[(fused==4)&(five[4]<.1)]=0
        fused[(fused==0)&(pre==0)&(deck>=.98)]=1
        selected=raw.copy()
        selected[(raw==3)&(static==4)&(five[4]>=.1)&((five[3]-five[4])<=.05)]=4
    assert not np.any((selected!=raw)&~((raw==3)&(selected==4)))
    return {'new_raw':raw,'new_fused':fused,'new_selected':selected}


def score(c,a,s,seal,replay=False):
    labels=TrainingLabels(a.artifact_root/'r2/sem_001/labels',c['names'],read(ROOT/c['label_manifest']),[1320,989])
    per=[]
    for name in c['names']:
        images={}
        for arm,e in seal['predictions'][name].items():
            base=ROOT if e['root']=='repo' else a.artifact_root
            verify_hashes(base,{e['path']:e['sha256']});images[arm]=png(base/e['path'])
        if replay:
            independent=construct(name,s,c,True)
            assert all(np.array_equal(images[k],v) for k,v in independent.items())
        target=labels.read(name)
        cms={k:(independent_confusion if replay else confusion)(target,images[k]) for k in ARMS}
        pairs={}
        for old,new in c['pairs']:
            matrix=np.bincount(target.ravel().astype(np.int64)*25+images[old].ravel().astype(np.int64)*5+images[new].ravel(),minlength=125).reshape(5,5,5)
            assert np.array_equal(matrix.sum(2),cms[old]) and np.array_equal(matrix.sum(1),cms[new])
            pairs[old+'->'+new]=matrix.tolist()
        per.append({'name':name,'confusions':{k:v.tolist() for k,v in cms.items()},'transitions_gt_old_new':pairs})
    total={k:metrics(sum(np.array(row['confusions'][k]) for row in per)) for k in ARMS}
    old=read(a.artifact_root/c['old_sem161']/'result.json')
    assert total['old_sem161']['confusion']==old['metrics']['confusion_matrix_rows_gt_columns_prediction']
    assert total['new_b2']==s['new_b2'][1]['evaluation']['development']['metrics']
    groups={}
    for group,names in c['groups'].items():
        rows=[r for r in per if r['name'] in names];assert len(rows)==len(names)
        groups[group]={arm:{'pooled':metrics(sum(np.array(r['confusions'][arm]) for r in rows)),
                          'mean_view_miou':float(np.mean([metrics(np.array(r['confusions'][arm]))['mIoU_with_background'] for r in rows]))} for arm in ARMS}
    pairs={}
    for old,new in c['pairs']:
        key=old+'->'+new;m=sum(np.array(row['transitions_gt_old_new'][key]) for row in per)
        corrected=sum(int(m[g,p,g]) for g in range(5) for p in range(5) if p!=g)
        harmed=sum(int(m[g,g,p]) for g in range(5) for p in range(5) if p!=g)
        changed=sum(int(m[g,o,n]) for g in range(5) for o in range(5) for n in range(5) if o!=n)
        pairs[key]={'corrected':corrected,'harmed':harmed,'wrong_to_wrong':changed-corrected-harmed,'transitions_gt_old_new':m.tolist()}
    return {'metrics':total,'groups':groups,'pairs':pairs},per


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['predict','evaluate','replay'],required=True)
    p.add_argument('--predictions',type=Path);p.add_argument('--evaluation',type=Path)
    a=p.parse_args()
    if a.mode!='predict' and a.predictions is None:p.error('Score requires prior prediction seal')
    if (a.mode=='replay')!=(a.evaluation is not None):p.error('Replay requires original evaluation')
    if not a.output.resolve().is_relative_to(ROOT/'outputs/quality/final_comparison'):p.error('Use isolated final_comparison output')
    a.output.mkdir(parents=True,exist_ok=False);c=read(ROOT/CONFIG)
    r={k:c[k] for k in ['experiment_id','git_commit','seed','dataset_manifest','split_id']}
    r.update(status='started',mode=a.mode,config_sha256=sha256(ROOT/CONFIG),code_sha256=c['code_sha256'],
             started_utc=datetime.now(timezone.utc).isoformat(),target_labels_decoded=False,
             no_new_training=True,no_new_inference=True,automatic_promotion=False)
    try:
        verify_hashes(ROOT,c['input_hashes']);verify_hashes(ROOT,c['code_sha256']);verify_hashes(a.artifact_root,c['artifact_hashes'])
        ctx=FinalContext(ROOT,a.artifact_root,ROOT/c['final_batch']);ctx.finish()
        if shutil.disk_usage(ROOT).free<100*1024**3:raise OSError('100GiB reserve')
        for n in [CONFIG,*c['code_sha256']]:
            d=a.output/'code'/n;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,d)
        s=sources(c,a)
        assert read(ROOT/c['final_batch']/'queue/result.json')['status']=='completed_final219_fit_and_two74_fresh_replays_no_quality_scoring'
        assert read(ROOT/c['final_batch']/'final-audit/result.json')['status']=='passed_independent_final219_artifact_audit_no_quality_scoring'
        assert s['fit'][1]['access']['prediction']==c['names']
        old_records={e['name']:e for e in s['old_sem161'][1]['predictions']}
        if a.mode=='predict':
            records={}
            for name in c['names']:
                arrays=construct(name,s,c);entries={}
                for arm in ARMS:
                    if arm in ['new_fused','new_selected']:
                        file=a.output/arm/name;file.parent.mkdir(exist_ok=True);Image.fromarray(arrays[arm]).save(file)
                        entry={'root':'repo','path':str(file.resolve().relative_to(ROOT)),'sha256':sha256(file)}
                    elif arm=='old_sem161':
                        e=old_records[name];file=s[arm][0]/e['prediction'];assert sha256(file)==e['prediction_sha256']
                        entry={'root':'artifact','path':str(file.relative_to(a.artifact_root)),'sha256':sha256(file)}
                    else:
                        key='fit' if arm=='new_raw' else 'new_b2';rel=('predictions/' if key=='fit' else 'development/deck/')+name
                        file=s[key][0]/rel;verify_hashes(s[key][0],{rel:s[key][1]['artifacts'][rel]})
                        entry={'root':'repo','path':str(file.relative_to(ROOT)),'sha256':sha256(file)}
                    entries[arm]=entry
                records[name]=entries
            seal={'experiment_id':c['experiment_id'],'config_sha256':r['config_sha256'],'code_sha256':c['code_sha256'],
                  'sealed_utc':datetime.now(timezone.utc).isoformat(),'target_labels_decoded':False,'predictions':records}
            write_new_json(a.output/'seal.json',seal)
            r.update(status='sealed_all74_five_fixed_arms_before_labels',seal_sha256=sha256(a.output/'seal.json'))
        else:
            source=read(a.predictions/'result.json');seal=read(a.predictions/'seal.json')
            assert source['status']=='sealed_all74_five_fixed_arms_before_labels'
            assert sha256(a.predictions/'seal.json')==source['seal_sha256']
            assert seal['config_sha256']==r['config_sha256'] and seal['code_sha256']==c['code_sha256'] and seal['target_labels_decoded'] is False
            assert list(seal['predictions'])==c['names']
            evaluation,per=score(c,a,s,seal,a.mode=='replay')
            if a.mode=='replay':
                original=read(a.evaluation/'result.json');assert original['status']=='completed_fixed74_evaluation_pending_replay'
                assert original['evaluation']==evaluation and read(a.evaluation/'per_view.json')==per
                assert original['per_view_sha256']==sha256(a.evaluation/'per_view.json')
                r['replay_source_sha256']=sha256(a.evaluation/'result.json')
            write_new_json(a.output/'per_view.json',per)
            r.update(status='passed_independent_fixed74_reconstruction_and_scoring' if a.mode=='replay' else 'completed_fixed74_evaluation_pending_replay',
                     evaluation=evaluation,per_view_sha256=sha256(a.output/'per_view.json'),target_labels_decoded=74,
                     predictions_seal_sha256=sha256(a.predictions/'seal.json'))
        ctx.finish();r.update(rgb_verified=293,frozen_verified=18)
    except BaseException as exc:
        r.update(status='failed_fixed_comparison_preserved',error=f'{type(exc).__name__}: {exc}');raise
    finally:write_new_json(a.output/'result.json',r)


if __name__=='__main__':main()
