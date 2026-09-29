#!/usr/bin/env python3
"""SEM363: one unique219 OOF fit, sealed74 final transfer, no model search."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_quality_selector_weight_ablation import read,load,sources
import numpy as np
from PIL import Image
from src.quality_protocol import sha256,verify_hashes,write_new_json
from src.quality_learned_selector import sample_rows,feature_rows,ridge_fit,predict
from src.quality_static_training import TrainingLabels,semantic_access
from src.quality_experts import verify_export
from src.quality_expert_final_pair import enforce_transfer_budget
from src.quality_semantic_audit import metrics
from scripts.diagnose_quality_selector_fit_gap import cube,summary

CONFIG='experiments/runs/SEM-363-FINAL-LEARNED-TRANSFER/config.yaml'
CODE=['scripts/run_quality_final_selector.py','scripts/audit_quality_final_selector.py','src/quality_learned_selector.py',
      'src/quality_expert_final_pair.py','src/quality_protocol.py','src/quality_static_training.py','src/quality_experts.py',
      'src/quality_semantic_audit.py','scripts/run_quality_selector_weight_ablation.py','scripts/diagnose_quality_selector_fit_gap.py',
      'scripts/audit_quality_selector_weight_ablation.py','scripts/diagnose_quality_raw_fusion.py','scripts/diagnose_quality_complementarity.py']


def guard(a,c):
    if time.time()>a.deadline:raise TimeoutError('Registered final CPU-stage deadline')
    enforce_transfer_budget(ROOT,read(ROOT/c['b2_config']))


def final_sources(c, include_b2=True):
    return {k:(ROOT/v,read(ROOT/v/'result.json')) for k,v in c['final_sources'].items() if include_b2 or k!='b2'}


def freeze(c,artifact_root):
    cfg=read(ROOT/c['b2_config']);proto=read(ROOT/cfg['protocol'])
    verify_hashes(ROOT,proto['input_hashes'])
    rgb=ROOT/cfg['study']['rgb_run'];parent=read(rgb/'result.json')
    verify_hashes(rgb,parent['artifacts'])
    train=ROOT/cfg['study']['rendered_root'];identity=read(train/'result.json')['identity']
    verify_export(train,c['train_names'],sha256(rgb/'result.json'),identity,'semantic-training')
    verify_export(ROOT/cfg['study']['target_root'],c['prediction_names'],sha256(rgb/'result.json'),identity,'development')
    frozen=read(ROOT/proto['frozen_manifest'])
    verify_hashes(artifact_root,{r['path']:r['sha256'] for r in frozen['artifacts']})
    return dict(rgb=293,frozen=len(frozen['artifacts']),identity=identity)


def provenance(a,c):
    verify_hashes(ROOT,c['input_hashes'])
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=c['git_commit']:
        raise ValueError('Git drift')
    pc=read(ROOT/c['parent_config'])
    if c['learner']!=pc['learner'] or c['sources']!=pc['sources'] or c['seed']!=pc['seed']:
        raise ValueError('Unregistered learner or OOF source change')
    if not a.fit_only:
        b2=read(ROOT/c['b2_run']/'result.json');audit=read(ROOT/c['b2_audit']/'result.json')
        if b2['trained_steps']!=6000 or b2['endpoints']!=[3000,6000] or audit['status']!='completed_independent_expert_replay' or audit['run_result_sha256']!=sha256(ROOT/c['b2_run']/'result.json'):
            raise ValueError('Unaccepted final B2 dependency')
    frozen=freeze(c,a.artifact_root)
    cfg=read(ROOT/c['b2_config']);split=read(ROOT/'experiments/splits/sem_001_scaffold_static_v1.json');oof=read(ROOT/'experiments/splits/sem_005_oof_spatial_v1.json')
    rgb_access=read(ROOT/cfg['study']['rgb_run']/'access.json')
    final_access=semantic_access(split,oof,rgb_access,'development')
    if c['train_names']!=final_access['train'] or c['prediction_names']!=final_access['prediction']:
        raise ValueError('Final219/74 roles differ')
    proof={};views={};fold_by_name={}
    for fold in c['sources']:
        expected=semantic_access(split,oof,rgb_access,fold)
        views[fold]=expected['prediction']
        for n in views[fold]:
            if n in fold_by_name:raise ValueError('Duplicate OOF fit view')
            fold_by_name[n]=fold
        for kind,(folder,r) in sources(c,fold).items():
            if any(r['access'][k]!=expected[k] for k in ('train','prediction','guard','partition')) or set(c['prediction_names'])&set(r['access']['train']):
                raise ValueError('OOF/final target label leakage')
            ident=r['binding'].get('identity',r['binding'].get('rgb_identity'))
            if ident!=frozen['identity'] or r['binding']['seed']!=c['seed']:
                raise ValueError('OOF identity or seed differs')
            modelnames={'five':['model.pt'],'static':['semantic_field.pt'],'b2':['expert-3000.pt','expert-6000.pt'],'deck':['expert.pt'],'stay_cable':['expert.pt']}[kind]
            verify_hashes(folder,{n:r['artifacts'][n] for n in modelnames})
            proof[fold+'/'+kind]=sha256(folder/'result.json')
    if sorted(fold_by_name)!=c['train_names'] or len(fold_by_name)!=219:
        raise ValueError('OOF union is not unique219')
    for kind,(folder,r) in final_sources(c,include_b2=not a.fit_only).items():
        if any(r['access'][k]!=final_access[k] for k in ('train','prediction','guard','partition')):
            raise ValueError('Final source role differs')
        if r['binding'].get('identity',r['binding'].get('rgb_identity'))!=frozen['identity'] or r['binding']['seed']!=c['seed']:
            raise ValueError('Final source identity differs')
        modelnames={'five':['model.pt'],'static':['semantic_field.pt'],'b2':['expert-3000.pt','expert-6000.pt'],'deck':['expert.pt'],'stay_cable':['expert.pt']}[kind]
        verify_hashes(folder,{n:r['artifacts'][n] for n in modelnames})
        proof['final/'+kind]=sha256(folder/'result.json')
    parent_batch=ROOT/'outputs/quality/learned_selector/SEM-361-r1'
    parent=read(parent_batch/'result.json');verify_hashes(ROOT,parent['code_sha256'])
    verify_hashes(parent_batch,read(parent_batch/'predictions_sealed.json')['files'])
    if parent['config_sha256']!=sha256(ROOT/c['parent_config']):raise ValueError('SEM361 config mismatch')
    samples={}
    for role in pc['roles']:
        for sample in read(parent_batch/role['id']/'fit.json')['sampling']:
            if sample['name'] in samples and sample!=samples[sample['name']]:raise ValueError('Repeated OOF sample differs')
            samples[sample['name']]=sample
    if sorted(samples)!=c['train_names']:raise ValueError('Unique219 sample inventory differs')
    return dict(views=views,fold_by_name=fold_by_name,original_samples=samples,source_result_hashes=proof,
                b2_audit_sha256=None if a.fit_only else sha256(ROOT/c['b2_audit']/'result.json'),freeze=frozen,final_dependency_pending=bool(a.fit_only))


def fit(a,c,proof):
    labels=TrainingLabels(a.artifact_root/'r2/sem_001/labels',c['train_names'],read(ROOT/'experiments/runs/SEM-001/label_manifest.json'),[1320,989])
    counts=np.zeros(5,np.int64);journal=[];xs=[];ys=[];weights=[];started=time.time()
    for name in c['train_names']:
        guard(a,c);fold=proof['fold_by_name'][name];data=load(sources(c,fold),name)
        ix,cls=sample_rows(data,c['seed'],fold,name,c['learner']['max_pixels_per_view'])
        row=dict(name=name,rows=len(ix),pixels=len(np.unique(ix)),sample_sha256=hashlib.sha256(np.column_stack([ix,cls]).astype('<i8').tobytes()).hexdigest())
        if row!=proof['original_samples'][name]:raise ValueError('Original fit sampling drift')
        gt=labels.read(name);counts+=np.bincount(gt.ravel(),minlength=5)
        if len(ix):
            xs.append(feature_rows(data,ix,cls));truth=gt.ravel()[ix]
            ys.append((truth==cls).astype(float)-(truth==data[0].ravel()[ix]).astype(float));weights.append(np.ones(len(ix))/len(ix))
        journal.append(row);del data,gt
    if np.any(counts==0) or not xs:raise ValueError('Final fit lacks registered class support')
    x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(weights);del xs,ys,weights
    model=dict(ridge_fit(x,y,w,c['learner']['ridge']),fallback=np.array(False));del x,y,w
    np.savez(a.output/'model.npz',**model)
    write_new_json(a.output/'fit.json',dict(training_views=c['train_names'],fit_labels_accessed=sorted(labels.accessed),
        fit_gt_counts=counts.tolist(),sampling=journal,sample_rows=sum(r['rows'] for r in journal),model_sha256=sha256(a.output/'model.npz'),
        fit_seconds=time.time()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,novel_fits=1))


def score(a,c):
    seal=read(a.output/'predictions_sealed.json');verify_hashes(a.output,seal['files'])
    labels=TrainingLabels(a.artifact_root/'r2/sem_001/labels',c['prediction_names'],read(ROOT/'experiments/runs/SEM-001/label_manifest.json'),[1320,989])
    previous={r['name']:r for r in read(ROOT/c['baseline_comparison']/'per_view.json')}
    if sorted(previous)!=c['prediction_names']:raise ValueError('Historical74 differs')
    independent={r['name']:r['cube'] for r in read(a.output/'audit.json')['rows']}
    five,fr=final_sources(c)['five'];rows=[]
    from scripts.diagnose_quality_complementarity import checked_image
    for name in c['prediction_names']:
        guard(a,c);gt=labels.read(name);raw=checked_image(five,fr,'predictions/'+name)
        pred=np.asarray(Image.open(a.output/'predictions'/name));v=cube(gt,raw,pred)
        if v.tolist()!=independent[name] or v.sum(2).tolist()!=previous[name]['confusions']['new_raw']:
            raise ValueError('Independent cube or raw comparator drift')
        rows.append(dict(name=name,gt_raw_selector_cube=v.tolist(),confusions={**{k:previous[name]['confusions'][k] for k in ('new_raw','new_selected','new_fused','old_sem161')},'selector':v.sum(1).tolist()}))
    def aggregate(part):
        s=summary(r['gt_raw_selector_cube'] for r in part)
        s['comparators']={arm:metrics(sum(np.array(r['confusions'][arm],np.int64) for r in part)) for arm in ('new_raw','new_selected','new_fused','old_sem161')}
        s['delta_vs_comparator_pp']={k:100*(s['selector']['mIoU_with_background']-v['mIoU_with_background']) for k,v in s['comparators'].items()}
        return s
    groups=read(ROOT/'experiments/runs/SEM-350-FIXED-FINAL-COMPARISON/config.yaml')['groups']
    result=dict(pooled=aggregate(rows),groups={k:aggregate([r for r in rows if r['name'] in ns]) for k,ns in groups.items()},labels_accessed=sorted(labels.accessed))
    historical=read(ROOT/c['baseline_comparison']/'result.json')['evaluation']['metrics']
    for k,v in result['pooled']['comparators'].items():
        if v!=historical[k]:raise ValueError('Historical74 aggregate mismatch')
    write_new_json(a.output/'per_view.json',rows)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fit-only',action='store_true');p.add_argument('--resume-fit',type=Path)
    a=p.parse_args();a.output=a.output.resolve()
    if a.fit_only and a.resume_fit:p.error('Fit-only and resume are separate phases')
    if not a.output.is_relative_to(ROOT/'outputs/quality/final_transfer'):p.error('Isolated transfer output required')
    a.output.mkdir(parents=True,exist_ok=False);c=read(ROOT/CONFIG);started=time.time();a.deadline=started+900
    rec={k:c[k] for k in ('experiment_id','git_commit','seed','dataset_manifest','split_id')}
    rec.update(status='started',config_sha256=sha256(ROOT/CONFIG),code_sha256={n:sha256(ROOT/n) for n in CODE},new_gpu_training=False,automatic_promotion=False)
    write_new_json(a.output/'started.json',rec)
    for n in [CONFIG,*CODE]:
        dest=a.output/'code'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dest)
    try:
        proof=provenance(a,c);write_new_json(a.output/'provenance.json',proof)
        if a.resume_fit:
            old=read(a.resume_fit/'result.json')
            if old['status']!='completed_fit_only_pending_final_B2' or old['config_sha256']!=rec['config_sha256']:
                raise ValueError('Unaccepted saved final fit')
            verify_hashes(a.resume_fit,old['fit_artifacts'])
            for n in ('model.npz','fit.json','audit_fit.json'):shutil.copyfile(a.resume_fit/n,a.output/n)
            rec['reused_fit_result_sha256']=sha256(a.resume_fit/'result.json')
        else:
            fit(a,c,proof);print('one unique219 fit completed',flush=True)
        if a.fit_only:
            with (a.output/'audit_fit.log').open('x') as log:
                subprocess.run([sys.executable,str(ROOT/'scripts/audit_quality_final_selector.py'),'--output',str(a.output),'--artifact-root',str(a.artifact_root),'--deadline',str(a.deadline),'--fit-only'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,a.deadline-time.time()))
            freeze(c,a.artifact_root);verify_hashes(ROOT,{CONFIG:rec['config_sha256'],**rec['code_sha256'],**c['input_hashes']});guard(a,c)
            rec.update(status='completed_fit_only_pending_final_B2',new_cpu_fits=1,new_74_predictions=0,new_74_scores=0,
                       seconds=time.time()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                       frozen_rgb_verified=293,frozen_artifacts_verified=18,
                       fit_artifacts={n:sha256(a.output/n) for n in ('model.npz','fit.json','audit_fit.json')})
            write_new_json(a.output/'result.json',rec);return
        with np.load(a.output/'model.npz',allow_pickle=False) as z:model={k:z[k] for k in z.files}
        (a.output/'predictions').mkdir();ss=final_sources(c)
        for name in c['prediction_names']:
            guard(a,c);pred=predict(load(ss,name),model);Image.fromarray(pred).save(a.output/'predictions'/name)
        files={str(f.relative_to(a.output)):sha256(f) for f in a.output.rglob('*') if f.is_file() and (f.parent.name=='predictions' or f.name in ('model.npz','fit.json'))}
        write_new_json(a.output/'predictions_sealed.json',dict(files=files,views=74,sealed_unix=time.time(),config_sha256=rec['config_sha256']))
        with (a.output/'audit.log').open('x') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts/audit_quality_final_selector.py'),'--output',str(a.output),'--artifact-root',str(a.artifact_root),'--deadline',str(a.deadline)]+(['--prediction-only'] if a.resume_fit else []),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,a.deadline-time.time()))
        result=score(a,c);freeze(c,a.artifact_root);verify_hashes(ROOT,{CONFIG:rec['config_sha256'],**rec['code_sha256'],**c['input_hashes']});guard(a,c)
        verify_hashes(a.output,files)
        rec.update(status='completed_fixed_final_transfer_development',**result,seconds=time.time()-started,
                   peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,new_cpu_fits=0 if a.resume_fit else 1,
                   frozen_rgb_verified=293,frozen_artifacts_verified=18,predictions_seal_sha256=sha256(a.output/'predictions_sealed.json'),
                   audit_sha256=sha256(a.output/'audit.json'),per_view_sha256=sha256(a.output/'per_view.json'))
        write_new_json(a.output/'result.json',rec);print('SEM363 delta raw pp',rec['pooled']['delta_miou_pp'],flush=True)
    except BaseException as exc:
        write_new_json(a.output/'failure.json',dict(type=type(exc).__name__,message=str(exc),seconds=time.time()-started));raise


if __name__=='__main__':main()
