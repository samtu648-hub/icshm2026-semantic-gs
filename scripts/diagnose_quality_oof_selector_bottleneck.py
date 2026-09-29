#!/usr/bin/env python3
"""SEM364: saved219 source diagnosis and original108 selector decisions, no fit."""
import argparse
import hashlib
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_quality_selector_weight_ablation import read,load,sources
from scripts.run_quality_final_selector import freeze
from scripts.audit_quality_selector_weight_ablation import reference_rows,local_reference,proposals
from src.quality_learned_selector import candidates,feature_rows,sample_rows
from src.quality_selector_bottleneck import source_counts,decision_counts,BUCKETS
from src.quality_static_training import TrainingLabels,semantic_access
from src.quality_protocol import sha256,verify_hashes,write_new_json
from src.quality_semantic_audit import metrics
from scripts.diagnose_quality_selector_fit_gap import cube
import numpy as np
from PIL import Image

CONFIG='experiments/runs/SEM-364-OOF-SELECTOR-BOTTLENECK/config.yaml'
CODE=['scripts/diagnose_quality_oof_selector_bottleneck.py','src/quality_selector_bottleneck.py',
      'src/quality_learned_selector.py','scripts/audit_quality_selector_weight_ablation.py',
      'scripts/run_quality_selector_weight_ablation.py','scripts/run_quality_final_selector.py',
      'scripts/diagnose_quality_selector_fit_gap.py','src/quality_static_training.py','src/quality_semantic_audit.py','src/quality_protocol.py']


def provenance(c, artifact):
    verify_hashes(ROOT,c['input_hashes'])
    pc=read(ROOT/c['parent_config']);fc=read(ROOT/c['freeze_config'])
    if any(c[k]!=pc[k] for k in ('seed','sources','roles','learner')) or c['views']!=fc['train_names']:
        raise ValueError('Changed scientific roles or source list')
    parent=ROOT/c['parent_batch'];record=read(parent/'result.json')
    verify_hashes(ROOT,record['code_sha256']);verify_hashes(ROOT,pc['input_hashes'])
    verify_hashes(parent,read(parent/'predictions_sealed.json')['files'])
    frozen=freeze(fc,artifact);cfg=read(ROOT/fc['b2_config'])
    rgb=read(ROOT/cfg['study']['rgb_run']/'access.json')
    split=read(ROOT/'experiments/splits/sem_001_scaffold_static_v1.json');oof=read(ROOT/'experiments/splits/sem_005_oof_spatial_v1.json')
    folds={};bindings={}
    for fold in c['sources']:
        expected=semantic_access(split,oof,rgb,fold)
        for name in expected['prediction']:
            if name in folds:raise ValueError('Repeated OOF view')
            folds[name]=fold
        for kind,(folder,r) in sources(c,fold).items():
            if any(r['access'][k]!=expected[k] for k in ('train','prediction','guard','partition')):
                raise ValueError('Source fold access mismatch')
            if set(c['forbidden_target_views'])&set(r['access']['train']):raise ValueError('Source saw final target labels')
            if r['binding']['seed']!=c['seed'] or r['binding'].get('identity',r['binding'].get('rgb_identity'))!=frozen['identity']:
                raise ValueError('Source identity mismatch')
            bindings[fold+'/'+kind]=sha256(folder/'result.json')
    if sorted(folds)!=c['views'] or set(folds)&set(c['forbidden_target_views']):raise ValueError('Unexpected219')
    for role in c['roles']:
        if set(role['fit'])&(set(role['evaluation'])|set(role['guard'])):raise ValueError('Meta role leakage')
        if any(folds[n]!=role['fold'] for n in role['fit']+role['evaluation']+role['guard']):raise ValueError('Cross-base-fold meta input')
        fit=read(parent/role['id']/'fit.json')
        if fit['role']!=role or fit['fit_labels_accessed']!=sorted(role['fit']):raise ValueError('Saved head role drift')
    return folds,bindings,frozen


def state():
    return dict(views=0,rows=0,sumx=np.zeros(43),sumxx=np.zeros(43),counts=np.zeros((5,5,3,2),np.int64),
                mass=np.zeros((5,5,3,2)),sum_squared_error=0.)


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-from',type=Path);a=p.parse_args();started=time.time();c=read(ROOT/CONFIG)
    a.output=a.output.resolve()
    if not a.output.is_relative_to(ROOT/'outputs/quality/selector_bottleneck'):p.error('Isolated diagnostic output required')
    a.output.mkdir(parents=True,exist_ok=False)
    prior=read(a.reference_from/'result.json')['seconds'] if a.reference_from else 0
    rec={k:c[k] for k in ('experiment_id','git_commit','seed','dataset_manifest','split_id')}
    rec.update(status='started',config_sha256=sha256(ROOT/CONFIG),code_sha256={n:sha256(ROOT/n) for n in CODE},
               new_fits=0,new_gpu_inference=0,new_74_reads=0)
    write_new_json(a.output/'started.json',rec)
    for n in [CONFIG,*CODE]:
        dest=a.output/'code'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dest)
    def guard():
        used=sum(f.stat().st_size for f in a.output.parent.rglob('*') if f.is_file())
        if time.time()-started+prior>c['resources']['maximum_seconds'] or used>c['resources']['maximum_output_mib']*2**20:
            raise RuntimeError('Registered aggregate diagnostic budget reached')
        if shutil.disk_usage(ROOT).free<(c['resources']['maximum_output_mib']*2**20-used):raise RuntimeError('Actual disk space insufficient')
    try:
        folds,bindings,frozen=provenance(c,a.artifact_root);parent=ROOT/c['parent_batch']
        previous={r['name']:r for r in read(parent/'per_view.json')}
        final_fit=read(ROOT/c['unique_fit_batch']/'fit.json');original_samples={r['name']:r for r in final_fit['sampling']}
        models={role['id']:{k:v for k,v in np.load(parent/role['id']/'model.npz',allow_pickle=False).items()} for role in c['roles']}
        states={role['id']:{kind:state() for kind in ('fit','evaluation')} for role in c['roles']}
        labels=TrainingLabels(a.artifact_root/'r2/sem_001/labels',c['views'],read(ROOT/'experiments/runs/SEM-001/label_manifest.json'),[1320,989])
        rows=[];all_gt=np.zeros(5,np.int64)
        for i,name in enumerate(c['views']):
            guard();fold=folds[name];data=load(sources(c,fold),name);gt=labels.read(name);raw=data[0]
            all_gt+=np.bincount(gt.ravel(),minlength=5)
            hist,oracle,recover=source_counts(gt,raw,data[1],reference=bool(a.reference_from))
            row=dict(name=name,fold=fold,source_error_mask_counts=hist.tolist(),oracle_confusion=oracle.tolist())
            if name in previous:
                role=next(r for r in c['roles'] if name in r['evaluation'])
                pred=np.asarray(Image.open(parent/role['id']/'predictions'/name))
                v=cube(gt,raw,pred,reference=bool(a.reference_from))
                if v.tolist()!=previous[name]['gt_raw_selector_cube']:raise ValueError('Original108 saved confusion differs')
                row.update(head=role['id'],decision_counts=decision_counts(gt,raw,pred,recover,reference=bool(a.reference_from)).tolist())
            if a.reference_from:
                allowed=proposals(data);pool=np.where(allowed.any(0))[0]
                token=hashlib.sha256(f"{c['seed']}/{fold}/{name}".encode()).digest()
                rng=np.random.Generator(np.random.PCG64(int.from_bytes(token[:8],'little')))
                if len(pool)>8192:pool=np.sort(rng.choice(pool,8192,replace=False))
                cls,pos=np.where(allowed[:,pool]);ix=pool[pos]
                data=(*data[:4],local_reference(data[0]));x=reference_rows(data,ix,cls)
            else:
                ix,cls=sample_rows(data,c['seed'],fold,name,c['learner']['max_pixels_per_view']);x=feature_rows(data,ix,cls)
            digest=hashlib.sha256(np.stack((ix,cls),axis=1).astype('<i8').tobytes()).hexdigest()
            if digest!=original_samples[name]['sample_sha256'] or len(ix)!=original_samples[name]['rows']:raise ValueError('Original219 sampling drift')
            y=(gt.ravel()[ix]==cls).astype(float)-(gt.ravel()[ix]==raw.ravel()[ix]).astype(float)
            raw_class=raw.ravel()[ix];utility=(y+1).astype(int)
            row.update(sample_sha256=digest,sample_rows=len(ix))
            for role in c['roles']:
                kind='fit' if name in role['fit'] else 'evaluation' if name in role['evaluation'] else None
                if kind is None:continue
                st=states[role['id']][kind];m=models[role['id']]
                if bool(m['fallback']):raise ValueError('Unexpected missing-class fallback')
                score=(x@(m['coef']/m['scale'])-np.dot(m['mean'],m['coef']/m['scale'])) if a.reference_from else ((x-m['mean'])/m['scale'])@m['coef']
                key=((raw_class.astype(int)*5+cls)*3+utility)*2+(score>0).astype(int)
                if a.reference_from:
                    counts=np.zeros(150,np.int64);np.add.at(counts,key,1)
                else:counts=np.bincount(key,minlength=150)
                st['views']+=1;st['rows']+=len(ix);st['sumx']+=x.mean(0);st['sumxx']+=(x*x).mean(0)
                st['counts']+=counts.reshape(5,5,3,2);st['mass']+=counts.reshape(5,5,3,2)/len(ix)
                st['sum_squared_error']+=float(np.mean((score-y)**2))
            rows.append(row)
            if (i+1)%40==0:print('diagnosed views',i+1,flush=True)
            del data,gt,x
        if all_gt.tolist()!=final_fit['fit_gt_counts']:raise ValueError('Unique219 GT totals differ')
        heads={}
        for role in c['roles']:
            h=role['id'];m=models[h];out={}
            for kind in ('fit','evaluation'):
                st=states[h][kind];n=st['views'];mean=st['sumx']/n;variance=np.maximum(0,st['sumxx']/n-mean**2)
                if n!=len(role[kind]):raise ValueError('Meta visit count differs')
                if kind=='fit':
                    mean_fit=mean.copy();mean_fit[-1]=0;scale=np.sqrt(variance);scale[scale<1e-12]=1;scale[-1]=1
                    if not np.allclose(mean_fit,m['mean'],atol=1e-9,rtol=1e-9) or not np.allclose(scale,m['scale'],atol=1e-7,rtol=1e-7):raise ValueError('Saved fit feature moments differ')
                out[kind]=dict(views=n,rows=st['rows'],mean=mean.tolist(),variance=variance.tolist(),
                    counts=st['counts'].tolist(),view_equal_mass=(st['mass']/n).tolist(),candidate_mse=st['sum_squared_error']/n)
            shift=(np.array(out['evaluation']['mean'])-np.array(out['fit']['mean']))/m['scale']
            out['standardized_mean_shift']=shift.tolist();heads[h]=out
        result=dict(rows=rows,heads=heads,gt_counts=all_gt.tolist(),decision_bucket_names=BUCKETS,
                    source_order=['static','b2_3000','b2_6000','deck_binary','stay_cable_binary'])
        write_new_json(a.output/'diagnosis.json',result)
        if a.reference_from:
            original=read(a.reference_from/'diagnosis.json')
            if result['rows']!=original['rows'] or result['gt_counts']!=original['gt_counts']:raise ValueError('Independent219 integer replay differs')
            for h in heads:
                for kind in ('fit','evaluation'):
                    for key in heads[h][kind]:
                        l=heads[h][kind][key];r=original['heads'][h][kind][key]
                        if key in ('views','rows','counts'):
                            if l!=r:raise ValueError('Independent head integer statistics differ')
                        elif not np.allclose(l,r,atol=1e-10,rtol=1e-9):raise ValueError('Independent feature/score statistics differ')
                if not np.allclose(heads[h]['standardized_mean_shift'],original['heads'][h]['standardized_mean_shift'],atol=1e-10,rtol=1e-9):raise ValueError('Independent shift differs')
        freeze(read(ROOT/c['freeze_config']),a.artifact_root)
        verify_hashes(ROOT,{CONFIG:rec['config_sha256'],**rec['code_sha256'],**c['input_hashes']})
        verify_hashes(parent,read(parent/'predictions_sealed.json')['files']);guard()
        rec.update(status='passed_independent_oof_diagnosis' if a.reference_from else 'completed_pending_independent_oof_diagnosis',
                   seconds=time.time()-started,prior_seconds=prior,views=219,saved_selector_views=108,fit_visits=426,
                   candidate_rows=sum(r['sample_rows'] for r in rows),source_result_hashes=bindings,
                   frozen_rgb=293,frozen_artifacts=18,diagnosis_sha256=sha256(a.output/'diagnosis.json'),
                   peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        write_new_json(a.output/'result.json',rec);print(rec['status'],rec['seconds'],flush=True)
    except BaseException as exc:
        write_new_json(a.output/'failure.json',dict(type=type(exc).__name__,message=str(exc),seconds=time.time()-started));raise


if __name__=='__main__':main()
