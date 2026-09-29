#!/usr/bin/env python3
"""SEM365 one fixed nine-group CPU experiment; no final/74 evaluation."""
import argparse
import hashlib
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_quality_selector_fit_expansion as base
from scripts.diagnose_quality_oof_selector_bottleneck import provenance as source_provenance
from scripts.run_quality_final_selector import freeze
from src.quality_class_conditional_selector import solve_blocks,predict
from src.quality_learned_selector import sample_rows,feature_rows
from src.quality_static_training import TrainingLabels
from src.quality_protocol import sha256,verify_hashes,write_new_json
import numpy as np
from PIL import Image
read,load,sources=base.read,base.load,base.sources
CONFIG='experiments/runs/SEM-365-CLASS-CONDITIONAL-SELECTOR/config.yaml'
CODE=['scripts/run_quality_class_conditional_selector.py','scripts/audit_quality_class_conditional_selector.py',
      'src/quality_class_conditional_selector.py','src/quality_learned_selector.py',
      'scripts/run_quality_selector_fit_expansion.py','scripts/diagnose_quality_oof_selector_bottleneck.py',
      'scripts/audit_quality_selector_fit_expansion.py','scripts/run_quality_final_selector.py',
      'scripts/run_quality_selector_weight_ablation.py','src/quality_static_training.py',
      'src/quality_protocol.py','src/quality_semantic_audit.py','tests/test_quality_class_conditional_selector.py']


def guard(batch,c,deadline):
    used=sum(f.stat().st_size for folder in batch.parent.glob('SEM-365-*') for f in folder.rglob('*') if f.is_file())
    remaining=c['resources']['maximum_output_mib']*2**20-used
    if time.time()>deadline or remaining<0 or shutil.disk_usage(ROOT).free<remaining:
        raise RuntimeError('Registered time/output/actual-space budget reached')
    return used


def matched(c):
    verify_hashes(ROOT,c['input_hashes']);pc=read(ROOT/c['parent_config'])
    for k in ('git_commit','seed','dataset_manifest','sources','roles','views','learner','camera_metadata','guard_radius','comparison'):
        if c[k]!=pc[k]:raise ValueError('Unregistered scientific change: '+k)
    parent=ROOT/c['paired_parent_batch'];r=read(parent/'result.json')
    if r['config_sha256']!=sha256(ROOT/c['parent_config']):raise ValueError('Parent configuration mismatch')
    verify_hashes(ROOT,r['code_sha256']);verify_hashes(parent,read(parent/'predictions_sealed.json')['files'])


def worker(a,c,role):
    folder=a.output/role['id'];folder.mkdir(exist_ok=False);start=time.time()
    rec=read(a.output/'started.json');verify_hashes(ROOT,{CONFIG:rec['config_sha256'],**rec['code_sha256'],**c['input_hashes']})
    parent=ROOT/c['paired_parent_batch']/role['id'];oldfit=read(parent/'fit.json')
    with np.load(parent/'model.npz',allow_pickle=False) as z:old={k:z[k] for k in z.files}
    if oldfit['role']!=role:raise ValueError('Parent role differs')
    labels=TrainingLabels(a.artifact_root/'r2/sem_001/labels',role['fit'],read(ROOT/'experiments/runs/SEM-001/label_manifest.json'),[1320,989])
    counts=np.zeros(5,np.int64)
    for name in role['fit']:
        guard(a.output,c,a.deadline);counts+=np.bincount(labels.read(name).ravel(),minlength=5)
    if counts.tolist()!=oldfit['fit_gt_counts']:raise ValueError('Parent fit label counts differ')
    fallback=bool(np.any(counts==0))
    if fallback!=bool(old['fallback']):raise ValueError('Missing-class fallback changed')
    gram=np.zeros((5,43,43));rhs=np.zeros((5,43));row_counts=np.zeros(5,np.int64);weight_total=0.
    sx=np.zeros(43);sxx=np.zeros(43);journal=[];ss=sources(c,role['fold'])
    if not fallback:
        for name in role['fit']:
            guard(a.output,c,a.deadline);data=load(ss,name)
            ix,cls=sample_rows(data,c['seed'],role['fold'],name,c['learner']['max_pixels_per_view'])
            journal.append(dict(name=name,rows=len(ix),pixels=len(np.unique(ix)),sample_sha256=hashlib.sha256(np.column_stack([ix,cls]).astype('<i8').tobytes()).hexdigest()))
            if len(ix):
                gt=labels.read(name).ravel()[ix];x=feature_rows(data,ix,cls)
                y=(gt==cls).astype(float)-(gt==data[0].ravel()[ix]).astype(float)
                z=(x-old['mean'])/old['scale'];w=np.full(len(ix),1/len(ix));weight_total+=w.sum()
                sx+=w@x;sxx+=w@(x*x)
                for cid in range(5):
                    keep=cls==cid;row_counts[cid]+=np.count_nonzero(keep)
                    q=z[keep];v=w[keep];gram[cid]+=q.T@(v[:,None]*q);rhs[cid]+=q.T@(v*y[keep])
            del data
        if journal!=oldfit['sampling']:raise ValueError('Parent samples changed')
        mean=sx/weight_total;mean[-1]=0
        variance=sxx/weight_total-2*mean*(sx/weight_total)+mean**2
        scale=np.sqrt(np.maximum(variance,0));scale[scale<1e-12]=1;scale[-1]=1
        if not np.allclose(mean,old['mean'],atol=1e-10,rtol=1e-10) or not np.allclose(scale,old['scale'],atol=1e-7,rtol=1e-7):raise ValueError('Parent standardization mismatch')
        if not np.allclose(gram.sum(0)/weight_total,old['gram'],atol=1e-8,rtol=1e-8) or not np.allclose(rhs.sum(0)/weight_total,old['rhs'],atol=1e-10,rtol=1e-9):raise ValueError('Block moments do not conserve parent global objective')
        solved=solve_blocks(gram,rhs,weight_total,row_counts,c['learner']['ridge'])
    else:solved=dict(coef=np.zeros((5,43)),gram=gram,rhs=rhs,active=np.zeros(5,bool))
    model=dict(solved,mean=old['mean'],scale=old['scale'],fallback=np.array(fallback),row_counts=row_counts)
    np.savez(folder/'model.npz',**model)
    write_new_json(folder/'fit.json',dict(role=role,fit_labels_accessed=sorted(labels.accessed),fit_gt_counts=counts.tolist(),
        sampling=journal,model_sha256=sha256(folder/'model.npz'),parent_model_sha256=sha256(parent/'model.npz'),
        fallback=fallback,global_weight_total=weight_total,block_row_counts=row_counts.tolist(),
        fit_seconds=time.time()-start,solved_blocks=int(solved['active'].sum()),standardization_reused_exact=True))
    del labels
    (folder/'predictions').mkdir();preds={}
    for name in role['evaluation']:
        guard(a.output,c,a.deadline);data=load(ss,name);out=predict(data,model)
        Image.fromarray(out).save(folder/'predictions'/name);preds[name]=sha256(folder/'predictions'/name);del data
    write_new_json(folder/'result.json',dict(role=role,model_sha256=sha256(folder/'model.npz'),fit_sha256=sha256(folder/'fit.json'),
        predictions=preds,seconds=time.time()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,eval_labels_read_in_worker=0))
    print(role['id'],'completed',round(time.time()-start,3),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker');p.add_argument('--deadline',type=float);a=p.parse_args();a.output=a.output.resolve();c=read(ROOT/CONFIG)
    if not a.output.is_relative_to(ROOT/'outputs/quality/learned_selector'):p.error('Isolated output required')
    if a.worker:worker(a,c,next(r for r in c['roles'] if r['id']==a.worker));return
    a.output.mkdir(parents=True,exist_ok=False);start=time.time();a.deadline=start+c['resources']['maximum_seconds']
    receipt=ROOT/'outputs/quality/learned_selector/SEM-365-preparation-r1/authorization.json';approval=read(receipt)
    if approval['config_sha256']!=sha256(ROOT/CONFIG) or not approval['user_confirmation']:raise ValueError('Authorization binding mismatch')
    rec={k:c[k] for k in ('experiment_id','git_commit','seed','dataset_manifest','split_id')}
    rec.update(status='started',config_sha256=sha256(ROOT/CONFIG),code_sha256={n:sha256(ROOT/n) for n in CODE},
        authorization_sha256=sha256(receipt),started_unix=start,deadline=a.deadline,gpu_training=False,base_training=False,new_check_or74_scores=0)
    write_new_json(a.output/'started.json',rec)
    for n in [CONFIG,*CODE]:
        dest=a.output/'code'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dest)
    try:
        matched(c);dc=read(ROOT/c['diagnostic_config']);folds,bindings,frozen=source_provenance(dc,a.artifact_root)
        write_new_json(a.output/'provenance_before_fit.json',dict(source_hashes=bindings,fold_by_name=folds,frozen=frozen))
        for role in c['roles']:
            guard(a.output,c,a.deadline)
            with (a.output/(role['id']+'.log')).open('x') as log:
                subprocess.run([sys.executable,__file__,'--artifact-root',str(a.artifact_root),'--output',str(a.output),
                    '--worker',role['id'],'--deadline',str(a.deadline)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,a.deadline-time.time()))
            print((a.output/(role['id']+'.log')).read_text(),end='',flush=True)
        files={str(f.relative_to(a.output)):sha256(f) for f in a.output.rglob('*') if f.is_file() and
               ('predictions' in f.parts or f.name in ('model.npz','fit.json','result.json'))}
        write_new_json(a.output/'predictions_sealed.json',dict(files=files,config_sha256=rec['config_sha256'],sealed_unix=time.time(),heads=9,views=108,no_inner_selection=True))
        with (a.output/'audit.log').open('x') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts/audit_quality_class_conditional_selector.py'),'--output',str(a.output),
                '--artifact-root',str(a.artifact_root),'--deadline',str(a.deadline)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,a.deadline-time.time()))
        result=base.score(a,c);parent=read(ROOT/c['paired_parent_batch']/'result.json')
        for scope in ['pooled','folds','heads']:
            keys=[None] if scope=='pooled' else list(result[scope])
            for k in keys:
                now=result[scope] if k is None else result[scope][k];old=parent[scope] if k is None else parent[scope][k]
                for arm in ('raw','foundation005','historical_fused'):
                    if now['metrics'][arm]!=old['metrics'][arm]:raise ValueError('Matched comparator changed')
                now['delta_vs_sem361_pp']=100*(now['metrics']['selector']['mIoU_with_background']-old['metrics']['selector']['mIoU_with_background'])
        freeze(read(ROOT/c['freeze_config']),a.artifact_root);matched(c)
        verify_hashes(ROOT,{CONFIG:rec['config_sha256'],**rec['code_sha256']});verify_hashes(a.output,files)
        rec.update(status='completed_fixed_development_evaluation',**result,seconds=time.time()-start,
            output_bytes=guard(a.output,c,a.deadline),frozen_rgb293_and18_unchanged=True,new_head_groups=9,
            solved_blocks=sum(read(a.output/r['id']/'fit.json')['solved_blocks'] for r in c['roles']),
            independent_audit_sha256=sha256(a.output/'audit.json'),prediction_seal_sha256=sha256(a.output/'predictions_sealed.json'))
        write_new_json(a.output/'result.json',rec);print('SEM365 complete delta SEM361',rec['pooled']['delta_vs_sem361_pp'],flush=True)
    except BaseException as exc:
        write_new_json(a.output/'failure.json',dict(type=type(exc).__name__,message=str(exc),elapsed_seconds=time.time()-start));raise


if __name__=='__main__':main()
