#!/usr/bin/env python3
"""SEM366 single final block fit and sealed historical74; advisory resources."""
import argparse
import hashlib
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_quality_final_selector as base
from src.quality_class_conditional_selector import solve_blocks,predict
from src.quality_learned_selector import sample_rows,feature_rows
from src.quality_static_training import TrainingLabels
from src.quality_protocol import sha256,verify_hashes,write_new_json
import numpy as np
from PIL import Image
read,load,sources=base.read,base.load,base.sources
CONFIG='experiments/runs/SEM-366-CLASS-CONDITIONAL-FINAL/config.yaml'
RECEIPT='outputs/quality/final_transfer/SEM-366-preparation-r1/authorization.json'
CODE=list(dict.fromkeys(base.CODE+['scripts/run_quality_conditional_final_selector.py','scripts/audit_quality_conditional_final_selector.py',
    'src/quality_class_conditional_selector.py','scripts/audit_quality_class_conditional_selector.py',
    'scripts/run_quality_class_conditional_selector.py','scripts/audit_quality_selector_fit_expansion.py',
    'scripts/run_quality_selector_fit_expansion.py','scripts/diagnose_quality_oof_selector_bottleneck.py']))


def guard(a,c):
    # Estimate overruns are telemetry, not an abort condition. No disk reserve.
    if shutil.disk_usage(a.output).free<=0:raise OSError('Actual disk space exhausted')


def matched(c):
    verify_hashes(ROOT,c['input_hashes']);old=read(ROOT/c['matched_final_config'])
    for key in ('learner','sources','final_sources','train_names','prediction_names','seed','dataset_manifest','split_id','camera_metadata','b2_run','b2_audit','baseline_comparison','baseline_predictions'):
        if c[key]!=old[key]:raise ValueError('Unregistered final change: '+key)
    for batch in (ROOT/c['matched_final_batch'],ROOT/'outputs/quality/learned_selector/SEM-365-r1'):
        result=read(batch/'result.json');verify_hashes(ROOT,result['code_sha256']);verify_hashes(batch,read(batch/'predictions_sealed.json')['files'])
    parent=ROOT/c['parent_fit'];old=read(parent/'result.json');verify_hashes(parent,old['fit_artifacts'])


def fit(a,c,proof):
    start=time.time();parent=ROOT/c['parent_fit'];oldfit=read(parent/'fit.json')
    with np.load(parent/'model.npz',allow_pickle=False) as z:old={k:z[k] for k in z.files}
    labels=TrainingLabels(a.artifact_root/'r2/sem_001/labels',c['train_names'],read(ROOT/'experiments/runs/SEM-001/label_manifest.json'),[1320,989])
    counts=np.zeros(5,np.int64);gram=np.zeros((5,43,43));rhs=np.zeros((5,43));row_counts=np.zeros(5,np.int64)
    total=0.;sx=np.zeros(43);sxx=np.zeros(43);journal=[]
    for i,name in enumerate(c['train_names']):
        guard(a,c);fold=proof['fold_by_name'][name];data=load(sources(c,fold),name)
        ix,cls=sample_rows(data,c['seed'],fold,name,c['learner']['max_pixels_per_view'])
        row=dict(name=name,rows=len(ix),pixels=len(np.unique(ix)),sample_sha256=hashlib.sha256(np.column_stack([ix,cls]).astype('<i8').tobytes()).hexdigest())
        if row!=proof['original_samples'][name]:raise ValueError('Original sample drift')
        journal.append(row);gt=labels.read(name);counts+=np.bincount(gt.ravel(),minlength=5)
        if len(ix):
            x=feature_rows(data,ix,cls);truth=gt.ravel()[ix];y=(truth==cls).astype(float)-(truth==data[0].ravel()[ix]).astype(float)
            z=(x-old['mean'])/old['scale'];w=np.full(len(ix),1/len(ix));total+=w.sum();sx+=w@x;sxx+=w@(x*x)
            for cid in range(5):
                keep=cls==cid;q=z[keep];v=w[keep];row_counts[cid]+=len(q);gram[cid]+=q.T@(v[:,None]*q);rhs[cid]+=q.T@(v*y[keep])
        del data,gt
        if (i+1)%50==0:print('fit streaming',i+1,flush=True)
    if (journal!=oldfit['sampling'] or counts.tolist()!=oldfit['fit_gt_counts'] or sorted(labels.accessed)!=c['train_names']
            or sum(r['rows'] for r in journal)!=1822350):raise ValueError('Parent219 fit inventory differs')
    mean=sx/total;mean[-1]=0;variance=sxx/total-2*mean*(sx/total)+mean**2
    scale=np.sqrt(np.maximum(variance,0));scale[scale<1e-12]=1;scale[-1]=1
    if not np.allclose(mean,old['mean'],atol=1e-10,rtol=1e-10) or not np.allclose(scale,old['scale'],atol=1e-7,rtol=1e-7):raise ValueError('Parent standardization differs')
    if not np.allclose(gram.sum(0)/total,old['gram'],atol=1e-7,rtol=1e-7) or not np.allclose(rhs.sum(0)/total,old['rhs'],atol=1e-10,rtol=1e-9):raise ValueError('Global objective differs')
    fallback=bool(np.any(counts==0));solved=solve_blocks(gram,rhs,total,row_counts,c['learner']['ridge']) if not fallback else dict(coef=np.zeros((5,43)),gram=gram/total,rhs=rhs/total,active=np.zeros(5,bool))
    model=dict(solved,mean=old['mean'],scale=old['scale'],fallback=np.array(fallback),row_counts=row_counts)
    np.savez(a.output/'model.npz',**model)
    write_new_json(a.output/'fit.json',dict(training_views=c['train_names'],fit_labels_accessed=sorted(labels.accessed),fit_gt_counts=counts.tolist(),
        sampling=journal,sample_rows=sum(r['rows'] for r in journal),model_sha256=sha256(a.output/'model.npz'),parent_model_sha256=sha256(parent/'model.npz'),
        fit_seconds=time.time()-start,global_weight_total=total,block_row_counts=row_counts.tolist(),solved_blocks=int(solved['active'].sum()),standardization_reused_exact=True,novel_fits=1))
    print('single final head fitted',round(time.time()-start,3),flush=True)
    # No training-label object escapes this function.
    return model


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output=a.output.resolve();a.fit_only=False;c=read(ROOT/CONFIG)
    if not a.output.is_relative_to(ROOT/'outputs/quality/final_transfer'):p.error('Isolated output required')
    a.output.mkdir(parents=True,exist_ok=False);start=time.time();approval=read(ROOT/RECEIPT)
    if not approval['user_confirmation'] or approval['config_sha256']!=sha256(ROOT/CONFIG):raise ValueError('Approval binding mismatch')
    rec={k:c[k] for k in ('experiment_id','git_commit','seed','dataset_manifest','split_id')}
    rec.update(status='started',started_unix=start,config_sha256=sha256(ROOT/CONFIG),code_sha256={n:sha256(ROOT/n) for n in CODE},authorization_sha256=sha256(ROOT/RECEIPT),gpu_work=False,automatic_promotion=False)
    write_new_json(a.output/'started.json',rec)
    for n in [CONFIG,RECEIPT,*CODE]:
        dest=a.output/'code'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dest)
    try:
        matched(c);proof=base.provenance(a,c);write_new_json(a.output/'provenance.json',proof)
        model=fit(a,c,proof);ss=base.final_sources(c);(a.output/'predictions').mkdir()
        for i,name in enumerate(c['prediction_names']):
            guard(a,c);out=predict(load(ss,name),model);Image.fromarray(out).save(a.output/'predictions'/name)
            if (i+1)%20==0:print('prediction progress',i+1,flush=True)
        files={str(f.relative_to(a.output)):sha256(f) for f in a.output.rglob('*') if f.is_file() and (f.parent.name=='predictions' or f.name in ('model.npz','fit.json'))}
        write_new_json(a.output/'predictions_sealed.json',dict(files=files,views=74,sealed_unix=time.time(),config_sha256=rec['config_sha256']))
        with (a.output/'audit.log').open('x') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts/audit_quality_conditional_final_selector.py'),'--artifact-root',str(a.artifact_root),'--output',str(a.output)],stdout=log,stderr=subprocess.STDOUT,check=True)
        # Reuse exact fixed scoring with only its execution-resource policy replaced locally.
        base.guard=guard;result=base.score(a,c);parent=read(ROOT/c['matched_final_batch']/'result.json')
        for scope in ('pooled','groups'):
            for k in ([None] if scope=='pooled' else result[scope]):
                now=result[scope] if k is None else result[scope][k];old=parent[scope] if k is None else parent[scope][k]
                if now['comparators']!=old['comparators']:raise ValueError('Final comparator drift')
                now['sem363']=old['selector'];now['delta_vs_sem363_pp']=100*(now['selector']['mIoU_with_background']-old['selector']['mIoU_with_background'])
        base.freeze(c,a.artifact_root);matched(c);verify_hashes(ROOT,{CONFIG:rec['config_sha256'],**rec['code_sha256']});verify_hashes(a.output,files)
        rec.update(status='completed_fixed_final_transfer_development',**result,seconds=time.time()-start,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,new_cpu_fits=1,solved_blocks=read(a.output/'fit.json')['solved_blocks'],
            new_74_predictions=74,new_74_scores=1,frozen_rgb_verified=293,frozen_artifacts_verified=18,
            audit_sha256=sha256(a.output/'audit.json'),prediction_seal_sha256=sha256(a.output/'predictions_sealed.json'),
            output_bytes=sum(f.stat().st_size for f in a.output.rglob('*') if f.is_file()),free_disk_bytes=shutil.disk_usage(a.output).free)
        write_new_json(a.output/'result.json',rec);print('SEM366 complete delta SEM363',rec['pooled']['delta_vs_sem363_pp'],flush=True)
    except BaseException as exc:
        write_new_json(a.output/'failure.json',dict(type=type(exc).__name__,message=str(exc),seconds=time.time()-start));raise


if __name__=='__main__':main()
