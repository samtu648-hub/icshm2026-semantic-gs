#!/usr/bin/env python3
"""SEM366 independent global block moments and native74 replay; no solves."""
import argparse
import hashlib
from pathlib import Path
import resource
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_quality_conditional_final_selector import CONFIG,read,load,sources,guard,base,matched
from scripts.audit_quality_selector_fit_expansion import reference_rows,local_reference,proposals
from scripts.audit_quality_class_conditional_selector import reference_predict
from scripts.diagnose_quality_selector_fit_gap import cube
from src.quality_static_training import TrainingLabels
from src.quality_protocol import sha256,verify_hashes,write_new_json
import numpy as np
from PIL import Image


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();c=read(ROOT/CONFIG);start=time.time();matched(c)
    rec=read(a.output/'started.json');verify_hashes(ROOT,{CONFIG:rec['config_sha256'],**rec['code_sha256']})
    seal=read(a.output/'predictions_sealed.json');verify_hashes(a.output,seal['files'])
    proof=read(a.output/'provenance.json');fit=read(a.output/'fit.json');parent_fit=read(ROOT/c['parent_fit']/'fit.json')
    with np.load(a.output/'model.npz',allow_pickle=False) as z:model={k:z[k] for k in z.files}
    with np.load(ROOT/c['parent_fit']/'model.npz',allow_pickle=False) as z:parent={k:z[k] for k in z.files}
    if (fit['training_views']!=c['train_names'] or fit['fit_labels_accessed']!=c['train_names'] or fit['sampling']!=parent_fit['sampling']
        or not np.array_equal(model['mean'],parent['mean']) or not np.array_equal(model['scale'],parent['scale'])
        or fit['model_sha256']!=sha256(a.output/'model.npz')):raise ValueError('Final fit/normalization binding differs')
    lm=read(ROOT/'experiments/runs/SEM-001/label_manifest.json')
    labels=TrainingLabels(a.artifact_root/'r2/sem_001/labels',c['train_names'],lm,[1320,989])
    counts=np.zeros(5,np.int64);mass=np.zeros(5);sx=np.zeros((5,43));sxx=np.zeros((5,43,43));sy=np.zeros(5);sxy=np.zeros((5,43));block_rows=np.zeros(5,np.int64)
    for name in c['train_names']:
        guard(a,c);gt=labels.read(name)
        for k in range(5):counts[k]+=np.count_nonzero(gt==k)
        if bool(model['fallback']):continue
        fold=proof['fold_by_name'][name];data=load(sources(c,fold),name);eligible=proposals(data);pool=np.where(eligible.any(0))[0]
        token=hashlib.sha256(f"{c['seed']}/{fold}/{name}".encode()).digest();rng=np.random.Generator(np.random.PCG64(int.from_bytes(token[:8],'little')))
        if len(pool)>8192:pool=np.sort(rng.choice(pool,8192,replace=False))
        cls,pos=np.where(eligible[:,pool]);ix=pool[pos]
        old=next(j for j in fit['sampling'] if j['name']==name)
        if hashlib.sha256(np.stack((ix,cls),axis=1).astype('<i8').tobytes()).hexdigest()!=old['sample_sha256'] or len(ix)!=old['rows']:raise ValueError('Independent samples differ')
        data=(*data[:4],local_reference(data[0]));x=reference_rows(data,ix,cls)
        y=np.zeros(len(ix));y[gt.ravel()[ix]==cls]+=1;y[gt.ravel()[ix]==data[0].ravel()[ix]]-=1
        for cid in range(5):
            selected=cls==cid;q=x[selected];v=np.full(len(q),1/len(ix));target=y[selected]
            block_rows[cid]+=len(q);mass[cid]+=v.sum();sx[cid]+=np.sum(v[:,None]*q,axis=0)
            sxx[cid]+=q.T@(v[:,None]*q);sy[cid]+=v@target;sxy[cid]+=q.T@(v*target)
        del data,gt
    if counts.tolist()!=fit['fit_gt_counts'] or counts.tolist()!=parent_fit['fit_gt_counts']:raise ValueError('Independent fit counts differ')
    residuals=[];total=mass.sum()
    if not bool(model['fallback']):
        if not np.isclose(total,fit['global_weight_total'],atol=1e-10) or block_rows.tolist()!=fit['block_row_counts']:raise ValueError('Global mass or block counts differ')
        mean=sx.sum(0)/total;mean[-1]=0
        variance=np.diag(sxx.sum(0))/total-2*mean*(sx.sum(0)/total)+mean**2
        scale=np.sqrt(np.maximum(variance,0));scale[scale<1e-12]=1;scale[-1]=1
        if not np.allclose(mean,model['mean'],atol=1e-10,rtol=1e-10) or not np.allclose(scale,model['scale'],atol=1e-7,rtol=1e-7):raise ValueError('Independent parent normalization differs')
        penalty=np.eye(43)*c['learner']['ridge'];penalty[-1,-1]=0;m=model['mean'];s=model['scale']
        for cid in range(5):
            if bool(model['active'][cid])!=(block_rows[cid]>0):raise ValueError('Inactive block mismatch')
            if not model['active'][cid]:
                if np.any(model['coef'][cid]!=0):raise ValueError('Missing block must abstain')
                residuals.append(None);continue
            gram=(sxx[cid]-np.outer(m,sx[cid])-np.outer(sx[cid],m)+mass[cid]*np.outer(m,m))/total/np.outer(s,s)
            rhs=(sxy[cid]-m*sy[cid])/total/s
            if not np.allclose(gram,model['gram'][cid],atol=1e-7,rtol=1e-7) or not np.allclose(rhs,model['rhs'][cid],atol=1e-10,rtol=1e-9):raise ValueError('Independent globally normalized block moments differ')
            residual=float(np.max(abs((gram+penalty)@model['coef'][cid]-rhs)))
            if residual>1e-7:raise ValueError('Conditional normal equation residual')
            residuals.append(residual)
    elif not np.any(counts==0):raise ValueError('Unsupported fallback')
    if int(block_rows.sum())!=1822350 or sorted(labels.accessed)!=c['train_names']:raise ValueError('Unique219 whitelist/sample count differs')
    del labels
    fit_audit=dict(status='passed_independent_final_conditional_fit',views=219,rows=int(block_rows.sum()),
        block_rows=block_rows.tolist(),global_weight_total=float(total),block_weight_mass=(mass/total).tolist(),ridge_equation_max_abs=residuals,
        standardization_exact=True,model_sha256=sha256(a.output/'model.npz'),fit_sha256=sha256(a.output/'fit.json'))
    write_new_json(a.output/'audit_fit.json',fit_audit);print('Independent unique219 block audit passed',flush=True)
    ss=base.final_sources(c)
    for name in c['prediction_names']:
        guard(a,c);data=load(ss,name);actual=reference_predict(data,model);stored=np.asarray(Image.open(a.output/'predictions'/name))
        if not np.array_equal(actual,stored):raise ValueError('Independent native74 replay differs: '+name)
        del data
    print('All74 native predictions independently replayed',flush=True)
    labels=TrainingLabels(a.artifact_root/'r2/sem_001/labels',c['prediction_names'],lm,[1320,989]);rows=[]
    from scripts.diagnose_quality_complementarity import checked_image
    folder,result=ss['five']
    for name in c['prediction_names']:
        guard(a,c);gt=labels.read(name);raw=checked_image(folder,result,'predictions/'+name);stored=np.asarray(Image.open(a.output/'predictions'/name))
        rows.append(dict(name=name,cube=cube(gt,raw,stored,reference=True).tolist()))
    verify_hashes(a.output,seal['files']);verify_hashes(ROOT,{CONFIG:rec['config_sha256'],**rec['code_sha256']})
    write_new_json(a.output/'audit.json',dict(status='passed_final_conditional_audit',fit_audit=fit_audit,rows=rows,native_prediction_replays=74,
        seconds=time.time()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,seal_sha256=sha256(a.output/'predictions_sealed.json'),new_cpu_fits=0))


if __name__=='__main__':main()
