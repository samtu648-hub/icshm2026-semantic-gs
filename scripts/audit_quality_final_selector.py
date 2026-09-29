#!/usr/bin/env python3
"""Independent final-head fit statistics and optional native74 replay; no fitting."""
import argparse
import hashlib
from pathlib import Path
import resource
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_quality_final_selector import read,load,sources,final_sources,guard,CONFIG
import numpy as np
from PIL import Image
from scripts.audit_quality_selector_weight_ablation import reference_rows,proposals,local_reference
from scripts.diagnose_quality_selector_fit_gap import reference_prediction,cube
from src.quality_static_training import TrainingLabels
from src.quality_protocol import sha256,verify_hashes,write_new_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True);p.add_argument('--fit-only',action='store_true');p.add_argument('--prediction-only',action='store_true')
    a=p.parse_args();c=read(ROOT/CONFIG);started=time.time();proof=read(a.output/'provenance.json');fit=read(a.output/'fit.json')
    if a.fit_only and a.prediction_only:p.error('Conflicting audit phases')
    if fit['model_sha256']!=sha256(a.output/'model.npz') or fit['fit_labels_accessed']!=c['train_names'] or fit['training_views']!=c['train_names']:
        raise ValueError('Final fit binding or label whitelist differs')
    with np.load(a.output/'model.npz',allow_pickle=False) as z:model={k:z[k] for k in z.files}
    lm=read(ROOT/'experiments/runs/SEM-001/label_manifest.json')
    if a.prediction_only:
        previous=read(a.output/'audit_fit.json')
        if previous['status']!='passed_independent_final_fit' or previous['model_sha256']!=sha256(a.output/'model.npz') or previous['fit_sha256']!=sha256(a.output/'fit.json'):
            raise ValueError('Missing accepted independent saved fit')
        fit_audit=previous
    else:
        labels=TrainingLabels(a.artifact_root/'r2/sem_001/labels',c['train_names'],lm,[1320,989])
        counts=np.zeros(5,np.int64);total=0.;sumx=np.zeros(43);sumxx=np.zeros((43,43));sumxy=np.zeros(43);sumy=0.;rows_total=0
        for name in c['train_names']:
            guard(a,c);fold=proof['fold_by_name'][name];data=load(sources(c,fold),name);allowed=proposals(data)
            data=(*data[:4],local_reference(data[0]));pool=np.where(allowed.any(0))[0]
            token=hashlib.sha256(f"{c['seed']}/{fold}/{name}".encode()).digest();rng=np.random.Generator(np.random.PCG64(int.from_bytes(token[:8],'little')))
            if len(pool)>c['learner']['max_pixels_per_view']:pool=np.sort(rng.choice(pool,c['learner']['max_pixels_per_view'],replace=False))
            cls,positions=np.where(allowed[:,pool]);ix=pool[positions]
            digest=hashlib.sha256(np.stack((ix,cls),axis=1).astype('<i8').tobytes()).hexdigest()
            old=next(r for r in fit['sampling'] if r['name']==name)
            if digest!=old['sample_sha256'] or len(ix)!=old['rows'] or len(np.unique(ix))!=old['pixels']:raise ValueError('Independent final sampling differs')
            gt=labels.read(name);counts+=np.bincount(gt.ravel(),minlength=5)
            if len(ix):
                x=reference_rows(data,ix,cls);truth=gt.ravel()[ix];y=(truth==cls).astype(float)-(truth==data[0].ravel()[ix]).astype(float);w=np.ones(len(ix))/len(ix)
                total+=w.sum();sumx+=np.sum(w[:,None]*x,axis=0);sumxx+=x.T@(w[:,None]*x);sumxy+=x.T@(w*y);sumy+=np.dot(w,y);rows_total+=len(ix)
            del data,gt
        if counts.tolist()!=fit['fit_gt_counts'] or rows_total!=fit['sample_rows'] or rows_total!=1822350:raise ValueError('Unique219 counts differ')
        mean=sumx/total;mean[-1]=0
        variance=np.diag(sumxx)/total-2*mean*(sumx/total)+mean**2
        scale=np.sqrt(np.maximum(variance,0));scale[scale<1e-12]=1;scale[-1]=1
        if not np.allclose(mean,model['mean'],atol=1e-10,rtol=1e-10) or not np.allclose(scale,model['scale'],atol=1e-7,rtol=1e-7):raise ValueError('Independent final standardization differs')
        m=model['mean'];s=model['scale'];gram=(sumxx/total-np.outer(m,sumx/total)-np.outer(sumx/total,m)+np.outer(m,m))/np.outer(s,s);rhs=(sumxy/total-m*sumy/total)/s
        if not np.allclose(gram,model['gram'],atol=1e-7,rtol=1e-7) or not np.allclose(rhs,model['rhs'],atol=1e-10,rtol=1e-9):raise ValueError('Independent final regression statistics differ')
        penalty=np.eye(43)*c['learner']['ridge'];penalty[-1,-1]=0;error=float(np.max(np.abs((gram+penalty)@model['coef']-rhs)))
        if error>1e-7:raise ValueError('Final normal equation residual')
        fit_audit=dict(status='passed_independent_final_fit',views=219,rows=rows_total,ridge_equation_max_abs=error,model_sha256=sha256(a.output/'model.npz'),fit_sha256=sha256(a.output/'fit.json'),audit_code_sha256=sha256(Path(__file__)))
        if a.fit_only:
            fit_audit.update(seconds=time.time()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            write_new_json(a.output/'audit_fit.json',fit_audit);print('Independent unique219 fit passed',flush=True);return
    seal=read(a.output/'predictions_sealed.json');verify_hashes(a.output,seal['files']);ss=final_sources(c)
    labels=TrainingLabels(a.artifact_root/'r2/sem_001/labels',c['prediction_names'],lm,[1320,989]);rows=[]
    for name in c['prediction_names']:
        guard(a,c);data=load(ss,name);actual=reference_prediction(data,model);stored=np.asarray(Image.open(a.output/'predictions'/name))
        if not np.array_equal(actual,stored):raise ValueError('Independent final74 prediction differs: '+name)
        gt=labels.read(name);v=cube(gt,data[0],stored,reference=True);rows.append(dict(name=name,cube=v.tolist()));del data,gt
    verify_hashes(a.output,seal['files'])
    write_new_json(a.output/'audit.json',dict(status='passed_final_selector_audit',fit_audit=fit_audit,rows=rows,native_prediction_replays=74,
        seconds=time.time()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,audit_code_sha256=sha256(Path(__file__))))


if __name__=='__main__':main()
