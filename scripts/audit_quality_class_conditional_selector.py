#!/usr/bin/env python3
"""SEM365 independent uncentered block moments and native108 replay; no fits."""
import argparse
import hashlib
from pathlib import Path
import resource
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_quality_class_conditional_selector import CONFIG,read,load,sources,guard
from scripts.audit_quality_selector_fit_expansion import reference_rows,local_reference,proposals
from src.quality_protocol import sha256,verify_hashes,write_new_json
from src.quality_static_training import TrainingLabels
import numpy as np
from PIL import Image


def reference_predict(data,model):
    raw=data[0];allowed=proposals(data);data=(*data[:4],local_reference(raw));answer=raw.ravel().copy()
    if bool(model['fallback']):return answer.reshape(raw.shape)
    scores=np.full(allowed.shape,-np.inf);beta=model['coef']/model['scale'][None,:];offset=-(beta@model['mean'])
    for cid in range(5):
        pixels=np.where(allowed[cid])[0]
        for start in range(0,len(pixels),32768):
            ix=pixels[start:start+32768]
            scores[cid,ix]=reference_rows(data,ix,np.full(len(ix),cid))@beta[cid]+offset[cid] if model['active'][cid] else 0
    highest=scores.max(0);win=scores.argmax(0);keep=(highest>0)&((scores==highest).sum(0)==1)
    answer[keep]=win[keep];return answer.reshape(raw.shape)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args();start=time.time();c=read(ROOT/CONFIG);seal=read(a.output/'predictions_sealed.json');verify_hashes(a.output,seal['files'])
    lm=read(ROOT/'experiments/runs/SEM-001/label_manifest.json');audits=[]
    for role in c['roles']:
        folder=a.output/role['id'];fit=read(folder/'fit.json');ss=sources(c,role['fold'])
        with np.load(folder/'model.npz',allow_pickle=False) as z:model={k:z[k] for k in z.files}
        with np.load(ROOT/c['paired_parent_batch']/role['id']/'model.npz',allow_pickle=False) as z:parent={k:z[k] for k in z.files}
        parent_fit=read(ROOT/c['paired_parent_batch']/role['id']/'fit.json')
        if (fit['role']!=role or fit['fit_labels_accessed']!=sorted(role['fit']) or fit['sampling']!=parent_fit['sampling']
                or not np.array_equal(model['mean'],parent['mean']) or not np.array_equal(model['scale'],parent['scale'])):
            raise ValueError('Changed fit roles/sampling/frozen standardization')
        labels=TrainingLabels(a.artifact_root/'r2/sem_001/labels',role['fit'],lm,[1320,989])
        counts=np.zeros(5,np.int64);mass=np.zeros(5);sx=np.zeros((5,43));sxx=np.zeros((5,43,43));sy=np.zeros(5);sxy=np.zeros((5,43));block_rows=np.zeros(5,np.int64)
        for name in role['fit']:
            guard(a.output,c,a.deadline);gt=labels.read(name)
            for k in range(5):counts[k]+=np.count_nonzero(gt==k)
            if bool(model['fallback']):continue
            data=load(ss,name);eligible=proposals(data);pool=np.where(eligible.any(0))[0]
            token=hashlib.sha256(f"{c['seed']}/{role['fold']}/{name}".encode()).digest();rng=np.random.Generator(np.random.PCG64(int.from_bytes(token[:8],'little')))
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
        del labels
        for name in role['evaluation']:
            guard(a.output,c,a.deadline);data=load(ss,name);actual=reference_predict(data,model)
            if not np.array_equal(actual,np.asarray(Image.open(folder/'predictions'/name))):raise ValueError('Independent native replay differs: '+name)
            del data
        audits.append(dict(head=role['id'],fit_images=len(role['fit']),replayed=len(role['evaluation']),block_rows=block_rows.tolist(),
            global_weight_total=float(total),block_weight_mass=(mass/total).tolist() if total else mass.tolist(),ridge_equation_max_abs=residuals,standardization_exact=True))
        print(role['id'],'independent block/pixel audit passed',flush=True)
    labels=TrainingLabels(a.artifact_root/'r2/sem_001/labels',sum(c['views'].values(),[]),lm,[1320,989]);rows=[]
    from scripts.diagnose_quality_complementarity import checked_image
    baseline=read(ROOT/c['comparison']/'result.json')
    for role in c['roles']:
        for name in role['evaluation']:
            guard(a.output,c,a.deadline);gt=labels.read(name);raw=checked_image(ROOT/c['comparison'],baseline,'calibration/mask2former_6k/raw/'+name)
            pred=np.asarray(Image.open(a.output/role['id']/'predictions'/name));cube=np.zeros((5,5,5),np.int64)
            for t in range(5):
                keep=gt==t;np.add.at(cube[t],(raw[keep],pred[keep]),1)
            rows.append(dict(name=name,gt_raw_selector_cube=cube.tolist()))
    verify_hashes(a.output,seal['files'])
    write_new_json(a.output/'audit.json',dict(status='passed',heads=audits,full_native_views=108,integer_cube_rows=rows,
        seconds=time.time()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,seal_sha256=sha256(a.output/'predictions_sealed.json')))


if __name__=='__main__':main()
