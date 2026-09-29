#!/usr/bin/env python3
"""Independent SEM357 feature/moment reconstruction and full saved-head replay."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import hashlib
from pathlib import Path
import resource
import sys
import time
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_quality_miou_utility_selector import read,load,sources,guard,CONFIG
from src.quality_protocol import sha256,verify_hashes,write_new_json
from src.quality_static_training import TrainingLabels
from src.quality_miou_utility import reference_utility_table
from scripts.diagnose_quality_complementarity import checked_image

def local_reference(raw):
    h,w=raw.shape
    padded=np.pad(raw.astype(np.int16),1,constant_values=-1)
    same=np.zeros(raw.shape,float);total=np.zeros(raw.shape,float)
    for j in range(3):
        for i in range(3):
            v=padded[j:j+h,i:i+w];same+=v==raw;total+=v>=0
    return same/total

def reference_rows(data,indices,labels):
    raw,maps,prob,binary,near=data
    out=np.zeros((len(indices),43),np.float64)
    row=np.arange(len(indices));r=raw.reshape(-1)[indices]
    for j in range(3):
        for c in range(5):out[:,j*5+c]=prob[j][c].reshape(-1)[indices]
    for j in range(2):out[:,15+j]=binary[j].reshape(-1)[indices]
    out[row,17+maps[0].reshape(-1)[indices]]=1
    out[row,22+r]=1;out[row,27+labels]=1
    out[:,32]=out[row,labels]-out[row,r]
    out[:,33]=out[row,10+labels]-out[row,10+r]
    for j in range(5):out[:,36+j]=maps[j].reshape(-1)[indices]==labels
    out[:,34]=out[:,36:41].sum(axis=1)
    for c in range(5):out[:,35]-=out[:,c]*np.log(np.maximum(out[:,c],1e-30))
    out[:,41]=near.reshape(-1)[indices];out[:,42]=1
    return out

def proposals(data):
    raw,maps=data[:2]
    answer=np.zeros((5,raw.size),bool)
    for j in range(5):
        for c in range(5):answer[c]|=(maps[j].ravel()==c)&(raw.ravel()!=c)
    return answer

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
    c=read(ROOT/CONFIG);seal=read(a.output/'predictions_sealed.json');verify_hashes(a.output,seal['files'])
    lm=read(ROOT/'experiments/runs/SEM-001/label_manifest.json');audit=[];start=time.time()
    for role in c['roles']:
        folder=a.output/role['id'];fit=read(folder/'fit.json');model=dict(np.load(folder/'model.npz',allow_pickle=False));s=sources(c,role['fold'])
        if fit['role']!=role or fit['fit_labels_accessed']!=sorted(role['fit']) or set(role['fit'])&(set(role['evaluation'])|set(role['guard'])):raise ValueError('Fit access leak')
        labels=TrainingLabels(a.artifact_root/'r2/sem_001/labels',role['fit'],lm,[1320,989])
        counts=np.zeros(5,np.int64)
        fit_cm=np.zeros((5,5),np.int64)
        comparator=read(ROOT/c['comparison']/'result.json')
        for name in role['fit']:
            gt=labels.read(name)
            for k in range(5):counts[k]+=np.count_nonzero(gt==k)
            raw=checked_image(ROOT/c['comparison'],comparator,'calibration/mask2former_6k/raw/'+name)
            np.add.at(fit_cm,(gt.ravel(),raw.ravel()),1)
        if counts.tolist()!=fit['fit_gt_counts']:raise ValueError('Fit class statistics')
        if fit_cm.tolist()!=fit['fit_raw_confusion']:raise ValueError('Independent fit raw confusion')
        utility=reference_utility_table(fit_cm) if not bool(model['fallback']) else None
        if utility is not None:
            expected=np.array(fit['utility_table'])
            valid=np.isfinite(utility)
            if not np.allclose(utility[valid],expected[valid],atol=1e-12,rtol=1e-12):raise ValueError('Rational utility target audit')
        weight_total=0.;sumx=np.zeros(43);sumxx=np.zeros((43,43));sumy=0.;sumxy=np.zeros(43)
        if not bool(model['fallback']):
            cw=np.full(5, 1.0)
            for name in role['fit']:
                guard(a.output,c,a.deadline);data=load(s,name)
                local=local_reference(data[0])
                if not np.array_equal(local,data[4]):raise ValueError('Neighbourhood border mismatch')
                data=(*data[:4],local)
                allowed=proposals(data);pool=np.nonzero(np.max(allowed,axis=0))[0]
                token=hashlib.sha256(f"{c['seed']}/{role['fold']}/{name}".encode()).digest()
                rng=np.random.Generator(np.random.PCG64(int.from_bytes(token[:8],'little')))
                if len(pool)>c['learner']['max_pixels_per_view']:pool=np.sort(rng.choice(pool,c['learner']['max_pixels_per_view'],replace=False))
                cs,positions=np.where(allowed[:,pool]);ix=pool[positions]
                row=next(j for j in fit['sampling'] if j['name']==name)
                digest=hashlib.sha256(np.stack((ix,cs),axis=1).astype('<i8').tobytes()).hexdigest()
                if digest!=row['sample_sha256'] or len(ix)!=row['rows']:raise ValueError('Sampling mismatch')
                if len(ix):
                    x=reference_rows(data,ix,cs);truth=labels.read(name).reshape(-1)[ix];base=data[0].reshape(-1)[ix]
                    y=utility[truth,base,cs];w=cw[truth]/len(ix)
                    weight_total+=w.sum();sumx+=np.sum(w[:,None]*x,axis=0);sumxx+=x.T@(w[:,None]*x);sumy+=np.dot(w,y);sumxy+=x.T@(w*y)
                del data
            mean=sumx/weight_total;mean[-1]=0
            variance=np.diag(sumxx)/weight_total-2*mean*(sumx/weight_total)+mean**2
            scale=np.sqrt(np.maximum(variance,0));scale[scale<1e-12]=1;scale[-1]=1
            # Directly centered stored scale avoids cancellation in nearly constant features.
            if not np.allclose(mean,model['mean'],atol=1e-10,rtol=1e-10) or not np.allclose(scale,model['scale'],atol=1e-7,rtol=1e-7):raise ValueError('Independent train-only standardization')
            m=model['mean'];z=model['scale']
            gram=(sumxx/weight_total-np.outer(m,sumx/weight_total)-np.outer(sumx/weight_total,m)+np.outer(m,m))/np.outer(z,z)
            rhs=(sumxy/weight_total-m*sumy/weight_total)/z
            if not np.allclose(gram,model['gram'],atol=1e-7,rtol=1e-7) or not np.allclose(rhs,model['rhs'],atol=1e-10,rtol=1e-9):raise ValueError('Independent ridge moments')
            reg=np.eye(43)*c['learner']['ridge'];reg[-1,-1]=0
            residual=float(np.max(np.abs((gram+reg)@model['coef']-rhs)))
            if residual>1e-7:raise ValueError('Ridge normal equation residual')
        else:
            if not np.any(counts==0) and any(j['rows'] for j in fit['sampling']):raise ValueError('Unsupported fallback')
            residual=None
        del labels
        for name in role['evaluation']:
            guard(a.output,c,a.deadline);data=load(s,name);raw=data[0];allowed=proposals(data)
            data=(*data[:4],local_reference(raw));answer=raw.reshape(-1).copy()
            if not bool(model['fallback']):
                scores=np.full(allowed.shape,-np.inf)
                # Algebraically fold normalization into weights, using separate feature code.
                beta=model['coef']/model['scale'];offset=-np.dot(model['mean'],beta)
                for cid in range(5):
                    pix=np.where(allowed[cid])[0]
                    for k in range(0,len(pix),32768):
                        ix=pix[k:k+32768];scores[cid,ix]=reference_rows(data,ix,np.full(len(ix),cid))@beta+offset
                highest=np.max(scores,axis=0);win=np.argmax(scores,axis=0)
                accept=(highest>0)&(np.sum(scores==highest,axis=0)==1)
                answer[accept]=win[accept]
            original=np.array(Image.open(folder/'predictions'/name))
            if not np.array_equal(answer.reshape(raw.shape),original):raise ValueError('Independent full native replay differs: '+name)
            del data
        audit.append(dict(utility_rational_checks=int(np.isfinite(utility).sum()) if utility is not None else 0,head=role['id'],fit_images=len(role['fit']),replayed=len(role['evaluation']),ridge_equation_max_abs=residual,fit_label_whitelist_exact=True))
        print(role['id'],'independent reconstruction passed',flush=True)
    # The seal predates every read in this evaluation stage.
    labels=TrainingLabels(a.artifact_root/'r2/sem_001/labels',sum(c['views'].values(),[]),lm,[1320,989]);rows=[]
    parent=read(ROOT/c['comparison']/'result.json')
    for role in c['roles']:
        for name in role['evaluation']:
            guard(a.output,c,a.deadline);gt=labels.read(name);raw=checked_image(ROOT/c['comparison'],parent,'calibration/mask2former_6k/raw/'+name);out=np.array(Image.open(a.output/role['id']/'predictions'/name))
            cube=np.zeros((5,5,5),np.int64)
            for t in range(5):
                mask=gt==t
                np.add.at(cube[t],(raw[mask],out[mask]),1)
            rows.append(dict(name=name,gt_raw_selector_cube=cube.tolist()))
    write_new_json(a.output/'audit.json',dict(status='passed',heads=audit,full_native_views=108,integer_cube_rows=rows,seconds=time.time()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,seal_sha256=sha256(a.output/'predictions_sealed.json')))
if __name__=='__main__':main()
