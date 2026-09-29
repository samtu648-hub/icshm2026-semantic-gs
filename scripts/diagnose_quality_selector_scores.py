#!/usr/bin/env python3
"""Frozen own-fold SEM361/365 score failure strata; no fitting or new rules."""
import argparse,json,sys,time,resource,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_quality_selector_weight_ablation import load,sources
from src.quality_learned_selector import candidates,feature_rows,choose
from scripts.diagnose_quality_expert_strata import boundary_bins
import numpy as np
from PIL import Image
CONFIG='experiments/runs/QUAL-006-FUSION-DECISION-DIAG/config.yaml'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
STATUS=['pixels','raw_wrong','no_correct_candidate','recoverable','recovered','correct_score_nonpositive','lost_positive_competition','positive_tie','raw_correct_damaged']
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);a=p.parse_args();c=read(ROOT/CONFIG);out=ROOT/c['output'];out.mkdir(exist_ok=False,parents=True);start=time.time();rows=[];proof={};binscore=[-np.inf,-.5,-.1,-.01,0,.01,.1,.5,np.inf]
 labels={x['image']:x['label_png_sha256'] for x in read(ROOT/'experiments/runs/SEM-001/label_manifest.json')['records']}
 try:
  shutil.copy2(__file__,out/Path(__file__).name);shutil.copy2(ROOT/CONFIG,out/'config.json')
  for role in c['roles']:
   models=[]
   for rel in c['parent_batches']:
    base=ROOT/rel/role['id'];fit=read(base/'fit.json');assert fit['role']==role
    assert not set(role['fit'])&set(role['evaluation']) and not set(role['guard'])&set(role['evaluation'])
    assert sha(base/'model.npz')==fit['model_sha256'];proof[str(base.relative_to(ROOT)/'model.npz')]=fit['model_sha256']
    with np.load(base/'model.npz',allow_pickle=False) as z:m={k:z[k] for k in z.files}
    assert not bool(m['fallback']);models.append((base,m,read(base/'result.json')))
   ss=sources(c,role['fold'])
   for kind,(path,rec) in ss.items():
    assert not set(role['evaluation'])&set(rec['access']['train']) and set(role['evaluation'])<=set(rec['access']['prediction'])
   for name in role['evaluation']:
    gtpath=a.artifact_root/'r2/sem_001/labels'/name;assert sha(gtpath)==labels[name];gt=np.array(Image.open(gtpath));distance=boundary_bins(gt).ravel();g=gt.ravel()
    data=load(ss,name);raw=data[0].ravel();allowed=candidates(data[0],data[1]).reshape(5,-1);ix=np.arange(len(g));recover=(raw!=g)&allowed[g,ix]
    allscores=[np.full(allowed.shape,-np.inf) for _ in models]
    for cid in range(5):
     pixels=np.flatnonzero(allowed[cid])
     for startix in range(0,len(pixels),32768):
      pos=pixels[startix:startix+32768];x=feature_rows(data,pos,np.full(len(pos),cid))
      for j,(_,m,_) in enumerate(models):
       coef=m['coef'] if m['coef'].ndim==1 else m['coef'][cid]
       z=(x-m['mean'])/m['scale'];val=z@coef
       if m['coef'].ndim==2 and not m['active'][cid]:val=np.zeros(len(pos))
       # Algebraically independent row dot products on bounded rows, not another fit.
       if len(pos):assert np.allclose(val[:3],np.sum(z[:3]*coef,axis=1),atol=1e-10,rtol=1e-10)
       allscores[j][cid,pos]=val
    for (base,m,rec),scores in zip(models,allscores):
     pred=choose(raw,allowed,scores);saved=base/'predictions'/name
     assert sha(saved)==rec['predictions'][name]
     assert np.array_equal(pred.reshape(gt.shape),np.array(Image.open(saved))),str(saved)
     score=scores[g,ix];mx=scores.max(0);ties=(scores==mx).sum(0)>1
     sets=[np.ones(g.shape,bool),raw!=g,(raw!=g)&~recover,recover,recover&(pred==g),recover&(pred!=g)&(score<=0),recover&(pred!=g)&(score>0)&(mx>score),recover&(pred!=g)&(score>0)&(mx==score)&ties,(raw==g)&(pred!=g)]
     assert np.array_equal(sets[3],sets[4]|sets[5]|sets[6]|sets[7])
     stat=np.stack([np.bincount((g*4+distance)[v],minlength=20).reshape(5,4) for v in sets],axis=-1)
     hist={str(cid):np.histogram(score[recover&(pred!=g)&(g==cid)],bins=binscore)[0].tolist() for cid in range(5)}
     rows.append({'head':base.parent.name,'role':role['id'],'fold':role['fold'],'name':name,'counts':stat.tolist(),'unused_correct_score_histogram':hist})
    print(role['id'],name,'scores checked',flush=True)
  result={k:c[k] for k in ['experiment_id','git_commit','seed','dataset_manifest','split_id']};result.update(status='completed_frozen_score_diagnosis',seconds=time.time()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,config_sha256=sha(ROOT/CONFIG),code_sha256=sha(__file__),models=proof,axes=['GT class','boundary_distance_bin','status'],statuses=STATUS,distance_bins=['<=2','2-8','8-32','>32'],score_bins=['-inf','-.5','-.1','-.01','0','.01','.1','.5','inf'],rows=rows,training=0)
  (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 except BaseException as e:
  (out/'failure.json').write_text(json.dumps({'error':repr(e),'seconds':time.time()-start,'completed_rows':rows})+'\n');raise
if __name__=='__main__':main()
