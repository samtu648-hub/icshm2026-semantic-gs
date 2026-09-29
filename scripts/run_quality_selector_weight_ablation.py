#!/usr/bin/env python3
"""SEM356 fixed nine-head queue. Fit workers cannot read their evaluation labels."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.quality_mask2former_final import FinalContext
from src.quality_static_training import TrainingLabels, semantic_access
from src.quality_semantic_audit import metrics
from src.quality_learned_selector import neighbourhood, sample_rows, feature_rows, ridge_fit, predict
from scripts.diagnose_quality_raw_fusion import read_probability
from scripts.diagnose_quality_complementarity import checked_image
from scripts.create_rgb_split import parse_colmap_images, kmeans_labels
CONFIG = 'experiments/runs/SEM-356-SELECTOR-WEIGHT-ABLATION/config.yaml'
CODE = ['scripts/run_quality_selector_weight_ablation.py', 'src/quality_learned_selector.py', 'scripts/audit_quality_selector_weight_ablation.py', 'tests/test_quality_learned_selector.py', 'tests/test_quality_selector_weight_ablation.py', 'src/quality_protocol.py', 'src/quality_static_training.py', 'src/quality_semantic_audit.py', 'src/quality_mask2former_final.py', 'scripts/diagnose_quality_raw_fusion.py', 'scripts/diagnose_quality_complementarity.py', 'scripts/create_rgb_split.py']

def read(p):
    return json.loads(Path(p).read_text())

def guard(batch, c, deadline):
    used = sum(p.stat().st_size for p in batch.rglob('*') if p.is_file())
    if (time.time() > deadline or used > c['resources']['maximum_output_mib']*2**20
            or shutil.disk_usage(ROOT).free < c['resources']['minimum_free_gib']*2**30):
        raise RuntimeError('Registered time/output/disk budget reached')
    return used

def sources(c, fold):
    return {k: (ROOT/v, read(ROOT/v/'result.json')) for k, v in c['sources'][fold].items()}

def load(s, name):
    stem = Path(name).stem
    p = [read_probability(*s['five'], f'probabilities/{stem}.npz', 5)]
    maps = [checked_image(*s['static'], 'predictions/'+name)]
    for step in (3000, 6000):
        q = read_probability(*s['b2'], f'step-{step}/probabilities/{stem}.npz', 5)
        m = checked_image(*s['b2'], f'step-{step}/predictions/{name}')
        if not np.array_equal(q.argmax(0), m):
            raise ValueError('B2 argmax differs')
        p.append(q); maps.append(m)
    b = []
    for kind, cid in [('deck', 1), ('stay_cable', 2)]:
        q = read_probability(*s[kind], f'probabilities/{stem}.npz', 2)
        m = checked_image(*s[kind], 'predictions/'+name)
        if not np.array_equal(q.argmax(0), m):
            raise ValueError('Binary argmax differs')
        b.append(q[1].copy()); maps.append(np.where(m == 1, cid, 5).astype(np.uint8))
    raw = p[0].argmax(0).astype(np.uint8)
    return raw, np.stack(maps), p, np.stack(b), neighbourhood(raw)

def provenance(c, ctx):
    verify_hashes(ROOT, c['input_hashes'])
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != c['git_commit']:
        raise ValueError('Git drift')
    cams = {v.name: v for v in parse_colmap_images(ROOT/c['camera_metadata'])}
    parent = read(ROOT/c['comparison']/'result.json')
    names = sum(c['views'].values(), [])
    if len(set(names)) != 108 or sorted(names) != sorted(parent['roles']['calibration']):
        raise ValueError('Original108 membership differs')
    if set(names) & (set(parent['roles']['check']) | set(ctx.access['prediction'])):
        raise ValueError('Check/development access')
    proof = {}
    for fold, ns in c['views'].items():
        expected = semantic_access(ctx.parent.split, ctx.parent.oof, read(ctx.parent.rgb_run/'access.json'), fold)
        for k, (root, rec) in sources(c, fold).items():
            access, binding = rec['access'], rec['binding']
            if any(access[key] != expected[key] for key in ('partition', 'train', 'prediction', 'guard')):
                raise ValueError('Same-base-fold roles differ')
            if binding.get('rgb_identity', binding.get('identity')) != ctx.binding['rgb_identity'] or binding['seed'] != c['seed']:
                raise ValueError('Base binding differs')
            if set(ns) & set(access['train']) or not set(ns) <= set(access['prediction']):
                raise ValueError('Base label leakage')
            for key in ('class_weight_views', 'labels_allowed_during_fit'):
                if key in access and not set(access[key]) <= set(expected['train']):
                    raise ValueError('Base fit statistics leakage')
            modelnames = {'five':['model.pt'], 'static':['semantic_field.pt'], 'b2':['expert-3000.pt','expert-6000.pt'], 'deck':['expert.pt'], 'stay_cable':['expert.pt']}[k]
            hashes = {n: rec['artifacts'][n] for n in modelnames}
            verify_hashes(root, hashes)
            proof[fold+'/'+k] = dict(result_sha256=sha256(root/'result.json'), models=hashes, target_label_overlap=0)
        records = [cams[n] for n in ns]
        clusters = kmeans_labels(records, 3, c['seed'])
        for i in range(3):
            role = next(x for x in c['roles'] if x['id'] == fold+'-block'+str(i))
            evaluation = [v.name for v, g in zip(records, clusters) if g == i]
            pool = [n for n in ns if n not in evaluation]
            guarded = [n for n in pool if min(math.dist(cams[n].center, cams[t].center) for t in evaluation) < c['guard_radius']]
            fit = [n for n in pool if n not in guarded]
            if role['evaluation'] != evaluation or role['fit'] != fit or role['guard'] != guarded:
                raise ValueError('Camera blocks or guard differ')
    if sorted(sum([r['evaluation'] for r in c['roles']], [])) != sorted(names):
        raise ValueError('Evaluation not exactly once')
    return proof

def worker(a, c, role):
    batch, folder = a.output, a.output/role['id']
    folder.mkdir(exist_ok=False)
    start = time.time()
    guard(batch, c, a.deadline)
    seal = read(batch/'started.json')
    verify_hashes(ROOT, {CONFIG:seal['config_sha256'], **seal['code_sha256'], **c['input_hashes']})
    s = sources(c, role['fold'])
    lm = read(ROOT/'experiments/runs/SEM-001/label_manifest.json')
    labels = TrainingLabels(a.artifact_root/'r2/sem_001/labels', role['fit'], lm, [1320,989])
    counts = np.zeros(5, np.int64)
    for name in role['fit']:
        guard(batch,c,a.deadline)
        gt = labels.read(name)
        actual = np.bincount(gt.ravel(), minlength=5)
        if actual.tolist() != [labels.records[name]['pixel_counts'].get(str(i), 0) for i in range(5)]:
            raise ValueError('Fit GT counts differ')
        counts += actual
    parent_fit = read(ROOT/c['paired_parent_batch']/role['id']/'fit.json')
    if parent_fit['role'] != role or counts.tolist() != parent_fit['fit_gt_counts']:
        raise ValueError('Matched parent fit role or GT counts drift')
    fallback = bool(np.any(counts == 0))
    journal, xs, ys, weights = [], [], [], []
    if not fallback:
        cw = np.ones(5, dtype=np.float64)
        for name in role['fit']:
            guard(batch,c,a.deadline)
            data = load(s,name)
            ix, cls = sample_rows(data,c['seed'],role['fold'],name,c['learner']['max_pixels_per_view'])
            journal.append(dict(name=name, rows=len(ix), pixels=len(np.unique(ix)), sample_sha256=hashlib.sha256(np.column_stack([ix,cls]).astype('<i8').tobytes()).hexdigest()))
            if len(ix):
                gt = labels.read(name).ravel()[ix]
                x = feature_rows(data,ix,cls)
                xs.append(x)
                ys.append((gt == cls).astype(float)-(gt == data[0].ravel()[ix]).astype(float))
                weights.append(cw[gt]/len(ix))
            del data
        if journal != parent_fit['sampling']:
            raise ValueError('Matched prediction-only samples differ from SEM355')
        fallback = not xs
    if fallback:
        model = dict(fallback=np.array(True), mean=np.zeros(43), scale=np.ones(43), coef=np.zeros(43))
    else:
        x,y,w = np.concatenate(xs),np.concatenate(ys),np.concatenate(weights)
        del xs,ys,weights
        model = dict(ridge_fit(x,y,w,c['learner']['ridge']),fallback=np.array(False))
        del x,y,w
    np.savez(folder/'model.npz', **model)
    write_new_json(folder/'fit.json',dict(role=role,fit_labels_accessed=sorted(labels.accessed),fit_gt_counts=counts.tolist(),sampling=journal,model_sha256=sha256(folder/'model.npz'),fallback=bool(fallback),fit_seconds=time.time()-start))
    del labels
    (folder/'predictions').mkdir()
    preds={}
    for name in role['evaluation']:
        guard(batch,c,a.deadline)
        data=load(s,name)
        result=predict(data,model)
        Image.fromarray(result).save(folder/'predictions'/name)
        preds[name]=sha256(folder/'predictions'/name)
        del data
    write_new_json(folder/'result.json',dict(role=role,model_sha256=sha256(folder/'model.npz'),fit_sha256=sha256(folder/'fit.json'),predictions=preds,seconds=time.time()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,eval_labels_read_in_worker=0))
    print(role['id'], 'completed',len(preds),'views',round(time.time()-start,2),'seconds',flush=True)

def score(a,c):
    seal=read(a.output/'predictions_sealed.json')
    verify_hashes(a.output,seal['files'])
    names=sum(c['views'].values(),[])
    labels=TrainingLabels(a.artifact_root/'r2/sem_001/labels',names,read(ROOT/'experiments/runs/SEM-001/label_manifest.json'),[1320,989])
    parent=read(ROOT/c['comparison']/'result.json')
    rows=[]
    for role in c['roles']:
        s=sources(c,role['fold'])
        for name in role['evaluation']:
            guard(a.output,c,a.deadline)
            gt=labels.read(name)
            q=read_probability(*s['five'],f'probabilities/{Path(name).stem}.npz',5)
            raw=checked_image(ROOT/c['comparison'],parent,'calibration/mask2former_6k/raw/'+name)
            if not np.array_equal(raw,q.argmax(0)):
                raise ValueError('Baseline raw differs')
            static=checked_image(*s['static'],'predictions/'+name)
            selected=raw.copy(); selected[(raw==3)&(static==4)&(q[4]>=.1)&((q[3]-q[4])<=.05)]=4
            fused=checked_image(ROOT/c['comparison'],parent,'calibration/mask2former_6k/calibrated/'+name)
            pred=np.array(Image.open(a.output/role['id']/'predictions'/name))
            arms={'raw':raw,'foundation005':selected,'historical_fused':fused,'selector':pred}
            cms={k:np.bincount((gt.astype(np.int64)*5+v).ravel(),minlength=25).reshape(5,5).tolist() for k,v in arms.items()}
            cube=np.bincount((gt.astype(np.int64)*25+raw.astype(np.int64)*5+pred).ravel(),minlength=125).reshape(5,5,5)
            rows.append(dict(name=name,fold=role['fold'],head=role['id'],confusions=cms,gt_raw_selector_cube=cube.tolist()))
    write_new_json(a.output/'per_view.json',rows)
    independent = {r['name']: r['gt_raw_selector_cube'] for r in read(a.output/'audit.json')['integer_cube_rows']}
    if any(r['gt_raw_selector_cube'] != independent[r['name']] for r in rows):
        raise ValueError('Independent integer cube mismatch')
    def summary(part):
        m={k:metrics(sum(np.array(r['confusions'][k],np.int64) for r in part)) for k in rows[0]['confusions']}
        cube=sum(np.array(r['gt_raw_selector_cube'],np.int64) for r in part)
        corrected=[int(sum(cube[t,p,t] for p in range(5) if p!=t)) for t in range(5)]
        harmed=[int(sum(cube[t,t,p] for p in range(5) if p!=t)) for t in range(5)]
        return dict(metrics=m,delta_miou_pp=100*(m['selector']['mIoU_with_background']-m['raw']['mIoU_with_background']),corrected_by_class=corrected,harmed_by_class=harmed,gt_raw_selector_cube=cube.tolist())
    return dict(pooled=summary(rows),folds={f:summary([r for r in rows if r['fold']==f]) for f in c['views']},heads={role['id']:summary([r for r in rows if r['head']==role['id']]) for role in c['roles']},labels_accessed=sorted(labels.accessed))

def matched_parent(c):
    parent=read(ROOT/'experiments/runs/SEM-355-LEARNED-SELECTOR-PILOT/config.yaml')
    scientific=('git_commit','seed','dataset_manifest','split_id','final_batch','comparison','views','sources','roles','camera_metadata','guard_radius')
    if any(c[k] != parent[k] for k in scientific):
        raise ValueError('Non-weight scientific variable differs')
    if {k:v for k,v in c['learner'].items() if k!='class_weight'} != {k:v for k,v in parent['learner'].items() if k!='class_weight'}:
        raise ValueError('Unregistered learner change')
    if c['learner']['class_weight'] != 'constant 1 for all GT classes; equal view total':
        raise ValueError('Incorrect ablation weight policy')
    seal=read(ROOT/c['paired_parent_batch']/'predictions_sealed.json')
    verify_hashes(ROOT/c['paired_parent_batch'],seal['files'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker');p.add_argument('--deadline',type=float)
    p.add_argument('--resume-from',type=Path)
    a=p.parse_args();a.output=a.output.resolve();c=read(ROOT/CONFIG)
    if not a.output.is_relative_to(ROOT/'outputs/quality/learned_selector'):
        p.error('Isolated learned_selector output required')
    if a.resume_from:
        a.resume_from = a.resume_from.resolve()
    if a.worker:
        worker(a,c,next(r for r in c['roles'] if r['id']==a.worker));return
    a.output.mkdir(parents=True,exist_ok=False)
    start=time.time();a.deadline=start+c['resources']['maximum_seconds']
    if a.resume_from:
        previous=read(a.resume_from/'started.json')
        if previous['config_sha256'] != sha256(ROOT/CONFIG):
            raise ValueError('Resume cannot change protocol')
        a.deadline=previous['deadline']
    record=dict(experiment_id=c['experiment_id'],git_commit=c['git_commit'],seed=c['seed'],dataset_manifest=c['dataset_manifest'],split_id=c['split_id'],config_sha256=sha256(ROOT/CONFIG),code_sha256={n:sha256(ROOT/n) for n in CODE},started_unix=start,deadline=a.deadline,status='started',gpu_training=False,base_training=False)
    write_new_json(a.output/'started.json',record)
    for n in [CONFIG,*CODE]:
        dst=a.output/'code'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dst)
    try:
        ctx=FinalContext(ROOT,a.artifact_root,ROOT/c['final_batch'])
        matched_parent(c)
        write_new_json(a.output/'provenance_before_fit.json',provenance(c,ctx))
        reused=[]
        for role in c['roles']:
            guard(a.output,c,a.deadline)
            if a.resume_from and (a.resume_from/role['id']/'result.json').is_file():
                old=a.resume_from/role['id'];rec=read(old/'result.json')
                if rec['role'] != role:
                    raise ValueError('Completed role changed')
                verify_hashes(old,{'model.npz':rec['model_sha256'],'fit.json':rec['fit_sha256'],**{'predictions/'+n:h for n,h in rec['predictions'].items()}})
                shutil.copytree(old,a.output/role['id'])
                reused.append(dict(head=role['id'],source=str(old.relative_to(ROOT)),result_sha256=sha256(old/'result.json'),original_code_sha256=previous['code_sha256']))
                print(role['id'],'reused completed head; no retraining',flush=True)
                continue
            cmd=[sys.executable,__file__,'--artifact-root',str(a.artifact_root),'--output',str(a.output),'--worker',role['id'],'--deadline',str(a.deadline)]
            with (a.output/(role['id']+'.log')).open('x') as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,a.deadline-time.time()))
            print((a.output/(role['id']+'.log')).read_text(),end='',flush=True)
        write_new_json(a.output/'resume_provenance.json',dict(reused=reused,new_fits=9-len(reused),previous_failure_sha256=sha256(a.resume_from/'failure.json') if a.resume_from else None))
        files={str(p.relative_to(a.output)):sha256(p) for p in a.output.rglob('*') if p.is_file() and ('predictions' in p.parts or p.name in ('model.npz','fit.json','result.json'))}
        write_new_json(a.output/'predictions_sealed.json',dict(files=files,config_sha256=record['config_sha256'],sealed_unix=time.time(),heads=9,views=108,no_inner_selection=True))
        with (a.output/'audit.log').open('x') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts/audit_quality_selector_weight_ablation.py'),'--output',str(a.output),'--artifact-root',str(a.artifact_root),'--deadline',str(a.deadline)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,a.deadline-time.time()))
        result=score(a,c)
        ctx.finish();verify_hashes(ROOT,{CONFIG:record['config_sha256'],**record['code_sha256'],**c['input_hashes']})
        record.update(status='completed_fixed_development_evaluation',**result,seconds=time.time()-start,output_bytes=guard(a.output,c,a.deadline),independent_audit_sha256=sha256(a.output/'audit.json'),prediction_seal_sha256=sha256(a.output/'predictions_sealed.json'),frozen_rgb293_and18_unchanged=True,new_check_or74_scores=0)
        write_new_json(a.output/'result.json',record)
        print('SEM356 complete',record['pooled']['delta_miou_pp'],flush=True)
    except BaseException as exc:
        write_new_json(a.output/'failure.json',dict(type=type(exc).__name__,message=str(exc),elapsed_seconds=time.time()-start));raise

if __name__=='__main__':main()
