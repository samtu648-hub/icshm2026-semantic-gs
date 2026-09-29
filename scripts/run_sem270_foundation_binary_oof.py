#!/usr/bin/env python3
"""Matched native-resolution five-class versus dedicated foundation training."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from audit_sem260_error_budget import manifest_paths, png
from run_sem250_colmap_track_deck_diag import verify_freeze
from src.semantic2d import read_json, write_json, sha256, confusion_metrics, CLASS_NAMES
from src.semantic_cable_binary import normalize_rgb
from src.semantic_foundation_binary import (ARMS, target_labels, fuse_foundation,
    training_plan, validate_fold, quality_gates)
from src.semantic_protected_fusion import partition_training_names, local_confusion
from src.semantic_render_domain import semantic_rgb
from src.submission_sem161 import sliding

EXP='SEM-270-FOUNDATION-BINARY-OOF'


def log(**values):
    print(json.dumps(values),flush=True)


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def initialize(modeldir, arm, seed):
    from transformers import SegformerForSemanticSegmentation
    names=CLASS_NAMES if arm==ARMS[0] else ['non_foundation','foundation']
    torch.manual_seed(seed)
    return SegformerForSemanticSegmentation.from_pretrained(modeldir,local_files_only=True,
        num_labels=len(names),id2label=dict(enumerate(names)),
        label2id={n:i for i,n in enumerate(names)},ignore_mismatched_sizes=True)


def shared_state_hash(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        if k.startswith('decode_head.classifier.'):
            continue
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def resource_gate(out,cfg,started):
    if shutil.disk_usage(out).free < cfg['minimum_free_disk_gib']*2**30:
        raise RuntimeError('Disk floor reached; no artifact deletion')
    if time.perf_counter()-started > cfg['maximum_gpu_hours']*3600:
        raise RuntimeError('Registered wall-clock budget reached')


def fit(out,art,cfg,fold,arm,plan,rgbpaths,labels,smoke,started):
    folder=out/f"fold{fold['fold']}_{arm}"
    folder.mkdir()
    names=sorted({r['name'] for r in plan}) if smoke else sorted(fold['train'])
    seed=cfg['seed'];random.seed(seed);np.random.seed(seed)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    classes=5 if arm==ARMS[0] else 2
    counts=np.zeros(classes,np.int64);cache={};access=[]
    for name in names:
        if name not in fold['train'] or name in fold['oof'] or name in fold['guard']:
            raise ValueError('Forbidden training label')
        rgb,gt=png(rgbpaths[name][0]),png(labels/name)
        if rgb.shape!=(989,1320,3) or gt.shape!=(989,1320):
            raise ValueError('Native grid mismatch')
        y=target_labels(gt,arm);counts+=np.bincount(y.ravel(),minlength=classes)
        cache[name]=(rgb,y)
        access.append({'name':name,'label_sha256':sha256(labels/name),'rgb_sha256':rgbpaths[name][1]})
    if np.any(counts<=0): raise ValueError('Missing train class')
    weights=np.clip(np.median(counts.astype(np.float64))/counts,.05,20).astype(np.float32)
    model=initialize(art/cfg['parents']['model_config'],arm,seed).cuda().train()
    shared=shared_state_hash(model)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay'])
    w=torch.from_numpy(weights).cuda();crop=cfg['training']['crop']
    losses=[];begin=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for step,r in enumerate(plan,1):
        rgb,gt=cache[r['name']];y0,x0=r['top'],r['left']
        im,y=rgb[y0:y0+crop,x0:x0+crop],gt[y0:y0+crop,x0:x0+crop]
        if r['flip']: im,y=im[:,::-1],y[:,::-1]
        x=normalize_rgb(np.ascontiguousarray(im)).unsqueeze(0).cuda()
        target=torch.from_numpy(y.copy().astype(np.int64)).unsqueeze(0).cuda()
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=F.interpolate(model(pixel_values=x).logits,(crop,crop),mode='bilinear',align_corners=False)
            loss=F.cross_entropy(logits.float(),target,weight=w)
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
        loss.backward();opt.step();losses.append(float(loss.detach()))
        if step==1 or step%250==0 or step==len(plan):
            log(stage='training',fold=fold['fold'],arm=arm,step=step,total=len(plan),loss=losses[-1],seconds=time.perf_counter()-begin)
            resource_gate(out,cfg,started)
    torch.cuda.synchronize();seconds=time.perf_counter()-begin
    peak=torch.cuda.max_memory_allocated()/2**20
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): replay=model(pixel_values=x).logits.cpu()
    metadata={'experiment_id':EXP,'fold':fold['fold'],'arm':arm,'seed':seed,'iterations':len(plan),
        'train':names,'outer_train':fold['train'],'oof':fold['oof'],'guard':fold['guard'],
        'actual_label_access':access,'actual_optimization_names':sorted({r['name'] for r in plan}),
        'sampling_plan_sha256':sha256(out/f"fold{fold['fold']}_sampling_plan.json"),
        'config_sha256':cfg['_sha256'],'oof_split_sha256':cfg['oof_split_sha256'],
        'initialization':cfg['_initialization'],'shared_initial_state_sha256':shared,
        'class_counts':counts.tolist(),'class_weights':weights.tolist(),
        'mode':'smoke' if smoke else 'full','training_task':'five_class' if classes==5 else 'foundation_vs_all_other_classes',
        'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
    path=folder/'model.pth'
    torch.save({'metadata':metadata,'model_state_dict':model.state_dict()},path)
    del model,opt,cache,logits,loss;torch.cuda.empty_cache()
    payload=torch.load(path,map_location='cpu',weights_only=True)
    fresh=initialize(art/cfg['parents']['model_config'],arm,seed).cuda().eval().requires_grad_(False)
    fresh.load_state_dict(payload['model_state_dict'],strict=True)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): actual=fresh(pixel_values=x).logits.cpu()
    if not torch.equal(actual,replay) or payload['metadata']!=metadata: raise ValueError('Reload mismatch')
    result={'path':str(path),'sha256':sha256(path),'metadata':metadata,'reload_exact':True,
        'seconds':seconds,'peak_cuda_allocated_mib':peak,
        'loss':{'first':losses[0],'last':losses[-1],'last100':float(np.mean(losses[-100:]))}}
    write_json(folder/'training_result.json',result)
    del payload,x,target;torch.cuda.empty_cache()
    return fresh,result


def evaluate(names,predictions,baseline,labels,cfg):
    matrices={a:np.zeros((5,5),np.int64) for a in ('baseline',*ARMS)}
    changes={a:{'added_foundation':0,'removed_foundation':0,'corrected':0,'harmed':0} for a in ARMS}
    per_view=[];access=[]
    for name in names:
        gt,base=png(labels/name),png(baseline[name][0])
        row={'name':name,'baseline':confusion_metrics(local_confusion(gt,base))}
        matrices['baseline']+=local_confusion(gt,base)
        for arm in ARMS:
            p=predictions[arm][name]
            if sha256(Path(p['path']))!=p['sha256']: raise ValueError('Sealed PNG changed')
            new=png(Path(p['path']));protected=np.isin(base,[1,2,3])
            if not np.array_equal(new[protected],base[protected]) or any(not np.array_equal(new==k,base==k) for k in [1,2,3]):
                raise ValueError('Protected class mask changed')
            cm=local_confusion(gt,new);matrices[arm]+=cm;row[arm]=confusion_metrics(cm)
            changed=new!=base
            for key,mask in [('added_foundation',(base==0)&(new==4)),('removed_foundation',(base==4)&(new==0)),
                             ('corrected',changed&(new==gt)),('harmed',changed&(base==gt))]:
                changes[arm][key]+=int(mask.sum())
        per_view.append(row);access.append({'name':name,'label_sha256':sha256(labels/name)})
    metrics={a:confusion_metrics(c) for a,c in matrices.items()}
    checks=quality_gates(metrics['baseline'],metrics[ARMS[1]],metrics[ARMS[0]],cfg['gates'])
    checks['protected_exact']=True
    return {'metrics':metrics,'checks':checks,'passed':all(checks.values()),'changes':changes,'per_view':per_view,'target_access':access}


def panels(names,predictions,baseline,rgbpaths,labels,out):
    out.mkdir();records=[]
    for name in names:
        rgb,gt,base=[png(p) for p in (rgbpaths[name][0],labels/name,baseline[name][0])]
        pred=[png(Path(predictions[a][name]['path'])) for a in ARMS]
        good=(pred[1]!=base)&(pred[1]==gt);bad=(pred[1]!=base)&(base==gt)
        change=np.zeros_like(rgb);change[good]=(40,220,80);change[bad]=(255,50,50)
        columns=[rgb,semantic_rgb(gt),semantic_rgb(base),*[semantic_rgb(p) for p in pred],change]
        titles=['Frozen rendered RGB','GT','SEM-160 OOF','Five-class control','Foundation binary','Green corrected / red harmed']
        h,w=base.shape
        def save(tag,box,scale=1):
            x0,y0,x1,y1=box;cw,ch=(x1-x0)*scale,(y1-y0)*scale
            canvas=Image.new('RGB',(6*cw,ch+24),'white');draw=ImageDraw.Draw(canvas)
            for i,(col,title) in enumerate(zip(columns,titles)):
                crop=Image.fromarray(col[y0:y1,x0:x1]).resize((cw,ch),Image.Resampling.NEAREST)
                canvas.paste(crop,(i*cw,24));draw.text((i*cw+4,4),name+' '+title,fill='black')
            path=out/f'{Path(name).stem}_{tag}.png';canvas.save(path)
            records.append({'path':str(path),'sha256':sha256(path),'crop_xyxy':list(box),'scale':scale})
        save('panel',(0,0,w,h))
        ys,xs=np.where((gt==4)|(base==4)|(pred[1]==4))
        if len(xs): save('foundation_zoom',(max(0,int(xs.min())-16),max(0,int(ys.min())-16),min(w,int(xs.max())+17),min(h,int(ys.max())+17)))
        # Fixed tiling includes both successes and failures; no candidate selection.
        for tag,mask in [('corrected',good),('harmed',bad)]:
            boxes=[(x,y,min(x+224,w),min(y+128,h)) for y in range(0,h-127,64) for x in range(0,w-223,112)]
            box=max(boxes,key=lambda b:int(mask[b[1]:b[3],b[0]:b[2]].sum()))
            save(tag+'_2x',box,2)
    write_json(out/'manifest.json',{'columns':titles,'files':records})
    return records


def run(args,out,started):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('Visible bf16 CUDA required')
    torch.set_num_threads(4)
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error();hf_logging.disable_progress_bar()
    cfg=read_json(args.config);cfg['_sha256']=sha256(args.config)
    if cfg['experiment_id']!=EXP or tuple(cfg['arm_names'])!=ARMS: raise ValueError('Wrong config')
    art=args.artifact_root.resolve()
    sourcepaths=[Path(__file__),ROOT/'src/semantic_foundation_binary.py',ROOT/'src/semantic_cable_binary.py',ROOT/'src/submission_sem161.py',ROOT/'src/semantic2d.py',ROOT/'src/semantic_protected_fusion.py',ROOT/'scripts/audit_sem260_error_budget.py',ROOT/'scripts/run_sem250_colmap_track_deck_diag.py']
    sources={str(p.relative_to(ROOT)):sha256(p) for p in sourcepaths}
    (out/'source_snapshot').mkdir()
    for p in (*sourcepaths,args.config): shutil.copyfile(p,out/'source_snapshot'/p.name)
    freeze=verify_freeze(ROOT/'experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json',art)
    write_json(out/'freeze_before.json',freeze)
    for k in ('semantic_split','oof_split'):
        if sha256(ROOT/cfg[k])!=cfg[k+'_sha256']: raise ValueError('Split changed')
    split=read_json(ROOT/cfg['oof_split']);allnames=read_json(ROOT/cfg['semantic_split'])['semantic_train_labeled']
    graph=ROOT/cfg['camera_graph']
    if sha256(graph)!=split['camera_graph_sha256']: raise ValueError('Camera graph changed')
    centers={n['name']:n['center'] for n in read_json(graph)['nodes']}
    held=[n for f in split['folds'] for n in f['oof']]
    if len(held)!=len(set(held)) or set(held)!=set(allnames): raise ValueError('Outer coverage changed')
    for f in split['folds']: validate_fold(f,allnames,centers,split['guard_policy']['radius'])
    modeldir=art/cfg['parents']['model_config']
    cfg['_initialization']=[{'path':str(p.relative_to(art)),'sha256':sha256(p)} for p in sorted(modeldir.iterdir()) if p.suffix in ('.json','.bin','.safetensors')]
    baseline=manifest_paths(art/cfg['parents']['sem160'],'prediction_manifest.json','predictions','prediction','prediction_sha256')
    rgbpaths=manifest_paths(art/cfg['parents']['rgb'],'rendered_dataset_manifest.json','rendered_images','path','sha256')
    for mapping in (baseline,rgbpaths):
        if set(mapping)!=set(allnames) or any(sha256(p)!=h for p,h in mapping.values()): raise ValueError('Parent image hash/universe changed')
    labels=art/cfg['parents']['labels'];predictions={a:{} for a in ARMS};foldrows=[]
    for fold in split['folds'][:1] if args.smoke else split['folds']:
        fi=fold['fold'];begin=time.perf_counter();resource_gate(out,cfg,started)
        plan=training_plan(sorted(fold['train']),cfg['seed'],fi,2 if args.smoke else cfg['training']['iterations'],cfg['training']['crop'])
        write_json(out/f'fold{fi}_sampling_plan.json',plan);row={'fold':fi,'arms':{}}
        shared=None
        for arm in ARMS:
            model,training=fit(out,art,cfg,fold,arm,plan,rgbpaths,labels,args.smoke,started)
            current=training['metadata']['shared_initial_state_sha256']
            if shared is not None and current!=shared: raise ValueError('Matched shared initialization differs')
            shared=current;inferstart=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            names=sorted(fold['oof'])[:1] if args.smoke else sorted(fold['oof'])
            for i,name in enumerate(names):
                rgb,base=png(rgbpaths[name][0]),png(baseline[name][0]);before=array_hash(rgb)
                classes=5 if arm==ARMS[0] else 2
                p=sliding(model,rgb,crop=cfg['inference']['crop'],stride=cfg['inference']['stride'],classes=classes)[4 if classes==5 else 1]
                if not np.array_equal(fuse_foundation(base,p,strength=0),base): raise ValueError('Disabled equivalence failed')
                new=fuse_foundation(base,p)
                if array_hash(rgb)!=before or any(not np.array_equal(new==k,base==k) for k in [1,2,3]): raise ValueError('RGB/protected changed')
                folder=out/arm;folder.mkdir(exist_ok=True)
                pp=folder/(Path(name).stem+'.npy');np.save(pp,p.astype(np.float32))
                path=folder/name;Image.fromarray(new).save(path)
                predictions[arm][name]={'path':str(path),'sha256':sha256(path),'probability_path':str(pp),'probability_sha256':sha256(pp),'fold':fi}
                if i%20==0 or i+1==len(names): log(stage='inference',fold=fi,arm=arm,views=i+1,total=len(names))
            torch.cuda.synchronize()
            row['arms'][arm]={'training':training,'inference_seconds':time.perf_counter()-inferstart,
                'inference_peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                'disabled_exact_views':len(names),'protected_exact':True,'rgb_exact':True}
            del model;torch.cuda.empty_cache()
        row['seconds']=time.perf_counter()-begin;foldrows.append(row);write_json(out/f'fold{fi}_result.json',row)
    write_json(out/'prediction_manifest.json',predictions)
    seal={'experiment_id':EXP,'config_sha256':cfg['_sha256'],'source_sha256':sources,
        'prediction_manifest_sha256':sha256(out/'prediction_manifest.json'),'folds':foldrows,
        'repository_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'scope':'Incremental OOF; inherited SEM-001 all219; reused calibration/check, not new blind data'}
    write_json(out/'prediction_seal.json',seal)
    if sha256(args.config)!=cfg['_sha256'] or any(sha256(ROOT/p)!=h for p,h in sources.items()): raise ValueError('Source/config changed while running')
    after=verify_freeze(ROOT/'experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json',art)
    if after!=freeze: raise ValueError('Frozen artifacts changed')
    result={'experiment_id':EXP,'status':'completed_smoke','seed':cfg['seed'],'config_sha256':cfg['_sha256'],
        'prediction_seal_sha256':sha256(out/'prediction_seal.json'),'freeze_before':freeze,'freeze_after':after,
        'folds':foldrows,'check_quality_scored_this_run':False,'developer_quality_scored_this_run':False}
    if not args.smoke:
        if any(set(predictions[a])!=set(allnames) for a in ARMS): raise ValueError('Incomplete predictions')
        partition=partition_training_names(allnames,expected_count=219)
        cal=evaluate(partition['calibration'],predictions,baseline,labels,cfg)
        cal['checks']['rgb_exact']=True;cal['passed']=all(cal['checks'].values())
        write_json(out/'calibration_sealed.json',cal)
        result['calibration']=cal;result['calibration_sha256']=sha256(out/'calibration_sealed.json')
        result['status']='completed_calibration_gate_failed'
        if cal['passed']:
            check=evaluate(partition['check'],predictions,baseline,labels,cfg)
            check['checks']['rgb_exact']=True;check['passed']=all(check['checks'].values())
            write_json(out/'check_sealed.json',check);result['check']=check
            result['check_sha256']=sha256(out/'check_sealed.json');result['check_quality_scored_this_run']=True
            result['status']='eligible_for_separate_final' if check['passed'] else 'completed_check_gate_failed'
        if not set(cfg['visual_samples']).issubset(partition['calibration']): raise ValueError('Panel outside calibration')
        result['panels']=panels(cfg['visual_samples'],predictions,baseline,rgbpaths,labels,out/'panels')
    result['seconds']=time.perf_counter()-started
    result['environment']={'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}
    write_json(out/'result.json',result)
    log(stage='completed',status=result['status'],seconds=result['seconds'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'experiments/runs/SEM-270-FOUNDATION-BINARY-OOF/config.yaml')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    try: run(args,out,started)
    except Exception as exc:
        write_json(out/'failure.json',{'experiment_id':EXP,'error':repr(exc),'seconds':time.perf_counter()-started,'status':'incomplete_not_quality_result'})
        raise


if __name__=='__main__': main()
