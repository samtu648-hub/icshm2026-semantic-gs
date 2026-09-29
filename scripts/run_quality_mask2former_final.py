#!/usr/bin/env python3
"""SEM349 CPU readiness, or explicitly approved one final fit / fresh replay."""
import argparse
import fcntl
import json
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.quality_mask2former_final import FinalContext, CONFIG, sampling_plan, sampling_digest, require_approval
from src.quality_protocol import sha256, verify_hashes, write_new_json


def preflight(ctx, out, rec):
    labels=ctx.labels(); counts=labels.audit_counts()
    if not (counts>0).all() or sorted(labels.accessed)!=ctx.access['train']:
        raise ValueError('Training-only class count audit incomplete')
    rows=sampling_plan(ctx.access['train'],ctx.cfg['seed'])
    write_new_json(out/'sampling.json',rows)
    rec.update(status='ready_cpu_roles_sampling_budget_no_gpu_execution',class_counts=counts.tolist(),
               class_weights='author set loss/no-object weights; no inverse-frequency CE',
               training_label_views=sorted(labels.accessed),sampling_sha256=sampling_digest(rows),
               sampled_training_views=len({x['name'] for x in rows}),
               input_rgb_hashes_verified=293, frozen_artifacts_verified=18,
               projected_new_bytes_upper_bound=4*1024**3)
    ctx.guard(pending_bytes=4*1024**3-ctx.guard())


def execute(a, ctx, rec, ready):
    import numpy as np
    import torch
    from PIL import Image
    from src.semantic_cable_binary import read_rgb
    from src.quality_mask2former import load_official_model, native_predict, author_optimizer, class_masks
    from src.quality_mask2former_training import amp_update, poly_factor, sampled_crop, save_checkpoint
    from src.quality_semantic_audit import check_replay
    rec['runtime']=ctx.setup_runtime()
    cfg=ctx.model_config()
    model, loaded=load_official_model(cfg,ctx.runtime/'weights/model_final_6b4a3a.pkl')
    expected=ctx.access['prediction']; start=time.monotonic()
    if a.mode=='train':
        from detectron2.structures import Instances
        rec.update(pretrained_load=loaded,parameters=sum(v.numel() for v in model.parameters()))
        if rec['parameters']!=215451514 or sum(v.numel() for v in model.parameters() if v.requires_grad)!=rec['parameters']:
            raise ValueError('Registered full Swin-L required')
        (a.output/'resolved_model.yaml').write_text(cfg.dump())
        model.cuda(); optimizer=author_optimizer(model,cfg); rates=[g['lr'] for g in optimizer.param_groups]
        t=ctx.cfg['training']; scaler=torch.amp.GradScaler('cuda',init_scale=t['amp_initial_scale'])
        labels=ctx.labels(); counts=labels.audit_counts()
        if counts.tolist()!=ready['class_counts']:raise ValueError('Training count audit changed')
        names=ctx.access['train']; images={n:read_rgb(ctx.export_root/'images'/n) for n in names}
        targets={n:labels.read(n) for n in names}; plan=sampling_plan(names,ctx.cfg['seed'])
        if sampling_digest(plan)!=ready['sampling_sha256']:raise ValueError('Sampling plan changed')
        rng=random.Random(ctx.cfg['seed']); model.train(); torch.cuda.reset_peak_memory_stats()
        rec.update(class_counts=counts.tolist(),training_label_views=sorted(labels.accessed),amp_skips=0,checkpoints={},
                   class_weights=ready['class_weights'],sampling_sha256=ready['sampling_sha256'])
        with (a.output/'training.jsonl').open('x') as stream:
            for step in range(1,6001):
                if time.monotonic()-start>3600:raise TimeoutError('One-hour train-stage limit')
                x,y,trace=sampled_crop(rng,names,images,targets,768)
                if trace!=plan[step-1]:raise ValueError('Actual sampler differs from CPU plan')
                ids,masks=class_masks(y); inst=Instances((768,768))
                inst.gt_classes=torch.from_numpy(ids);inst.gt_masks=torch.from_numpy(masks)
                image=torch.from_numpy(x.transpose(2,0,1).copy()).float()
                factor=poly_factor(step-1)
                for g,rate in zip(optimizer.param_groups,rates):g['lr']=rate*factor
                row=amp_update(model,optimizer,scaler,image,inst,max_attempts=t['amp_max_attempts_per_step'],clip=t['clip_global_norm'])
                row.update(step=step,sample=trace,lr_factor=factor,elapsed_seconds=time.monotonic()-start)
                stream.write(json.dumps(row)+'\n');rec['trained_steps']=step;rec['amp_skips']+=len(row['skips'])
                if step in (3000,6000):
                    ctx.guard(pending_bytes=900*1024**2)
                    rec['checkpoints'][str(step)]=save_checkpoint(model,ctx,step,a.output/('model.pt' if step==6000 else 'model-3000.pt'))
                if step%100==0 or step==1:
                    ctx.guard();stream.flush();print(json.dumps({'step':step,'amp_skips':rec['amp_skips']}),flush=True)
        rec.update(training_seconds=time.monotonic()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
        optimizer.zero_grad(set_to_none=True);del optimizer,scaler,images,targets
        torch.cuda.empty_cache()
        (a.output/'probabilities').mkdir();(a.output/'predictions').mkdir()
    else:
        source=json.loads((a.run/'result.json').read_text())
        if (source['status']!='completed_final219_fit_pending_two_replays' or source['binding']!=ctx.binding
                or source['access']!=ctx.access or source['runtime']!=rec['runtime'] or source['trained_steps']!=6000
                or source['prediction_views']!=expected):raise ValueError('Invalid final replay source')
        verify_hashes(a.run,source['artifacts'])
        payload=torch.load(a.run/'model.pt',map_location='cpu',weights_only=True)
        if payload['binding']!=ctx.binding or payload['access']!=ctx.access or payload['steps']!=6000:
            raise ValueError('Final checkpoint provenance mismatch')
        model.load_state_dict(payload['model'],strict=True);del payload;model.cuda()
        rec.update(run_result_sha256=sha256(a.run/'result.json'),trained_steps=6000)
    rec['prediction_views']=expected;rec['compared_views']=0;infer_start=time.monotonic()
    for name in expected:
        if time.monotonic()-infer_start>900:raise TimeoutError('15-minute inference/replay limit')
        ctx.guard(pending_bytes=27*1024**2 if a.mode=='train' else 0)
        probability=native_predict(model,read_rgb(ctx.prediction_root/'images'/name))
        if a.mode=='train':
            np.savez_compressed(a.output/'probabilities'/(Path(name).stem+'.npz'),probabilities=probability)
            Image.fromarray(probability.argmax(0).astype(np.uint8)).save(a.output/'predictions'/name)
        else:
            with np.load(a.run/'probabilities'/(Path(name).stem+'.npz'),allow_pickle=False) as z:stored=z['probabilities']
            with Image.open(a.run/'predictions'/name) as im:png=np.array(im)
            check_replay(probability,stored,png);rec['compared_views']+=1
    rec.update(status='completed_final219_fit_pending_two_replays' if a.mode=='train' else 'completed_final219_fresh_replay',
               probabilities_exact=True if a.mode=='replay' else None,inference_seconds=time.monotonic()-infer_start)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['artifact-root','batch','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--mode',choices=['preflight','train','replay'],required=True)
    for n in ['preflight','approval','run']:p.add_argument('--'+n,type=Path)
    a=p.parse_args();a.batch=a.batch.resolve();a.output=a.output.resolve()
    if not a.batch.is_relative_to(ROOT/'outputs/quality/mask2former') or not a.output.is_relative_to(a.batch) or a.output==a.batch:
        p.error('Use a new stage directory inside the isolated mask2former batch')
    if a.mode!='preflight' and (a.preflight is None or a.approval is None):p.error('Execution needs readiness and explicit approval receipt')
    if (a.mode=='replay')!=(a.run is not None):p.error('Only replay requires an existing final run')
    if a.mode=='train' and a.output!=a.batch/'fit':p.error('One final fit per approved batch only')
    if a.mode=='replay' and (a.output not in [a.batch/'fresh-1',a.batch/'fresh-2'] or a.run.resolve()!=a.batch/'fit'):
        p.error('Only the two final-fit replay slots are allowed')
    a.output.mkdir(parents=True,exist_ok=False)
    r=dict(experiment_id='SEM-349-MASK2FORMER-FINAL',status='initializing',mode=a.mode,trained_steps=0,
           target_labels_decoded=False,quality_metrics=None,automatic_promotion=False,artifacts={})
    start=time.monotonic();ctx=None
    try:
        ctx=FinalContext(ROOT,a.artifact_root,a.batch)
        r.update(binding=ctx.binding,access=ctx.access,git_commit=ctx.cfg['git_commit'],seed=ctx.cfg['seed'])
        for n in [CONFIG,*ctx.cfg['code_sha256']]:
            dest=a.output/'code'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dest)
        if a.mode=='preflight':preflight(ctx,a.output,r)
        else:
            ready=json.loads(a.preflight.read_text())
            if ready['status']!='ready_cpu_roles_sampling_budget_no_gpu_execution' or ready['binding']!=ctx.binding:
                raise ValueError('Readiness binding changed')
            verify_hashes(a.preflight.parent,ready['artifacts'])
            r['approval_sha256']=require_approval(a.approval,ctx,a.preflight)
            r['preflight_sha256']=sha256(a.preflight)
            with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);execute(a,ctx,r,ready)
        ctx.finish()
        r.update(rgb_exports_verified=293,frozen_artifacts_verified=18,
                 batch_bytes=ctx.guard(),artifacts={str(p.relative_to(a.output)):sha256(p) for p in a.output.rglob('*') if p.is_file()})
    except BaseException as exc:
        r.update(status='failed_final219_stage_preserved',error=f'{type(exc).__name__}: {exc}')
        (a.output/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        r['seconds']=time.monotonic()-start;write_new_json(a.output/'result.json',r)
        print(json.dumps({'status':r['status'],'mode':a.mode}),flush=True)


if __name__=='__main__':main()
