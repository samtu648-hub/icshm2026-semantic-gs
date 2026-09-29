#!/usr/bin/env python3
"""Independent SEM349 completed-artifact audit; CPU only, no target labels."""
import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.quality_mask2former_final import FinalContext, require_approval


def read(p):return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['artifact-root','batch','approval','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.batch=a.batch.resolve();a.output=a.output.resolve()
    if not a.output.is_relative_to(a.batch):p.error('Keep small audit evidence within the approved batch budget')
    a.output.mkdir(exist_ok=False)
    r=dict(experiment_id='SEM-349-MASK2FORMER-FINAL',status='started_cpu_final_audit',
           target_labels_decoded=False,new_inference=False,quality_metrics=None,
           audit_code_sha256=sha256(Path(__file__)))
    try:
        ctx=FinalContext(ROOT,a.artifact_root,a.batch)
        approval_hash=require_approval(a.approval,ctx,a.batch/'preflight/result.json')
        queue=read(a.batch/'queue/result.json');fit=read(a.batch/'fit/result.json')
        assert queue['status']=='completed_final219_fit_and_two74_fresh_replays_no_quality_scoring'
        assert queue['approval_sha256']==approval_hash and queue['config_sha256']==ctx.binding['config_sha256']
        assert [x['stage'] for x in queue['stages']]==['fit','fresh-1','fresh-2']
        for stage in queue['stages']:
            base=a.batch/stage['stage'];record=read(base/'result.json')
            assert sha256(base/'result.json')==stage['result_sha256']
            assert record['binding']==ctx.binding and record['access']==ctx.access
            assert record['approval_sha256']==approval_hash and record['target_labels_decoded'] is False
            verify_hashes(base,record['artifacts'])
            if stage['stage']!='fit':
                assert record['probabilities_exact'] and record['compared_views']==74
                assert record['prediction_views']==ctx.access['prediction']
                assert record['run_result_sha256']==sha256(a.batch/'fit/result.json')
                assert not (base/'probabilities').exists() and not (base/'predictions').exists()
        assert fit['trained_steps']==6000 and fit['parameters']==215451514
        pre=read(a.batch/'preflight/result.json');plan=read(a.batch/'preflight/sampling.json')
        logs=[json.loads(line) for line in (a.batch/'fit/training.jsonl').read_text().splitlines()]
        assert len(logs)==6000 and fit['class_counts']==pre['class_counts']
        assert fit['training_label_views']==ctx.access['train'] and fit['sampling_sha256']==pre['sampling_sha256']
        skips=0;last_elapsed=-1.;seen=set()
        for i,row in enumerate(logs):
            assert row['step']==i+1 and row['sample']==plan[i]
            assert row['sample']['name'] in ctx.access['train'] and row['sample']['name'] not in ctx.access['prediction']
            assert abs(row['lr_factor']-(1-i/6000)**.9)<1e-15
            assert all(math.isfinite(row[k]) for k in ['loss','grad_norm','scale','elapsed_seconds'])
            assert all(math.isfinite(v) for v in row['losses'].values())
            assert row['elapsed_seconds']>=last_elapsed and row['scale']>0
            last_elapsed=row['elapsed_seconds'];seen.add(row['sample']['name'])
            for event in row['skips']:
                assert event['scale_after']<event['scale_before'] and event['bad_gradient_count']>0
                skips+=1
        assert skips==fit['amp_skips'] and len(seen)==219
        import torch
        torch.set_num_threads(4)
        checkpoints={}
        for steps,name in [(3000,'model-3000.pt'),(6000,'model.pt')]:
            file=a.batch/'fit'/name;payload=torch.load(file,map_location='cpu',weights_only=True)
            assert payload['steps']==steps and payload['binding']==ctx.binding and payload['access']==ctx.access
            assert all(torch.isfinite(t).all().item() for t in payload['model'].values())
            assert fit['checkpoints'][str(steps)]['sha256']==sha256(file)
            assert fit['checkpoints'][str(steps)]['rng_preserved'] is True
            checkpoints[name]={'sha256':sha256(file),'tensor_count':len(payload['model']),'all_finite':True}
            del payload
        names=ctx.access['prediction'];assert fit['prediction_views']==names
        assert sorted(x.name for x in (a.batch/'fit/predictions').glob('*.png'))==names
        assert sorted(x.stem+'.png' for x in (a.batch/'fit/probabilities').glob('*.npz'))==names
        per=[]
        for name in names:
            with np.load(a.batch/'fit/probabilities'/(Path(name).stem+'.npz'),allow_pickle=False) as z:
                assert z.files==['probabilities'];prob=z['probabilities']
            with Image.open(a.batch/'fit/predictions'/name) as im:label=np.array(im)
            assert prob.shape==(5,989,1320) and prob.dtype==np.float32 and np.isfinite(prob).all()
            assert prob.min()>=0 and prob.max()<=1 and np.allclose(prob.sum(0),1,atol=2e-6,rtol=0)
            assert label.dtype==np.uint8 and np.array_equal(prob.argmax(0),label)
            per.append({'name':name,'predicted_class_counts':np.bincount(label.ravel(),minlength=5).tolist()})
        ctx.finish();write_new_json(a.output/'prediction_inventory.json',per)
        r.update(status='passed_independent_final219_artifact_audit_no_quality_scoring',
                 config_sha256=ctx.binding['config_sha256'],approval_sha256=approval_hash,
                 queue_sha256=sha256(a.batch/'queue/result.json'),fit_sha256=sha256(a.batch/'fit/result.json'),
                 trained_steps=6000,training_views_sampled=219,amp_skips=skips,checkpoints=checkpoints,
                 prediction_views=74,fresh_replays=2,rgb_exports_verified=293,frozen_verified=18,
                 inventory_sha256=sha256(a.output/'prediction_inventory.json'),batch_bytes=ctx.guard())
    except BaseException as exc:
        r.update(status='failed_independent_final_artifact_audit',error=f'{type(exc).__name__}: {exc}');raise
    finally:write_new_json(a.output/'result.json',r)


if __name__=='__main__':main()
