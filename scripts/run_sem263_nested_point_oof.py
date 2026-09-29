#!/usr/bin/env python3
"""Strict inner-teacher cross-fitting versus matched in-sample point refinement."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
import run_sem261_point_refinement as point
from run_sem005_segformer import load_model as five_model
from src.semantic_cable_binary import load_binary_model, normalize_rgb, random_native_pair
from src.semantic_nested_oof import (nested_plan, validate_teacher_metadata,
                                      validate_inner, common_sampling_base)
from src.semantic_refinement import point_head, fuse_residual, mutable_lane
from src.semantic2d import read_json, write_json, sha256
from src.semantic_protected_fusion import partition_training_names

ARMS = ('insample_control', 'nested_candidate')
EXP = 'SEM-263-NESTED-POINT-OOF'


def digest_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def log(**value):
    print(json.dumps(value), flush=True)


def resource_gate(out, config, started):
    if shutil.disk_usage(out).free < config['nested']['minimum_free_disk_gib'] * 2**30:
        raise RuntimeError('Disk safety floor reached; no artifact deletion')
    if time.perf_counter()-started > config['maximum_gpu_hours']*3600:
        raise RuntimeError('Registered runtime budget reached')


def initial_model(kind, modeldir, seed):
    return (five_model if kind == 'five' else load_binary_model)(modeldir, seed)


def teacher(out, art, cfg, outer, inner, kind, rgbpaths, labels, smoke, started):
    """Fresh pre-contest initialization; no outer/final learned weights loaded."""
    folder = out/f"outer{outer['fold']}_inner{inner['inner_fold']}_{kind}"
    folder.mkdir()
    names = inner['train'][:2] if smoke else inner['train']
    seed = cfg['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    modeldir = art/cfg['parents']['model_config']
    # CPU caching changes no pixels/augmentation; it removes repeated PNG decode.
    cache, access = {}, []
    classes = 5 if kind == 'five' else 2
    counts = np.zeros(classes, np.int64)
    for name in names:
        if name not in inner['train'] or name in outer['oof'] or name in outer['guard']:
            raise ValueError('Forbidden inner fitting access')
        rgb, gt = point.png(rgbpaths[name][0]), point.png(labels/name)
        target = gt if kind == 'five' else (gt == (1 if kind == 'deck' else 2)).astype(np.uint8)
        cache[name] = (rgb, target)
        counts += np.bincount(target.ravel(), minlength=classes)
        access.append({'name': name, 'label_sha256': sha256(labels/name),
                       'rendered_rgb_sha256': rgbpaths[name][1]})
    if np.any(counts <= 0):
        raise ValueError(f'Missing inner train class: {counts.tolist()}')
    weights = np.clip(np.median(counts.astype(np.float64))/counts, .05, 20).astype(np.float32)
    model = initial_model(kind, modeldir, seed).cuda().train()
    weight = torch.from_numpy(weights).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['nested']['teacher_learning_rate'],
                           weight_decay=cfg['nested']['teacher_weight_decay'])
    rng = random.Random(seed + (outer['fold']*2 + inner['inner_fold'])*100003)
    crop = cfg['nested']['teacher_crop'][kind]
    steps = 2 if smoke else cfg['nested']['teacher_iterations']
    losses, observed = [], set()
    begin = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, steps+1):
        name = names[rng.randrange(len(names))]
        observed.add(name)
        image, target = random_native_pair(*cache[name], crop, rng)
        x = normalize_rgb(image).unsqueeze(0).cuda()
        y = torch.from_numpy(target.astype(np.int64)).unsqueeze(0).cuda()
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(pixel_values=x).logits
            logits = F.interpolate(logits, (crop,crop), mode='bilinear', align_corners=False)
            loss = F.cross_entropy(logits.float(), y, weight=weight)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite inner-teacher loss')
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if step == 1 or step % 250 == 0 or step == steps:
            log(stage='inner_teacher', outer=outer['fold'], inner=inner['inner_fold'],
                kind=kind, step=step, total=steps, loss=losses[-1], seconds=time.perf_counter()-begin)
            resource_gate(out, cfg, started)
    torch.cuda.synchronize()
    seconds = time.perf_counter()-begin
    peak = torch.cuda.max_memory_allocated()/2**20
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        replay = model(pixel_values=x).logits.detach().cpu()
    metadata = {'experiment_id': EXP, 'outer_fold': outer['fold'], 'inner_fold': inner['inner_fold'],
                'kind': kind, 'seed': seed, 'iterations': steps, 'train': inner['train'],
                'oof': inner['oof'], 'guard': inner['guard'], 'actual_label_access': access,
                'actual_optimization_names': sorted(observed), 'config_sha256': cfg['_sha256'],
                'nested_plan_sha256': cfg['_plan_sha256'], 'mode': 'smoke' if smoke else 'full',
                'initialization': cfg['_initialization_files'],
                'class_counts': counts.tolist(), 'class_weights': weights.tolist()}
    validate_teacher_metadata(metadata, outer, inner, kind)
    path = folder/'teacher.pth'
    torch.save({'metadata': metadata, 'model_state_dict': model.state_dict()}, path)
    del model, opt, cache, logits, loss
    torch.cuda.empty_cache()
    payload = torch.load(path, map_location='cpu', weights_only=True)
    fresh = initial_model(kind, modeldir, seed).cuda().eval().requires_grad_(False)
    fresh.load_state_dict(payload['model_state_dict'], strict=True)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        actual = fresh(pixel_values=x).logits.cpu()
    if payload['metadata'] != metadata or not torch.equal(actual, replay):
        raise ValueError('Inner checkpoint reload changed predictions')
    result = {'path': str(path), 'sha256': sha256(path), 'metadata': metadata,
              'seconds': seconds, 'peak_cuda_allocated_mib': peak, 'reload_exact': True,
              'loss': {'first': losses[0], 'last': losses[-1], 'last100': float(np.mean(losses[-100:]))}}
    write_json(folder/'result.json', result)
    del fresh, payload, x, y
    torch.cuda.empty_cache()
    return result


def load_inner(records, art, cfg, outer, inner):
    models = {}
    for kind, record in records.items():
        path = Path(record['path'])
        if sha256(path) != record['sha256']:
            raise ValueError('Inner checkpoint changed')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        validate_teacher_metadata(payload['metadata'], outer, inner, kind)
        model = initial_model(kind, art/cfg['parents']['model_config'], cfg['seed'])
        model.load_state_dict(payload['model_state_dict'], strict=True)
        models[kind] = model.cuda().eval().requires_grad_(False)
    return models


@torch.no_grad()
def feature_cache(models, names, selects, rgbpaths, staticpaths, outer, inner=None):
    result, audit = {}, []
    for index, name in enumerate(names):
        if name not in outer['train'] or (inner is not None and name not in inner['oof']):
            raise ValueError('Wrong teacher feature role')
        rgb, static = point.png(rgbpaths[name][0]), point.png(staticpaths[name][0])
        rgbhash = digest_array(rgb)
        fine, p = point.extract(models['deck'], rgb)
        cable = point.sliding(models['cable'],rgb,crop=960,stride=480,classes=2)[1].astype(np.float16).astype(np.float32)
        five = point.sliding(models['five'],rgb,crop=768,stride=512,classes=5).astype(np.float16).astype(np.float32)
        five /= np.maximum(five.sum(0,keepdims=True),np.finfo(np.float32).tiny)
        base = point.fuse(static,five,cable,p.astype(np.float16).astype(np.float32))['SEM-161']
        select = selects[name]
        features = point.point_inputs(fine,rgb,p,base,static,select).half().cpu()
        if digest_array(rgb) != rgbhash or any(p.requires_grad or p.grad is not None for m in models.values() for p in m.parameters()):
            raise ValueError('Feature extraction mutated RGB/parent gradients')
        result[name] = features
        audit.append({'name':name, 'feature_sha256':digest_array(features.numpy()),
                      'base_sha256':digest_array(base), 'rgb_exact':True,
                      'sample_base_deck':int((base.ravel()[select]==1).sum()),
                      'sample_protected':int((~mutable_lane(base,static)).ravel()[select].sum())})
        del fine
        if index % 20 == 0 or index+1 == len(names):
            log(stage='features', outer=outer['fold'], inner=None if inner is None else inner['inner_fold'],
                views=index+1,total=len(names))
    return result, audit


def fit_heads(cache, targets, names, cfg, outer, out, smoke, provenance):
    torch.manual_seed(cfg['seed'])
    control = point_head().cuda()
    heads = {ARMS[0]:control, ARMS[1]:copy.deepcopy(control)}
    rng = random.Random(cfg['seed']+outer['fold']*100003)
    np_rng = np.random.default_rng(cfg['seed']+outer['fold']*100003)
    positive = sum(int(y.sum()) for y in targets.values())
    negative = sum(len(y) for y in targets.values())-positive
    if min(positive,negative) <= 0:
        raise ValueError('Missing sampled target class')
    weight = torch.tensor(negative/positive, device='cuda')
    opts = {a:torch.optim.AdamW(h.parameters(),lr=cfg['training']['learning_rate'],
                              weight_decay=cfg['training']['weight_decay']) for a,h in heads.items()}
    losses = {a:[] for a in ARMS}
    sample = cache[ARMS[0]][names[0]][:128].float().cuda()
    if any(torch.count_nonzero(h(sample)).item() for h in heads.values()):
        raise ValueError('Nonzero initial residual')
    steps = 8 if smoke else cfg['training']['iterations_per_fold']
    begin = time.perf_counter()
    for step in range(1,steps+1):
        name = names[rng.randrange(len(names))]
        index = np_rng.integers(len(targets[name]),size=cfg['training']['points_per_step'])
        y = targets[name][index].cuda()
        for arm, head in heads.items():
            x = cache[arm][name][index].float().cuda()
            opts[arm].zero_grad(set_to_none=True)
            score = cfg['training']['baseline_margin']*x[:,69] + head(x)[:,0]
            loss = F.binary_cross_entropy_with_logits(score,y,pos_weight=weight)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite point loss')
            loss.backward()
            opts[arm].step()
            losses[arm].append(float(loss.detach()))
        if step == 1 or step%500 == 0 or step == steps:
            log(stage='heads',outer=outer['fold'],step=step,loss={a:v[-1] for a,v in losses.items()})
    torch.cuda.synchronize()
    seconds = time.perf_counter()-begin
    checkpoints = {}
    for arm, head in heads.items():
        metadata = {'experiment_id':EXP,'fold':outer['fold'],'seed':cfg['seed'],
                    'config_sha256':cfg['_sha256'],'nested_plan_sha256':cfg['_plan_sha256'],
                    'iterations':steps,'train':names,'oof':outer['oof'],'guard':outer['guard'],
                    'positive_loss_weight':float(weight),'shared_positive_count':positive,
                    'shared_negative_count':negative,'provenance':provenance}
        path = out/f"fold{outer['fold']}_{arm}.pth"
        torch.save({'metadata':metadata,'arm':arm,'state_dict':head.state_dict()},path)
        payload = torch.load(path,map_location='cpu',weights_only=True)
        fresh = point_head().cuda().eval()
        fresh.load_state_dict(payload['state_dict'],strict=True)
        if payload['metadata'] != metadata or not torch.equal(fresh(sample),head(sample)):
            raise ValueError('Point checkpoint reload mismatch')
        heads[arm] = fresh
        checkpoints[arm] = {'path':str(path),'sha256':sha256(path),'reload_exact':True}
    return heads, {'checkpoints':checkpoints,'train_seconds':seconds,'positive_loss_weight':float(weight),
                   'positive_points':positive,'negative_points':negative,
                   'loss':{a:{'first':v[0],'last':v[-1],'last100':float(np.mean(v[-100:]))} for a,v in losses.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'experiments/runs/SEM-263-NESTED-POINT-OOF/config.yaml')
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter()
    try:
        run(args,out,started)
    except Exception as exc:
        write_json(out/'failure.json',{'experiment_id':EXP,'error':repr(exc),
                   'seconds':time.perf_counter()-started,'status':'incomplete_not_a_quality_result'})
        raise


def run(args,out,started):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Visible bf16-capable CUDA required')
    torch.set_num_threads(4)
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    cfg = read_json(args.config)
    if cfg['experiment_id'] != EXP or tuple(cfg['arm_names']) != ARMS:
        raise ValueError('Wrong registered configuration')
    cfg['_sha256'] = sha256(args.config)
    art = args.artifact_root.resolve()
    source_paths = [Path(__file__),ROOT/'src/semantic_nested_oof.py',ROOT/'src/semantic_refinement.py',
                    ROOT/'scripts/run_sem261_point_refinement.py',ROOT/'scripts/create_sem005_oof_split.py',
                    ROOT/'src/semantic_cable_binary.py',ROOT/'src/submission_sem161.py',
                    ROOT/'scripts/run_sem005_segformer.py',ROOT/'src/semantic2d.py']
    sources = {str(p.relative_to(ROOT)):sha256(p) for p in source_paths}
    (out/'source_snapshot').mkdir()
    for path in (*source_paths,args.config):
        shutil.copyfile(path,out/'source_snapshot'/path.name)
    freeze = point.verify_freeze(ROOT/'experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json',art)
    write_json(out/'freeze_before.json',freeze)
    for key in ('semantic_split','oof_split'):
        if sha256(ROOT/cfg[key]) != cfg[key+'_sha256']:
            raise ValueError('Split changed')
    split = read_json(ROOT/cfg['oof_split'])
    folds = split['folds']
    allnames = read_json(ROOT/cfg['semantic_split'])['semantic_train_labeled']
    graph = read_json(ROOT/cfg['camera_graph'])
    if sha256(ROOT/cfg['camera_graph']) != split['camera_graph_sha256']:
        raise ValueError('Official camera graph changed')
    centers = {n['name']:np.asarray(n['center']) for n in graph['nodes']}
    plan = nested_plan(folds,centers,cfg['seed'],split['guard_policy']['radius'])
    write_json(out/'nested_plan.json',plan)
    cfg['_plan_sha256'] = sha256(out/'nested_plan.json')
    modeldir = art/cfg['parents']['model_config']
    cfg['_initialization_files'] = [{'path':str(p.relative_to(art)), 'sha256':sha256(p)}
        for p in sorted(modeldir.iterdir()) if p.suffix in ('.bin','.safetensors','.json')]
    if not any(Path(p['path']).suffix in ('.bin','.safetensors') for p in cfg['_initialization_files']):
        raise ValueError('No original pretrained weights')
    baseline = point.manifest_paths(art/cfg['parents']['sem160'],'prediction_manifest.json','predictions','prediction','prediction_sha256')
    staticpaths = point.manifest_paths(art/cfg['parents']['sem001'],'training_sem001_render_manifest.json','rendered','semantic_png','sha256')
    rgbpaths = point.manifest_paths(art/cfg['parents']['rgb'],'rendered_dataset_manifest.json','rendered_images','path','sha256')
    labels = art/'r2/sem_001/labels'
    for mapping in (baseline,staticpaths,rgbpaths):
        if set(mapping) != set(allnames):
            raise ValueError('Parent universe mismatch')
        if any(sha256(p) != h for p,h in mapping.values()):
            raise ValueError('Parent image hash mismatch')
    records, predictions = [], {a:{} for a in ARMS}
    for outer, nest in zip(folds[:1] if args.smoke else folds,plan):
        fi = outer['fold']
        foldstart = time.perf_counter()
        train, held, guard = (set(outer[k]) for k in ('train','oof','guard'))
        if train&held or train&guard or held&guard or train|held|guard != set(allnames):
            raise ValueError('Outer split leakage')
        resource_gate(out,cfg,started)
        nested_records = []
        for inner in nest['inners'][:1] if args.smoke else nest['inners']:
            validate_inner(outer,inner,centers,split['guard_policy']['radius'])
            teachers = {kind:teacher(out,art,cfg,outer,inner,kind,rgbpaths,labels,args.smoke,started)
                        for kind in ('deck','cable','five')}
            nested_records.append({'inner':inner,'teachers':teachers})
        names = nested_records[0]['inner']['oof'][:2] if args.smoke else sorted(train)
        rng = np.random.default_rng(cfg['seed']+fi*100003)
        selects, targets, access = {}, {}, []
        for name in names:
            static = point.png(staticpaths[name][0])
            frame = common_sampling_base(static)
            select = point.sample_indices(frame,frame,rng,cfg['training'])
            # These labels supervise the OUTER head, never an inner teacher
            # generating this image's features.
            gt = point.png(labels/name)
            target = torch.from_numpy((gt.ravel()[select]==1).astype(np.float32))
            selects[name],targets[name] = select,target
            access.append({'name':name,'label_sha256':sha256(labels/name),
                           'sample_indices_sha256':digest_array(select),'targets_sha256':digest_array(target.numpy())})
        cache, feature_records = {a:{} for a in ARMS}, {a:[] for a in ARMS}
        feature_start = time.perf_counter()
        for record in nested_records:
            inner = record['inner']
            chosen = [n for n in names if n in inner['oof']]
            models = load_inner(record['teachers'],art,cfg,outer,inner)
            part, audit = feature_cache(models,chosen,selects,rgbpaths,staticpaths,outer,inner)
            cache[ARMS[1]].update(part)
            feature_records[ARMS[1]].extend(audit)
            del models
            torch.cuda.empty_cache()
        parents, parent_records = {}, {}
        for kind in ('deck','cable','five'):
            parents[kind],parent_records[kind] = point.load_parent(art,cfg,fi,kind)
            if parent_records[kind]['metadata']['oof_split_sha256'] != cfg['oof_split_sha256']:
                raise ValueError('Original teacher was fitted on a different outer split')
        cache[ARMS[0]], feature_records[ARMS[0]] = feature_cache(parents,names,selects,rgbpaths,staticpaths,outer)
        if any(set(v) != set(names) for v in cache.values()):
            raise ValueError('Incomplete paired feature cache')
        cachepath = out/f'fold{fi}_paired_features.pth'
        torch.save({'features':cache,'targets':targets,
                    'sample_indices':{n:torch.from_numpy(s) for n,s in selects.items()},
                    'metadata':{'config_sha256':cfg['_sha256'],'access':access}},cachepath)
        feature_seconds = time.perf_counter()-feature_start
        provenance = {'outer_parents':parent_records,'inner_teachers':nested_records,
                      'train_label_access':access,'paired_cache_sha256':sha256(cachepath)}
        heads, fitting = fit_heads(cache,targets,names,cfg,outer,out,args.smoke,provenance)
        # Training-side descriptive error shift, not held-out quality selection.
        trainstats = {}
        for arm in ARMS:
            sign = torch.cat([cache[arm][n][:,69] for n in names]) > 0
            y = torch.cat([targets[n] for n in names]) > 0
            trainstats[arm] = {'tp':int((sign&y).sum()),'fp':int((sign&~y).sum()),
                              'fn':int((~sign&y).sum()),'tn':int((~sign&~y).sum())}
        del cache, targets, selects
        torch.cuda.reset_peak_memory_stats()
        inferstart = time.perf_counter()
        targetnames = sorted(held)[:1] if args.smoke else sorted(held)
        for index,name in enumerate(targetnames):
            rgb,static,base = [point.png(p) for p in (rgbpaths[name][0],staticpaths[name][0],baseline[name][0])]
            rgbhash = digest_array(rgb)
            fine,p = point.extract(parents['deck'],rgb)
            if index == 0:
                difference = float(np.abs(p-point.sliding(parents['deck'],rgb,crop=960,stride=480,classes=2)[1]).max())
                if difference != 0:
                    raise ValueError('Coarse probability changed')
            if not np.array_equal(fuse_residual(base,static,np.zeros(base.shape,np.float32)),base):
                raise ValueError('Zero residual mismatch')
            new = point.infer_heads(heads,fine,rgb,p,base,static,cfg)
            if digest_array(rgb) != rgbhash:
                raise ValueError('Inference RGB changed')
            for arm, value in new.items():
                if not np.array_equal(value[~mutable_lane(base,static)],base[~mutable_lane(base,static)]):
                    raise ValueError('Protected pixels changed')
                path = out/arm/name
                path.parent.mkdir(exist_ok=True)
                Image.fromarray(value).save(path)
                predictions[arm][name] = {'path':str(path),'sha256':sha256(path),'fold':fi}
            del fine
            if index%20 == 0 or index+1 == len(targetnames):
                log(stage='outer_inference',outer=fi,views=index+1,total=len(targetnames))
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter()-inferstart
        for kind,model in parents.items():
            state = torch.load(art/parent_records[kind]['path'],map_location='cpu',weights_only=True)['model_state_dict']
            if any(not torch.equal(v.detach().cpu(),state[k]) for k,v in model.state_dict().items()):
                raise ValueError('Original parent tensor changed')
        row = {'fold':fi,**fitting,**provenance,'feature_records':feature_records,
               'sampled_base_confusions_training_only':trainstats,'oof_names':targetnames,
               'zero_residual_exact_images':len(targetnames),'protected_exact':True,'rgb_arrays_exact':True,
               'frozen_parent_tensors_exact':True,'coarse_probability_max_difference':difference,
               'feature_seconds':feature_seconds,'inference_seconds':inference_seconds,
               'fold_seconds':time.perf_counter()-foldstart,
               'peak_cuda_inference_mib':torch.cuda.max_memory_allocated()/2**20}
        records.append(row)
        write_json(out/f'fold{fi}_result.json',row)
        del parents, heads, state
        torch.cuda.empty_cache()
    write_json(out/'prediction_manifest.json',predictions)
    seal = {'experiment_id':EXP,'config_sha256':cfg['_sha256'],'nested_plan_sha256':cfg['_plan_sha256'],
            'source_sha256':sources,'prediction_manifest_sha256':sha256(out/'prediction_manifest.json'),
            'repository_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'scope':'Nested 2D teacher training evidence; inherited SEM-001 all219, incremental OOF; reused development partitions',
            'folds':records}
    write_json(out/'prediction_seal.json',seal)
    result = {'experiment_id':EXP,'seed':cfg['seed'],'folds':records,'freeze_before':freeze,
              'config_sha256':cfg['_sha256'],'seal_sha256':sha256(out/'prediction_seal.json'),
              'check_target_labels_read_this_run':[],'developer_target_labels_read_this_run':[],
              'status':'completed_smoke' if args.smoke else 'sealed_oof'}
    # Fail engineering/provenance checks BEFORE opening calibration target metrics.
    if sha256(args.config) != cfg['_sha256'] or any(sha256(ROOT/p) != h for p,h in sources.items()):
        raise ValueError('Protocol/source changed during execution')
    result['freeze_after'] = point.verify_freeze(ROOT/'experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json',art)
    if result['freeze_after'] != freeze:
        raise ValueError('Frozen artifacts changed')
    if not args.smoke:
        if any(set(predictions[a]) != set(allnames) for a in ARMS):
            raise ValueError('OOF predictions incomplete')
        partition = partition_training_names(allnames,expected_count=219)
        point.ARMS = ARMS
        calibration = point.evaluate_partition(partition['calibration'],predictions,baseline,staticpaths,labels,cfg)
        calibration['checks']['rgb_exact'] = all(r['rgb_arrays_exact'] for r in records)
        calibration['passed'] = all(calibration['checks'].values())
        write_json(out/'calibration_sealed.json',calibration)
        result['calibration'],result['calibration_sha256'] = calibration,sha256(out/'calibration_sealed.json')
        result['status'] = 'calibration_passed' if calibration['passed'] else 'completed_calibration_gate_failed'
        if calibration['passed']:
            check = point.evaluate_partition(partition['check'],predictions,baseline,staticpaths,labels,cfg)
            result['check'],result['check_target_labels_read_this_run'] = check,partition['check']
            result['status'] = 'qualified_for_final' if check['passed'] else 'completed_check_gate_failed'
        names = [n for n in cfg['visual_samples'] if n in partition['calibration']]
        result['panels'] = point.comparison_panels(names,predictions,baseline,rgbpaths,labels,out/'panels')
    result['seconds'] = time.perf_counter()-started
    result['environment'] = {'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
                             'python':sys.version,'head_parameters_per_arm':sum(p.numel() for p in point_head().parameters())}
    write_json(out/'result.json',result)
    log(status=result['status'],seconds=result['seconds'],checks=result.get('calibration',{}).get('checks'),
        result_sha256=sha256(out/'result.json'))


if __name__ == '__main__':
    main()
