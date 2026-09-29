#!/usr/bin/env python3
"""Matched frozen-backbone point refinement; predictions precede metric reads."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from audit_sem260_error_budget import manifest_paths
from run_sem250_colmap_track_deck_diag import verify_freeze
from src.semantic2d import read_json, write_json, sha256, tile_starts, confusion_metrics
from src.semantic_cable_binary import normalize_rgb
from src.semantic_protected_fusion import (partition_training_names, local_confusion,
                                           metrics_from_confusion)
from src.semantic_refinement import (mutable_lane, fuse_residual, point_head,
                                     mask_fine_features)
from src.semantic_render_domain import semantic_rgb
from src.submission_sem161 import sliding, fuse

ARMS = ('coarse_control', 'fine_candidate')


def png(path):
    return np.asarray(Image.open(path))


def load_parent(art, config, fold, kind):
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
    spec = {'deck': (2, 'deck_binary_final.pth', 'SEM-160-DECK-BINARY-OOF'),
            'cable': (2, 'cable_binary_final.pth', 'SEM-140-CABLE-BINARY-OOF'),
            'five': (5, 'segformer_b2_final.pth', 'SEM-101-RENDERED-SEG-OOF')}
    classes, filename, experiment = spec[kind]
    parentroot = art/config['parents'][kind+'_folds']/f'fold-{fold}'
    path = parentroot/'checkpoints'/filename
    digest = sha256(path)
    if digest != read_json(parentroot/'result.json')['checkpoint_sha256']:
        raise ValueError('Frozen fold checkpoint hash differs')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    metadata = payload['metadata']
    if metadata['experiment_id'] != experiment or int(metadata['fold']) != fold:
        raise ValueError('Wrong fold/experiment parent: no cross-fold teacher allowed')
    specconfig = SegformerConfig.from_pretrained(art/config['parents']['model_config'], local_files_only=True)
    specconfig.num_labels = classes
    model = SegformerForSemanticSegmentation(specconfig)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    model.cuda().eval().requires_grad_(False)
    return model, {'path': str(path.relative_to(art)), 'sha256': digest,
                   'experiment_id': experiment, 'fold': fold,
                   'metadata': metadata}


@torch.no_grad()
def extract(model, rgb, crop=960, stride=480):
    """Keep exactly the binary parent's probability tiling/AMP order.

    Fine stage-1 features are upsampled to native pixels and overlap-averaged.
    No camera/world coordinates or GT maps participate in this operation.
    """
    height, width = rgb.shape[:2]
    padded = np.pad(rgb, ((0,max(0,crop-height)),(0,max(0,crop-width)),(0,0)), mode='edge')
    h, w = padded.shape[:2]
    fine = torch.zeros((64,h,w), device='cuda')
    total = np.zeros((h,w), np.float32)
    count = np.zeros((h,w), np.float32)
    for y in tile_starts(h,crop,stride):
        for x in tile_starts(w,crop,stride):
            tile = normalize_rgb(padded[y:y+crop,x:x+crop]).unsqueeze(0).cuda()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(pixel_values=tile, output_hidden_states=True)
                logits = F.interpolate(output.logits,(crop,crop),mode='bilinear',align_corners=False)
                probability = logits.float().softmax(1)[0,1]
                feature = F.interpolate(output.hidden_states[0].float(),(crop,crop),mode='bilinear',align_corners=False)[0]
            total[y:y+crop,x:x+crop] += probability.cpu().numpy()
            count[y:y+crop,x:x+crop] += 1
            fine[:,y:y+crop,x:x+crop] += feature
    fine /= torch.from_numpy(count).cuda()[None]
    return fine[:,:height,:width], (total/count)[:height,:width]


def point_inputs(fine, rgb, probability, base, static, indices):
    device = fine.device
    idx = torch.as_tensor(indices, dtype=torch.long, device=device)
    # Advanced indexing avoids a full 75-channel native feature allocation.
    feature = fine.flatten(1)[:,idx].T
    colors = normalize_rgb(rgb).flatten(1).T.to(device)[idx]
    p = torch.from_numpy(np.ascontiguousarray(probability.ravel())).to(device)[idx]
    sign = torch.from_numpy(np.where(base.ravel()==1,1.,-1.).astype(np.float32)).to(device)[idx]
    onehot = F.one_hot(torch.from_numpy(static.ravel().astype(np.int64)).to(device)[idx],5).float()
    return torch.cat((feature, colors, (1-p)[:,None],p[:,None],sign[:,None],onehot),1)


def sample_indices(base, static, rng, config):
    lane = mutable_lane(base, static)
    eligible = np.flatnonzero(lane)
    if not len(eligible):
        raise ValueError('No trainable bg/deck points in selected training view')
    radius = config['predicted_boundary_radius_px']
    deck = (base==1).astype(np.uint8)
    kernel = np.ones((2*radius+1,2*radius+1),np.uint8)
    band = (cv2.dilate(deck,kernel) != cv2.erode(deck,kernel)) & lane
    boundary = np.flatnonzero(band)
    n = config['points_cached_per_view']
    nb = int(n*config['boundary_sample_fraction']) if len(boundary) else 0
    return np.concatenate((rng.choice(boundary,nb,replace=True),
                           rng.choice(eligible,n-nb,replace=True)))


@torch.no_grad()
def infer_heads(heads, fine, rgb, p, base, static, config):
    indices = np.flatnonzero(mutable_lane(base,static))
    residuals = {arm: np.zeros(base.size,np.float32) for arm in heads}
    chunk = config['inference']['point_chunk']
    for start in range(0,len(indices),chunk):
        select = indices[start:start+chunk]
        features = point_inputs(fine,rgb,p,base,static,select)
        for arm, head in heads.items():
            feature_arm = 'fine_candidate' if config.get('control_uses_fine_features',False) else arm
            residuals[arm][select] = head(mask_fine_features(features,feature_arm))[:,0].cpu().numpy()
    return {arm: fuse_residual(base,static,delta.reshape(base.shape),
                margin=config['training']['baseline_margin']) for arm,delta in residuals.items()}


def evaluate_partition(names, predictions, base_paths, static_paths, labels, config):
    sums = {arm: np.zeros((5,5),np.int64) for arm in (*ARMS,'baseline')}
    changes = {arm:{'added_deck':0,'removed_deck':0,'corrected':0,'harmed':0} for arm in ARMS}
    access, per_view = [], []
    for name in names:
        gt = png(labels/name)
        old = png(base_paths[name][0])
        static = png(static_paths[name][0])
        sums['baseline'] += local_confusion(gt,old)
        row = {'name':name,'baseline':confusion_metrics(local_confusion(gt,old))}
        for arm in ARMS:
            record = predictions[arm][name]
            path = Path(record['path'])
            if sha256(path) != record['sha256']:
                raise ValueError('Sealed prediction changed')
            new = png(path)
            if not np.array_equal(new[~mutable_lane(old,static)],old[~mutable_lane(old,static)]):
                raise ValueError('Protected pixels changed')
            sums[arm] += local_confusion(gt,new)
            changed = new!=old
            changes[arm]['added_deck'] += int(((old==0)&(new==1)).sum())
            changes[arm]['removed_deck'] += int(((old==1)&(new==0)).sum())
            changes[arm]['corrected'] += int((changed&(new==gt)).sum())
            changes[arm]['harmed'] += int((changed&(old==gt)).sum())
            row[arm] = confusion_metrics(local_confusion(gt,new))
        per_view.append(row)
        access.append({'name':name,'label_sha256':sha256(labels/name)})
    metrics = {arm:metrics_from_confusion(c) for arm,c in sums.items()}
    b,c,k = metrics['baseline'],metrics[ARMS[1]],metrics[ARMS[0]]
    bd,cd,kd = [m['per_class']['deck'] for m in (b,c,k)]
    gate = config['gates']
    checks = {'deck_iou_gain':cd['iou']>=bd['iou']+gate['deck_iou_gain'],
              'deck_recall_gain':cd['recall']>=bd['recall']+gate['deck_recall_gain'],
              'deck_precision':cd['precision']>=gate['deck_precision'],
              'miou_gain':c['mIoU_with_background']>=b['mIoU_with_background']+gate['miou_gain'],
              'candidate_beats_matched_control_iou':cd['iou']>kd['iou'],
              'protected_exact':True}
    return {'metrics':metrics,'changes':changes,'checks':checks,'passed':all(checks.values()),
            'target_access':access,'per_view':per_view}


def comparison_panels(names, paths, basepaths, rgbpaths, labels, out):
    out.mkdir()
    records=[]
    for name in names:
        gt,old,rgb=[png(p) for p in (labels/name,basepaths[name][0],rgbpaths[name][0])]
        pred=[png(Path(paths[a][name]['path'])) for a in ARMS]
        change=np.zeros_like(rgb)
        changed=pred[1]!=old
        change[changed&(pred[1]==gt)]=(40,220,80)
        change[changed&(pred[1]!=gt)]=(255,50,50)
        columns=[rgb,semantic_rgb(gt),semantic_rgb(old),*[semantic_rgb(p) for p in pred],change]
        full=out/(Path(name).stem+'_panel.png')
        Image.fromarray(np.concatenate(columns,1)).save(full)
        ys,xs=np.where((gt==1)|(old==1))
        if len(xs):
            y0,y1=max(0,int(ys.min())-16),min(old.shape[0],int(ys.max())+17)
            x0,x1=max(0,int(xs.min())-16),min(old.shape[1],int(xs.max())+17)
            zoom=out/(Path(name).stem+'_deck_zoom.png')
            Image.fromarray(np.concatenate([a[y0:y1,x0:x1] for a in columns],1)).save(zoom)
            records.append({'path':str(zoom),'sha256':sha256(zoom)})
        records.append({'path':str(full),'sha256':sha256(full)})
    return records


def main():
    global ARMS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--config',type=Path,default=ROOT/'experiments/runs/SEM-261-POINT-REFINEMENT-OOF/config.yaml')
    args=parser.parse_args()
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU substitute')
    torch.set_num_threads(4)
    cfgpath=args.config.resolve()
    config=read_json(cfgpath)
    ARMS=tuple(config.get('arm_names',ARMS))
    if len(ARMS)!=2 or len(set(ARMS))!=2:
        raise ValueError('Exactly two distinct matched arms are required')
    cfgdigest=sha256(cfgpath)
    source_paths=(Path(__file__).resolve(),ROOT/'src/semantic_refinement.py')
    source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in source_paths}
    (out/'source_snapshot').mkdir()
    for path in (*source_paths,cfgpath):
        shutil.copyfile(path,out/'source_snapshot'/path.name)
    art=args.artifact_root
    freeze=verify_freeze(ROOT/'experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json',art)
    for key in ('semantic_split','oof_split'):
        if sha256(ROOT/config[key]) != config[key+'_sha256']:
            raise ValueError('Split changed')
    allnames=read_json(ROOT/config['semantic_split'])['semantic_train_labeled']
    partition=partition_training_names(allnames,expected_count=219)
    folds=read_json(ROOT/config['oof_split'])['folds']
    baseline=manifest_paths(art/config['parents']['sem160'],'prediction_manifest.json','predictions','prediction','prediction_sha256')
    staticpaths=manifest_paths(art/config['parents']['sem001'],'training_sem001_render_manifest.json','rendered','semantic_png','sha256')
    rgbpaths=manifest_paths(art/config['parents']['rgb'],'rendered_dataset_manifest.json','rendered_images','path','sha256')
    labels=art/'r2/sem_001/labels'
    # Verify all label-free parent inputs before fitting; no GT read here.
    for mapping in (baseline,staticpaths,rgbpaths):
        if set(mapping)!=set(allnames):
            raise ValueError('Parent manifest view universe differs')
        for path,digest in mapping.values():
            if sha256(path)!=digest:
                raise ValueError('Parent input changed')
    predictions={a:{} for a in ARMS}
    records=[]
    for fold in (folds[:1] if args.smoke else folds):
        fi=int(fold['fold'])
        train,targets,guard=map(set,(fold['train'],fold['oof'],fold['guard']))
        if train&targets or train&guard or targets&guard or train|targets|guard != set(allnames):
            raise ValueError('Fold leakage')
        trainnames=sorted(train)[:2] if args.smoke else sorted(train)
        targetnames=sorted(targets)[:1] if args.smoke else sorted(targets)
        torch.manual_seed(config['seed'])
        np_rng=np.random.default_rng(config['seed']+fi*100003)
        rng=random.Random(config['seed']+fi*100003)
        parents, parent_records={},{}
        for kind in ('deck','cable','five'):
            parents[kind],parent_records[kind]=load_parent(art,config,fi,kind)
        torch.cuda.reset_peak_memory_stats()
        foldstart=time.perf_counter()
        cache=[]
        train_access=[]
        extraction_start=time.perf_counter()
        for index,name in enumerate(trainnames):
            if name not in train or name in targets or name in guard:
                raise ValueError('Illegal fitting target')
            rgb,static=[png(p) for p in (rgbpaths[name][0],staticpaths[name][0])]
            fine,pdeck=extract(parents['deck'],rgb)
            # Build training decisions from THIS fold's frozen networks, never
            # a cross-fold OOF map. Match stored float16 parent probabilities.
            cable=sliding(parents['cable'],rgb,crop=960,stride=480,classes=2)[1].astype(np.float16).astype(np.float32)
            five=sliding(parents['five'],rgb,crop=768,stride=512,classes=5).astype(np.float16).astype(np.float32)
            five/=np.maximum(five.sum(0,keepdims=True),np.finfo(np.float32).tiny)
            base=fuse(static,five,cable,pdeck.astype(np.float16).astype(np.float32))['SEM-161']
            select=sample_indices(base,static,np_rng,config['training'])
            feat=point_inputs(fine,rgb,pdeck,base,static,select).half().cpu()
            gt=png(labels/name)
            target=torch.from_numpy((gt.ravel()[select]==1).astype(np.float32))
            cache.append((feat,target))
            train_access.append({'name':name,'label_sha256':sha256(labels/name),
                                  'sample_indices_sha256':__import__('hashlib').sha256(select.tobytes()).hexdigest()})
            del fine
            if index%10==0 or index+1==len(trainnames):
                print(json.dumps({'fold':fi,'feature_views':index+1,'total':len(trainnames)}),flush=True)
        feature_seconds=time.perf_counter()-extraction_start
        torch.manual_seed(config['seed'])
        first=point_head().cuda()
        heads={ARMS[0]:first,ARMS[1]:copy.deepcopy(first)}
        sample=cache[0][0][:128].float().cuda()
        feature_arms={a:('fine_candidate' if config.get('control_uses_fine_features',False) else a) for a in ARMS}
        if any(torch.count_nonzero(h(mask_fine_features(sample,feature_arms[a]))).item() for a,h in heads.items()):
            raise ValueError('Zero residual failed')
        positive_points=sum(int(y.sum()) for _,y in cache)
        negative_points=sum(len(y) for _,y in cache)-positive_points
        if min(positive_points,negative_points)<=0:
            raise ValueError('Sampled training points miss a binary class')
        loss_weights={a:1.0 for a in ARMS}
        if config.get('candidate_positive_weight')=='sampled_train_negatives_over_positives':
            loss_weights[ARMS[1]]=negative_points/positive_points
        optimizers={a:torch.optim.AdamW(h.parameters(),lr=config['training']['learning_rate'],
                     weight_decay=config['training']['weight_decay']) for a,h in heads.items()}
        losses={a:[] for a in ARMS}
        trainstart=time.perf_counter()
        iterations=8 if args.smoke else config['training']['iterations_per_fold']
        for step in range(1,iterations+1):
            features,target=cache[rng.randrange(len(cache))]
            idx=np_rng.integers(len(target),size=config['training']['points_per_step'])
            x,y=features[idx].float().cuda(),target[idx].cuda()
            for arm,head in heads.items():
                optimizers[arm].zero_grad(set_to_none=True)
                delta=head(mask_fine_features(x,feature_arms[arm]))[:,0]
                score=config['training']['baseline_margin']*x[:,69]+delta
                loss=F.binary_cross_entropy_with_logits(score,y,
                    pos_weight=torch.tensor(loss_weights[arm],device=y.device))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite point loss')
                loss.backward()
                optimizers[arm].step()
                losses[arm].append(float(loss.detach()))
            if step==1 or step%500==0 or step==iterations:
                print(json.dumps({'fold':fi,'step':step,'loss':{a:v[-1] for a,v in losses.items()}}),flush=True)
            if time.perf_counter()-started>config['maximum_gpu_hours']*3600:
                raise RuntimeError('Registered runtime budget exceeded')
        torch.cuda.synchronize()
        training_seconds=time.perf_counter()-trainstart
        if any(p.grad is not None or p.requires_grad for m in parents.values() for p in m.parameters()):
            raise ValueError('Parent received gradients')
        metadata={'experiment_id':config['experiment_id'],'fold':fi,'seed':config['seed'],
            'config_sha256':cfgdigest,'parents':parent_records,'train':trainnames,
            'oof':targetnames,'guard':sorted(guard),'iterations':iterations,
            'oof_split_sha256':config['oof_split_sha256'],'train_label_access':train_access}
        metadata['positive_loss_weights']=loss_weights
        metadata['sampled_training_class_counts']={'positive':positive_points,'negative':negative_points}
        checkpoints={}
        for arm,head in heads.items():
            path=out/f'fold{fi}_{arm}.pth'
            torch.save({'metadata':metadata,'arm':arm,'state_dict':head.state_dict()},path)
            loaded=torch.load(path,map_location='cpu',weights_only=True)
            fresh=point_head().cuda()
            fresh.load_state_dict(loaded['state_dict'],strict=True)
            if loaded['metadata']!=metadata or not torch.equal(fresh(sample),head(sample)):
                raise ValueError('Checkpoint reload mismatch')
            heads[arm]=fresh.eval()
            checkpoints[arm]={'path':str(path),'sha256':sha256(path),'reload_exact':True}
        del cache,optimizers
        inferstart=time.perf_counter()
        probability_diff=0.
        zero_images=0
        for index,name in enumerate(targetnames):
            rgb,static,base=[png(p) for p in (rgbpaths[name][0],staticpaths[name][0],baseline[name][0])]
            before=__import__('hashlib').sha256(rgb.tobytes()).hexdigest()
            fine,pdeck=extract(parents['deck'],rgb)
            # Verify feature extraction preserves the parent's coarse output.
            if index==0:
                repeat=sliding(parents['deck'],rgb,crop=960,stride=480,classes=2)[1]
                probability_diff=float(np.max(np.abs(pdeck-repeat)))
                if probability_diff!=0:
                    raise ValueError('Hidden-state extraction changed parent output')
            zero=fuse_residual(base,static,np.zeros(base.shape,np.float32))
            if not np.array_equal(zero,base):
                raise ValueError('Zero residual does not replay baseline')
            zero_images+=1
            new=infer_heads(heads,fine,rgb,pdeck,base,static,config)
            if before!=__import__('hashlib').sha256(rgb.tobytes()).hexdigest():
                raise ValueError('RGB array mutated')
            for arm,values in new.items():
                path=out/arm/name
                path.parent.mkdir(exist_ok=True)
                Image.fromarray(values).save(path)
                predictions[arm][name]={'path':str(path),'sha256':sha256(path),'fold':fi}
            del fine
            if index%20==0 or index+1==len(targetnames):
                print(json.dumps({'fold':fi,'predictions':index+1,'total':len(targetnames)}),flush=True)
        torch.cuda.synchronize()
        inference_seconds=time.perf_counter()-inferstart
        for kind,m in parents.items():
            payload=torch.load(art/parent_records[kind]['path'],map_location='cpu',weights_only=True)
            if any(not torch.equal(v.detach().cpu(),payload['model_state_dict'][k]) for k,v in m.state_dict().items()):
                raise ValueError('Frozen parent tensor changed')
        row={'fold':fi,'checkpoints':checkpoints,'parent_checkpoints':parent_records,
             'train_label_access':train_access,'oof_names':targetnames,'zero_residual_exact_images':zero_images,
             'coarse_probability_max_difference':probability_diff,'frozen_parent_tensors_exact':True,
             'rgb_arrays_exact':True,'train_seconds':training_seconds,'feature_seconds':feature_seconds,
             'positive_loss_weights':loss_weights,
             'inference_seconds':inference_seconds,'fold_seconds':time.perf_counter()-foldstart,
             'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
             'loss':{a:{'first':v[0],'last':v[-1],'last100':float(np.mean(v[-100:]))} for a,v in losses.items()}}
        records.append(row)
        write_json(out/f'fold{fi}_result.json',row)
        del parents,heads
        torch.cuda.empty_cache()
    write_json(out/'prediction_manifest.json',predictions)
    seal={'prediction_manifest_sha256':sha256(out/'prediction_manifest.json'),
          'config_sha256':cfgdigest,'repository_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'source_sha256':source_hashes,
          'label_access_before_seal':'only each fold train; no target GT used during its inference',
          'scope':'incremental OOF; inherited SEM-001 trained on all219; reused evaluation partitions',
          'folds':records}
    write_json(out/'prediction_seal.json',seal)
    result={'experiment_id':config['experiment_id'],'freeze_before':freeze,'folds':records,
            'seed':config['seed'],'config_sha256':cfgdigest,'seal_sha256':sha256(out/'prediction_seal.json'),
            'check_target_labels_read_this_run':[],'developer_target_labels_read_this_run':[],
            'status':'completed_smoke' if args.smoke else 'sealed_oof'}
    if not args.smoke:
        if any(set(predictions[a])!=set(allnames) for a in ARMS):
            raise ValueError('OOF universe incomplete')
        calibration=evaluate_partition(partition['calibration'],predictions,baseline,staticpaths,labels,config)
        result['calibration']=calibration
        write_json(out/'calibration_sealed.json',calibration)
        result['calibration_sha256']=sha256(out/'calibration_sealed.json')
        result['status']='calibration_passed' if calibration['passed'] else 'completed_calibration_gate_failed'
        if calibration['passed']:
            check=evaluate_partition(partition['check'],predictions,baseline,staticpaths,labels,config)
            result['check']=check
            result['check_target_labels_read_this_run']=partition['check']
            result['status']='qualified_for_final' if check['passed'] else 'completed_check_gate_failed'
        sample_names=[n for n in config['visual_samples'] if n in partition['calibration']]
        result['panels']=comparison_panels(sample_names,predictions,baseline,rgbpaths,labels,out/'panels')
    result['freeze_after']=verify_freeze(ROOT/'experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json',art)
    if result['freeze_after']!=freeze or sha256(cfgpath)!=cfgdigest:
        raise ValueError('Frozen configuration/artifacts changed')
    if any(sha256(ROOT/p)!=digest for p,digest in source_hashes.items()):
        raise ValueError('Experiment implementation changed during execution')
    result['seconds']=time.perf_counter()-started
    result['environment']={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
                           'python':sys.version,'head_parameters_per_arm':sum(p.numel() for p in point_head().parameters())}
    write_json(out/'result.json',result)
    print(json.dumps({'status':result['status'],'seconds':result['seconds'],
          'checks':result.get('calibration',{}).get('checks'), 'result_sha256':sha256(out/'result.json')}),flush=True)


if __name__=='__main__':
    main()
