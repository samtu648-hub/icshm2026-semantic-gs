#!/usr/bin/env python3
"""Independent SEM-270 provenance, paired training and native prediction replay."""
import argparse
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_sem270_foundation_binary_oof import initialize, array_hash, shared_state_hash
from audit_sem260_error_budget import manifest_paths, png
from run_sem250_colmap_track_deck_diag import verify_freeze
from src.semantic2d import read_json, write_json, sha256
from src.submission_sem161 import sliding


def require(ok,message):
    if not ok: raise ValueError(message)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('artifact-root','run-dir','output-dir'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'experiments/runs/SEM-270-FOUNDATION-BINARY-OOF/config.yaml')
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    torch.set_num_threads(4)
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error();hf_logging.disable_progress_bar()
    run=read_json(args.run_dir/'result.json');seal=read_json(args.run_dir/'prediction_seal.json');cfg=read_json(args.config)
    require(run['experiment_id']==cfg['experiment_id']=='SEM-270-FOUNDATION-BINARY-OOF','Experiment mismatch')
    require(run['config_sha256']==seal['config_sha256']==sha256(args.config),'Configuration changed')
    require(run['prediction_seal_sha256']==sha256(args.run_dir/'prediction_seal.json'),'Seal changed')
    require(seal['prediction_manifest_sha256']==sha256(args.run_dir/'prediction_manifest.json'),'Prediction manifest changed')
    require(run['calibration_sha256']==sha256(args.run_dir/'calibration_sealed.json'),'Calibration changed')
    for name,digest in seal['source_sha256'].items():
        require(sha256(ROOT/name)==digest,'Current source changed')
        require(sha256(args.run_dir/'source_snapshot'/Path(name).name)==digest,'Snapshot changed')
    for k in ('semantic_split','oof_split'): require(sha256(ROOT/cfg[k])==cfg[k+'_sha256'],'Split changed')
    folds=read_json(ROOT/cfg['oof_split'])
    graph=ROOT/cfg['camera_graph'];require(sha256(graph)==folds['camera_graph_sha256'],'Camera graph changed')
    centers={n['name']:np.asarray(n['center']) for n in read_json(graph)['nodes']}
    allnames=read_json(ROOT/cfg['semantic_split'])['semantic_train_labeled']
    rgbpaths=manifest_paths(args.artifact_root/cfg['parents']['rgb'],'rendered_dataset_manifest.json','rendered_images','path','sha256')
    baseline=manifest_paths(args.artifact_root/cfg['parents']['sem160'],'prediction_manifest.json','predictions','prediction','prediction_sha256')
    labels={r['image']:r for r in read_json(ROOT/'experiments/runs/SEM-001/label_manifest.json')['records']}
    pred=read_json(args.run_dir/'prediction_manifest.json');rows=[];counter=0
    for mapping in (baseline,rgbpaths):
        require(set(mapping)==set(allnames),'Wrong parent universe')
        for path,digest in mapping.values(): require(sha256(path)==digest,'Parent RGB/baseline changed')
    for arm in cfg['arm_names']: require(set(pred[arm])==set(allnames),'Incomplete OOF')
    for fold,row in zip(folds['folds'],run['folds']):
        fi=fold['fold'];require(fi==row['fold'],'Fold ordering')
        train,held,guard=(set(fold[k]) for k in ['train','oof','guard'])
        require(not (train&held or train&guard or held&guard) and train|held|guard==set(allnames),'Fold overlap')
        distance=np.linalg.norm(np.asarray([centers[n] for n in train])[:,None]-np.asarray([centers[n] for n in held])[None],axis=-1)
        require(float(distance.min())>=folds['guard_policy']['radius']-1e-9,'Camera guard violation')
        planpath=args.run_dir/f'fold{fi}_sampling_plan.json';plan=read_json(planpath)
        require(len(plan)==cfg['training']['iterations'],'Training budget changed')
        rng=random.Random(cfg['seed']+fi*100003);names=sorted(train);crop=cfg['training']['crop']
        for step in plan:
            expected={'name':names[rng.randrange(len(names))],'top':rng.randint(0,989-crop),'left':rng.randint(0,1320-crop),'flip':rng.random()<.5}
            require(step==expected,'Shared augmentation sequence differs')
        shared=[]
        for arm in cfg['arm_names']:
            record=row['arms'][arm]['training'];path=Path(record['path'])
            require(sha256(path)==record['sha256'],'Checkpoint changed')
            payload=torch.load(path,map_location='cpu',weights_only=True);meta=payload['metadata']
            require(meta==record['metadata'] and meta['experiment_id']==cfg['experiment_id'] and meta['arm']==arm and meta['fold']==fi,'Wrong checkpoint metadata')
            require(meta['mode']=='full' and meta['iterations']==3000 and meta['seed']==cfg['seed'],'Wrong fit budget/seed')
            require(set(meta['train'])==train and set(meta['outer_train'])==train and set(meta['oof'])==held and set(meta['guard'])==guard,'Wrong fit universe')
            require(meta['config_sha256']==sha256(args.config) and meta['oof_split_sha256']==cfg['oof_split_sha256'],'Checkpoint protocol mismatch')
            access=meta['actual_label_access']
            require(len(access)==len(train) and {a['name'] for a in access}==train,'Actual label access differs')
            require(set(meta['actual_optimization_names'])==train,'Training view not optimized')
            require(meta['sampling_plan_sha256']==sha256(planpath),'Sampling hash differs')
            for a in access:
                require(a['label_sha256']==labels[a['name']]['label_png_sha256']==sha256(args.artifact_root/cfg['parents']['labels']/a['name']),'Label differs from original manifest')
                require(a['rgb_sha256']==rgbpaths[a['name']][1],'Wrong fitting RGB')
            counts=np.sum([[labels[n]['pixel_counts'].get(str(k),0) for k in range(5)] for n in train],axis=0)
            if arm=='binary_candidate': counts=np.array([counts.sum()-counts[4],counts[4]])
            require(counts.tolist()==meta['class_counts'],'Wrong class count')
            weights=np.clip(np.median(counts.astype(np.float64))/counts,.05,20).astype(np.float32)
            require(weights.tolist()==meta['class_weights'],'Wrong contribution/loss weights')
            for init in meta['initialization']: require(sha256(args.artifact_root/init['path'])==init['sha256'],'Original initialization changed')
            shared.append(meta['shared_initial_state_sha256'])
            for name in sorted(held):
                r=pred[arm][name];pp=Path(r['probability_path']);pn=Path(r['path'])
                require(r['fold']==fi and sha256(pp)==r['probability_sha256'] and sha256(pn)==r['sha256'],'Prediction provenance changed')
                prob=np.load(pp,allow_pickle=False);base=png(baseline[name][0]);new=png(pn)
                require(prob.dtype==np.float32 and prob.shape==base.shape and np.isfinite(prob).all() and np.all((prob>=0)&(prob<=1)),'Invalid probability')
                expected=base.copy();lane=(base==0)|(base==4);expected[lane]=np.where(prob[lane]>=.5,4,0)
                require(np.array_equal(expected,new),'Independent fusion mismatch')
                require(all(np.array_equal(new==k,base==k) for k in (1,2,3)),'Protected class mask changed')
                counter+=1
            # Independently reload and render first held-out native RGB, no GT.
            model=initialize(args.artifact_root/cfg['parents']['model_config'],arm,cfg['seed'])
            require(shared_state_hash(model)==meta['shared_initial_state_sha256'],'Reconstructed shared initialization differs')
            changed_tensors=sum(not torch.equal(v,payload['model_state_dict'][k]) for k,v in model.state_dict().items())
            require(changed_tensors>0,'New classifier was not trained')
            require(sum(p.numel() for p in model.parameters())==meta['parameters']==meta['trainable_parameters'],'Parameter budget differs')
            model.load_state_dict(payload['model_state_dict'],strict=True)
            model.cuda().eval().requires_grad_(False)
            first=sorted(held)[0];rgb=png(rgbpaths[first][0]);before=array_hash(rgb)
            classes=5 if arm=='five_control' else 2
            probability=sliding(model,rgb,crop=960,stride=480,classes=classes)[4 if classes==5 else 1]
            require(np.array_equal(probability,np.load(pred[arm][first]['probability_path'],allow_pickle=False)),'Reloaded probability replay differs')
            require(array_hash(rgb)==before,'RGB input mutated')
            rows.append({'fold':fi,'arm':arm,'checkpoint_sha256':record['sha256'],'native_replay_view':first,'probability_replay_exact':True,'changed_parameter_tensors':changed_tensors,'weights':weights.tolist(),'class_counts':counts.tolist()})
            del model,payload;torch.cuda.empty_cache()
        require(shared[0]==shared[1],'Initial shared weights differ')
    require(len(rows)==6 and counter==438,'Missing models or predictions')
    freeze=verify_freeze(ROOT/'experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json',args.artifact_root)
    require(freeze==run['freeze_before']==run['freeze_after'],'Frozen manifest mismatch')
    result={'experiment_id':cfg['experiment_id'],'status':'passed','checkpoint_replays':rows,'prediction_files_verified':counter,
        'paired_training_steps':9000,'parent_images_verified':438,'frozen_artifacts':18,'target_labels_decoded_for_metrics':[],
        'result_sha256':sha256(args.run_dir/'result.json'),'code_sha256':sha256(Path(__file__)),
        'seconds':time.perf_counter()-started,'note':'Metadata, original label counts/hashes and unlabeled model replay; no new target metric evaluation'}
    write_json(args.output_dir/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__': main()
