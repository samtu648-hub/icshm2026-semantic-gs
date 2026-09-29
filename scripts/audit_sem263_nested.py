#!/usr/bin/env python3
"""Independent post-run provenance/cache/paired-head audit, without target scoring."""
import argparse
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
import torch
import transformers
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from src.semantic2d import read_json, write_json, sha256
from src.semantic_nested_oof import validate_inner, validate_teacher_metadata
from src.semantic_protected_fusion import partition_training_names
from src.semantic_refinement import point_head, mutable_lane, fuse_residual
from audit_sem260_error_budget import manifest_paths
from run_sem250_colmap_track_deck_diag import verify_freeze


def array_hash(tensor):
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('artifact-root','run-dir','output-dir'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'experiments/runs/SEM-263-NESTED-POINT-OOF/config.yaml')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    torch.set_num_threads(4)
    cfg=read_json(args.config)
    run=read_json(args.run_dir/'result.json')
    seal=read_json(args.run_dir/'prediction_seal.json')
    require(run['seal_sha256']==sha256(args.run_dir/'prediction_seal.json'),'Seal changed')
    require(run['config_sha256']==sha256(args.config),'Configuration changed')
    require(seal['prediction_manifest_sha256']==sha256(args.run_dir/'prediction_manifest.json'),'Prediction manifest changed')
    require(all(sha256(ROOT/p)==h for p,h in seal['source_sha256'].items()),'Source changed')
    plan=read_json(args.run_dir/'nested_plan.json')
    require(seal['nested_plan_sha256']==sha256(args.run_dir/'nested_plan.json'),'Nested plan changed')
    folds=read_json(ROOT/cfg['oof_split'])['folds']
    radius=read_json(ROOT/cfg['oof_split'])['guard_policy']['radius']
    centers={n['name']:np.asarray(n['center']) for n in read_json(ROOT/cfg['camera_graph'])['nodes']}
    allnames=read_json(ROOT/cfg['semantic_split'])['semantic_train_labeled']
    label_manifest_path=ROOT/'experiments/runs/SEM-001/label_manifest.json'
    label_records={r['image']:r for r in read_json(label_manifest_path)['records']}
    partition=partition_training_names(allnames,expected_count=219)
    require([a['name'] for a in run['calibration']['target_access']]==partition['calibration'],'Calibration target access differs')
    require(not run['developer_target_labels_read_this_run'],'Unexpected developer scoring')
    require(run['calibration']['passed'] or not run['check_target_labels_read_this_run'],'Check opened after calibration failure')
    predictions=read_json(args.run_dir/'prediction_manifest.json')
    baseline=manifest_paths(args.artifact_root/cfg['parents']['sem160'],'prediction_manifest.json','predictions','prediction','prediction_sha256')
    staticpaths=manifest_paths(args.artifact_root/cfg['parents']['sem001'],'training_sem001_render_manifest.json','rendered','semantic_png','sha256')
    rgbpaths=manifest_paths(args.artifact_root/cfg['parents']['rgb'],'rendered_dataset_manifest.json','rendered_images','path','sha256')
    for mapping in (baseline,staticpaths,rgbpaths):
        require(set(mapping)==set(allnames),'Parent image universe differs')
        require(all(sha256(p)==h for p,h in mapping.values()),'Parent RGB/static/baseline artifact changed')
    counters={'inner_teachers':0,'outer_heads':0,'paired_views':0,'prediction_files':0,
              'paired_sample_coordinates':0,'zero_output_heads':0}
    files=[]
    feature_alignment=[]
    for outer,nested,row in zip(folds,plan,run['folds']):
        fi=outer['fold']
        require(row['fold']==fi==nested['outer_fold'],'Outer fold mismatch')
        require(set(row['oof_names'])==set(outer['oof']),'Outer target coverage mismatch')
        require({a['name'] for a in row['train_label_access']}==set(outer['train']),'Head training access mismatch')
        for inner,record in zip(nested['inners'],row['inner_teachers']):
            validate_inner(outer,inner,centers,radius)
            require(record['inner']==inner,'Inner plan mismatch')
            require(set(record['teachers'])=={'deck','cable','five'},'Teacher family missing')
            for kind,teacher in record['teachers'].items():
                path=Path(teacher['path'])
                require(sha256(path)==teacher['sha256'],'Teacher changed')
                payload=torch.load(path,map_location='cpu',weights_only=True)
                metadata=payload['metadata']
                validate_teacher_metadata(metadata,outer,inner,kind)
                require(metadata==teacher['metadata'],'Teacher metadata changed')
                require(metadata['mode']=='full' and metadata['iterations']==3000,'Incomplete teacher')
                require({a['name'] for a in metadata['actual_label_access']}==set(inner['train']),'Inner actual label access mismatch')
                require(set(metadata['actual_optimization_names'])==set(inner['train']),'Inner image never optimized')
                for access in metadata['actual_label_access']:
                    expected=label_records[access['name']]['label_png_sha256']
                    require(access['label_sha256']==expected,'Teacher label differs from original manifest')
                    require(sha256(args.artifact_root/'r2/sem_001/labels'/access['name'])==expected,'Teacher label changed')
                class_counts=np.sum([[label_records[n]['pixel_counts'].get(str(k),0) for k in range(5)]
                                     for n in inner['train']],axis=0)
                if kind!='five':
                    positive=class_counts[1 if kind=='deck' else 2]
                    class_counts=np.array([class_counts.sum()-positive,positive])
                require(class_counts.tolist()==metadata['class_counts'],'Teacher weights used wrong class counts')
                weights=np.clip(np.median(class_counts.astype(np.float64))/class_counts,.05,20).astype(np.float32)
                require(weights.tolist()==metadata['class_weights'],'Teacher class weighting differs')
                for source in metadata['initialization']:
                    require(sha256(args.artifact_root/source['path'])==source['sha256'],'Pre-contest initialization changed')
                require(teacher['reload_exact'],'Teacher reload failed')
                counters['inner_teachers']+=1
                files.append({'path':str(path.relative_to(ROOT)),'sha256':teacher['sha256']})
                del payload
        cachepath=args.run_dir/f'fold{fi}_paired_features.pth'
        require(sha256(cachepath)==row['paired_cache_sha256'],'Paired cache changed')
        cache=torch.load(cachepath,map_location='cpu',weights_only=True)
        targets=cache['targets']
        access={a['name']:a for a in row['train_label_access']}
        require(set(targets)==set(outer['train']),'Paired target coverage differs')
        alignment={'fold':fi,'points':0,'fine_cosine_sum':0.,'fine_squared_difference_sum':0.,
                   'coarse_probability_absolute_difference_sum':0.,'baseline_sign_disagreements':0}
        for name in sorted(targets):
            left=cache['features'][cfg['arm_names'][0]][name].float()
            right=cache['features'][cfg['arm_names'][1]][name].float()
            alignment['points']+=len(left)
            alignment['fine_cosine_sum']+=float(torch.nn.functional.cosine_similarity(left[:,:64],right[:,:64],dim=1).sum())
            alignment['fine_squared_difference_sum']+=float((left[:,:64]-right[:,:64]).square().sum())
            alignment['coarse_probability_absolute_difference_sum']+=float((left[:,68]-right[:,68]).abs().sum())
            alignment['baseline_sign_disagreements']+=int((left[:,69]!=right[:,69]).sum())
            require(torch.equal(left[:,64:67],right[:,64:67]),'Paired native RGB features differ')
            require(torch.equal(left[:,70:],right[:,70:]),'Paired inherited static features differ')
        alignment['fine_mean_cosine']=alignment['fine_cosine_sum']/alignment['points']
        alignment['fine_rms_difference']=(alignment['fine_squared_difference_sum']/(alignment['points']*64))**.5
        alignment['coarse_probability_mae']=alignment['coarse_probability_absolute_difference_sum']/alignment['points']
        feature_alignment.append(alignment)
        for name,y in targets.items():
            index=cache['sample_indices'][name]
            require(array_hash(index)==access[name]['sample_indices_sha256'],'Sample coordinate mismatch')
            require(array_hash(y)==access[name]['targets_sha256'],'Shared target mismatch')
            require(access[name]['label_sha256']==label_records[name]['label_png_sha256'],'Head label differs from original manifest')
            counters['paired_sample_coordinates']+=len(index)
            counters['paired_views']+=1
        for arm in cfg['arm_names']:
            features=cache['features'][arm]
            require(set(features)==set(targets),'Feature/target universe mismatch')
            for feature_record in row['feature_records'][arm]:
                tensor=features[feature_record['name']]
                require(tensor.shape==(16384,75) and tensor.dtype==torch.float16,'Unexpected features')
                require(array_hash(tensor)==feature_record['feature_sha256'],'Feature values changed')
            torch.manual_seed(cfg['seed'])
            zero=point_head()
            probe=features[sorted(features)[0]][:128].float()
            require(torch.count_nonzero(zero(probe)).item()==0,'Zero head not exact')
            counters['zero_output_heads']+=1
            checkpoint=row['checkpoints'][arm]
            path=Path(checkpoint['path'])
            require(sha256(path)==checkpoint['sha256'],'Point checkpoint changed')
            payload=torch.load(path,map_location='cpu',weights_only=True)
            first,second=point_head(),point_head()
            first.load_state_dict(payload['state_dict'],strict=True)
            second.load_state_dict(payload['state_dict'],strict=True)
            require(torch.equal(first(probe),second(probe)),'Independent CPU reload mismatch')
            m=payload['metadata']
            require(m['iterations']==3000 and m['config_sha256']==sha256(args.config),'Head fit protocol mismatch')
            require(m['positive_loss_weight']==row['positive_loss_weight'],'Shared weight mismatch')
            require(m['provenance']['paired_cache_sha256']==row['paired_cache_sha256'],'Head cache mismatch')
            require(set(predictions[arm])==set(allnames),'Prediction universe mismatch')
            counters['outer_heads']+=1
            files.append({'path':str(path.relative_to(ROOT)),'sha256':checkpoint['sha256']})
        del cache
    for name in allnames:
        base=np.asarray(Image.open(baseline[name][0]))
        static=np.asarray(Image.open(staticpaths[name][0]))
        protected=np.isin(base,[2,3,4]) | ((base==0)&(static==4))
        require(np.array_equal(protected,~mutable_lane(base,static)),'Protection implementation differs from independent contract')
        require(np.array_equal(fuse_residual(base,static,np.zeros(base.shape,np.float32)),base),'Native zero residual differs')
        for arm in cfg['arm_names']:
            entry=predictions[arm][name]
            require(sha256(Path(entry['path']))==entry['sha256'],'Prediction changed')
            require(name in folds[entry['fold']]['oof'],'Wrong outer prediction fold')
            new=np.asarray(Image.open(entry['path']))
            require(np.array_equal(new[protected],base[protected]),'Protected predictions differ')
            require(all(np.array_equal(new==k,base==k) for k in (2,3,4)),'Protected class masks differ')
            counters['prediction_files']+=1
    require(counters['inner_teachers']==18 and counters['outer_heads']==6,'Missing model')
    require(counters['prediction_files']==438,'Missing prediction')
    freeze=verify_freeze(ROOT/'experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json',args.artifact_root)
    require(freeze==run['freeze_before']==run['freeze_after'],'Freeze mismatch')
    write_json(args.output_dir/'result.json',{'experiment_id':cfg['experiment_id'],'status':'passed',
        'counts':counters,'frozen_artifacts':freeze['count'],'parent_images_verified':657,
        'native_zero_equivalence_views':219,'protected_class_masks_exact':True,
        'seconds':time.perf_counter()-started,
        'result_sha256':sha256(args.run_dir/'result.json'),'files':files,
        'label_manifest_sha256':sha256(label_manifest_path),
        'paired_feature_alignment_label_free':feature_alignment,
        'current_environment':{'torch':torch.__version__,'transformers':transformers.__version__,
                               'python':sys.version},
        'labels_decoded_for_metrics':[],'note':'Metadata, tensor/cache/file hashes and reload only; no new target scoring'})
    print(counters)


if __name__=='__main__':
    main()
