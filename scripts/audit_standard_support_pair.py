#!/usr/bin/env python3
"""Verify pair identity and preserve provenance distinctions without retraining."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,verify_hashes

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--support-run',type=Path,default=ROOT/'outputs/quality/rgb141/support-r1');args=parser.parse_args()
    out=ROOT/'outputs/quality/rgb141';paths=[out/'control-r1',args.support_run]
    records=[json.loads((p/'result.json').read_text()) for p in paths]
    configs=[json.loads((p/'code/config.yaml').read_text()) for p in paths]
    for p,r in zip(paths,records):
        assert r['status']=='completed_all400' and r['actual_iterations']==30000
        verify_hashes(p,r['artifacts'])
        for name,h in r['code_sha256'].items():assert sha256(p/'code'/name)==h,name
    import numpy as np
    from plyfile import PlyData
    finite_counts={}
    for p in paths:
        vertices=PlyData.read(p/'model/point_cloud/iteration_30000/point_cloud.ply')['vertex'].data
        assert all(np.isfinite(vertices[name]).all() for name in vertices.dtype.names)
        finite_counts[p.name]=len(vertices)
    assert records[0]['source']==records[1]['source']
    for name in ['scripts/worker_standard_support.py','src/standard_support_densification.py','scripts/backbone_all400_worker.py']:
        assert records[0]['code_sha256'][name]==records[1]['code_sha256'][name]
    assert sha256(paths[0]/'executed_train.py')==sha256(paths[1]/'executed_train.py')
    for key in ['data','dataset_manifest','split_id','seed','source']:
        assert configs[0][key]==configs[1][key],key
    shared=[{k:v for k,v in c['support'].items() if k!='arm'} for c in configs]
    assert shared[0]==shared[1]
    recipe=[{k:v for k,v in c['training'].items() if k!='output'} for c in configs]
    assert recipe[0]==recipe[1]
    for k in ['actual_train','actual_evaluation','initialization']:
        assert records[0]['worker'][k]==records[1]['worker'][k]
    logs=[[json.loads(x) for x in (p/'numeric.jsonl').read_text().splitlines()] for p in paths]
    assert len(logs[0])==len(logs[1])==300
    assert [(r['iteration'],r['camera']) for r in logs[0]]==[(r['iteration'],r['camera']) for r in logs[1]]
    evaluations=[json.loads((p/'evaluation/result.json').read_text()) for p in paths]
    for p,r in zip(paths,evaluations):
        assert r['status']=='completed' and r['train_count']==400 and r['trajectory_count']==144
        assert r['fresh_fixed9_exact'] and len(r['image_hashes'])==544
        for name,h in r['image_hashes'].items():assert sha256(p/'evaluation/images'/name)==h,name
        assert all(x['max_byte_difference']==0 for x in r['historical_rgb140_fixed9_replay'].values())
    assert evaluations[0]['trajectory_metadata_sha256']==evaluations[1]['trajectory_metadata_sha256']
    identity=json.loads((out/'input_identity.json').read_text())
    labels=json.loads((out/'evaluation_label_identity.json').read_text())['labels_sha256']
    for name,h in labels.items():
        assert sha256(ROOT/'outputs/quality/all_data/SEM-370-r1/labels'/name)==h,name
    for name,h in identity['parent_models'].items():assert sha256(ROOT/name)==h,name
    for name,h in identity['sparse_sha256'].items():assert sha256(ROOT/'outputs/quality/all_data/RGB-120-r2/scene/sparse/0'/name)==h,name
    for r in records:
        assert r['split_sha256']==identity['split_sha256'] and r['manifest_sha256']==identity['manifest_sha256']
    before=json.loads((out/'workspace_before.json').read_text());missing=[name for name in before if not (ROOT/name).exists()]
    assert not missing
    changed=[name for name,h in before.items() if sha256(ROOT/name)!=h]
    result={'status':'passed','models_and_official_camera_files_unchanged':True,
        'all_saved_gaussian_fields_finite':finite_counts,
        'evaluation_label_masks_unchanged':len(labels),
        'same_400_train_and_zero_holdout':True,'same_initialization':True,'same_native_source_files':len(records[0]['source']),
        'same_scientific_hook_and_executed_native_train':True,'same_budget_and_training_recipe_except_arm':True,
        'camera_order_verified_at_300_logged_iterations':True,
        'evaluation_1088_image_hashes_verified':True,'same_144_trajectory_metadata':True,
        'historical_rgb140_fixed9_replay_byte_exact_both_arms':True,
        'baseline_workspace_files_missing':missing,'baseline_workspace_files_changed':changed,
        'baseline_workspace_note':'Other independent tasks may append shared documents; this is an inventory, not attribution of all changes to this task.',
        'metadata_clarification':'The legacy method.all_other_densification_and_optimization_settings field in raw runner results describes the inherited native recipe only. registered_support, code/config.yaml and executed hooks additionally impose the shared split budget and optional support ranking. Neither arm is historical RGB140.',
        'orchestration_note':'Control started with a nonblocking lock. Candidate uses a blocking lock plus context-drain wait for the independent semantic task; scientific hook hashes remain equal.'}
    (out/'pair_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
