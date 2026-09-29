#!/usr/bin/env python3
"""Read-only identity, budget, placement, recovery and saved-image audit."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,verify_hashes


def audit_recovery(run,expected_used):
    import torch
    latest=json.loads((run/'recovery/latest.json').read_text());path=Path(latest['path'])
    q=torch.load(path,map_location='cpu',weights_only=False)
    required={'binding','iteration','model','exposure','exposure_optimizer','python_rng','numpy_rng','torch_rng','cuda_rng',
              'viewpoint_stack','viewpoint_indices','ema_loss','ema_depth','geometry_state'}
    assert required<=set(q);assert len(q['model'])==13
    n=len(q['model'][1]);assert q['model'][8].shape==q['model'][9].shape==q['model'][10].shape==(n,1)
    assert q['model'][7].shape==(n,);assert len(q['viewpoint_stack'])==len(q['viewpoint_indices'])
    assert len(set(q['viewpoint_stack']))==len(q['viewpoint_stack']);assert len(q['cuda_rng'])==1
    assert len(q['geometry_state']['used'])==len(set(q['geometry_state']['used']))==expected_used
    assert q['geometry_state']['pending'] is None
    for tensor in q['model'][1:11]:assert torch.isfinite(tensor).all()
    for state in q['model'][11]['state'].values():
        for v in state.values():
            if isinstance(v,torch.Tensor):assert torch.isfinite(v).all()
    return {'iteration':q['iteration'],'points':n,'used_proposals':expected_used,'remaining_cameras':len(q['viewpoint_stack']),
            'complete_required_state_present':True,'path':str(path.relative_to(ROOT)),'sha256':sha256(path)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'outputs/quality/rgb144');p.add_argument('--stage',choices=['smoke','full'],default='full');a=p.parse_args()
    out=a.root.resolve();names=[f'{arm}-smoke-r1' if a.stage=='smoke' else f'{arm}-r1' for arm in ('control','candidate')];runs=[out/n for n in names]
    records=[json.loads((r/'result.json').read_text()) for r in runs];configs=[json.loads((r/'code/config.yaml').read_text()) for r in runs]
    assert all(r['status']==('completed_smoke' if a.stage=='smoke' else 'completed_all400') for r in records)
    for key in ['source','code_sha256','geometry_cache_sha256','initialization_ply_sha256','manifest_sha256','split_sha256','seed']:
        assert records[0][key]==records[1][key],key
    for key in ['data','dataset_manifest','split_id','seed','source']:assert configs[0][key]==configs[1][key],key
    assert [{k:v for k,v in c['geometry'].items() if k!='arm'} for c in configs][0]==[{k:v for k,v in c['geometry'].items() if k!='arm'} for c in configs][1]
    assert [{k:v for k,v in c['training'].items() if k!='output'} for c in configs][0]==[{k:v for k,v in c['training'].items() if k!='output'} for c in configs][1]
    assert sha256(runs[0]/'executed_train.py')==sha256(runs[1]/'executed_train.py')
    verify_hashes(ROOT/configs[0]['source'],records[0]['source'])
    assert sha256(ROOT/configs[0]['geometry']['cache'])==records[0]['geometry_cache_sha256']
    logs=[[json.loads(x) for x in (r/'numeric.jsonl').read_text().splitlines()] for r in runs]
    assert [(x['iteration'],x['camera']) for x in logs[0]]==[(x['iteration'],x['camera']) for x in logs[1]]
    props={p['id']:p for p in json.loads((out/'geometry-r1/proposals.json').read_text())['proposals']};summary={}
    for run,rec,config in zip(runs,records,configs):
        verify_hashes(run,rec['artifacts'])
        for name,h in rec['code_sha256'].items():assert sha256(run/'code'/name)==h
        assert len(rec['worker']['actual_train'])==400 and not rec['worker']['actual_evaluation']
        assert rec['worker']['initialization']['common_point_count']==72292
        events=[json.loads(x) for x in (run/'densification.jsonl').read_text().splitlines()]
        placements=[json.loads(x) for x in (run/'placements.jsonl').read_text().splitlines()]
        assert len(events)==len(placements)==(144 if a.stage=='full' else 11)
        assert [e['iteration'] for e in events]==list(range(600,15000 if a.stage=='full' else 1601,100))
        used=[];displacements=[];applied=0
        for e,p in zip(events,placements):
            assert e['iteration']==p['iteration'];assert e['selected']<=2048
            assert e['associated']==len(p['placements'])<=int(.25*e['selected'])
            assert e['selected']<=max(0,e['population_target']-e['before_split'])
            assert len({q['parent_id'] for q in p['placements']})==len(p['placements'])
            matrix=np.load(run/'associations'/f"{e['iteration']:05d}.npz")
            for q in p['placements']:
                assert q['proposal_id'] not in used;used.append(q['proposal_id']);assert q['distance_sigma']<=3
                np.testing.assert_allclose(q['photo_xyz'],props[q['proposal_id']]['xyz'],rtol=1e-6,atol=1e-6)
                match=(matrix['parent_row']==q['parent_row'])&(matrix['proposal_index']==q['proposal_index'])
                assert match.sum()==1 and matrix['reject_code'][match][0]==0 and matrix['projection_cameras'][match][0]>=3
                assert int(matrix['parent_ids'][q['parent_row']])==q['parent_id']
                assert q['applied']==(config['geometry']['arm']=='candidate');applied+=q['applied'];displacements.append(q['displacement'])
        assert events[-1]['guided_total']==applied;assert events[-1]['associated_total']==len(used)
        summary[run.name]={'associated':len(used),'applied':applied,'events':len(events),'displacement_quantiles':np.quantile(displacements,[0,.25,.5,.75,1]).tolist() if displacements else [],
                           'recovery':audit_recovery(run,len(used))}
    assert summary[runs[0].name]['applied']==0 and summary[runs[1].name]['applied']>0
    for name,h in json.loads((out/'preserved-inputs.json').read_text()).items():assert sha256(ROOT/name)==h,name
    if a.stage=='full':
        photo_hashes={}
        for file in (out/'geometry-r1/pairs').glob('*.npz'):
            pair=np.load(file)
            for i in (0,1):
                name=str(pair[f'camera{i}']);h=str(pair['input_sha256'][i])
                assert name not in photo_hashes or photo_hashes[name]==h
                photo_hashes[name]=h
        assert len(photo_hashes)==400
        for name,h in photo_hashes.items():assert sha256(ROOT/configs[0]['data']['scene']/'images'/name)==h,name
        labels=json.loads((ROOT/'outputs/quality/rgb141/evaluation_label_identity.json').read_text())['labels_sha256']
        assert len(labels)==300
        for name,h in labels.items():assert sha256(ROOT/'outputs/quality/all_data/SEM-370-r1/labels'/name)==h,name
        assert sha256(ROOT/configs[0]['dataset_manifest'])==records[0]['manifest_sha256']
        assert sha256(ROOT/configs[0]['split_path'])==records[0]['split_sha256']
        evals=[json.loads((r/'evaluation/result.json').read_text()) for r in runs]
        assert evals[0]['code_sha256']==evals[1]['code_sha256']==sha256(ROOT/'scripts/evaluate_standard_geometry.py')
        for run,e in zip(runs,evals):
            assert e['status']=='completed' and e['train_count']==400 and e['trajectory_count']==144 and e['fresh_fixed9_exact']
            assert len(e['image_hashes'])==544 and len({v['name'] for v in e['per_view']})==400
            assert all(v['max_byte_difference']==0 for v in e['historical_rgb140_fixed9_replay'].values())
            for name,h in e['image_hashes'].items():assert sha256(run/'evaluation/images'/name)==h
        assert evals[0]['trajectory_metadata_sha256']==evals[1]['trajectory_metadata_sha256']
    result={'status':'passed','stage':a.stage,'runs':summary,'same_native_source_config_cache_camera_order':True,'old_models_and_cameras_unchanged':True,
            'all400_fit_zero_holdout':True,'paired_budget_not_exact_final_count':True,'only_candidate_applies_xyz':True,
            'evaluation_photo400_label300_identity_verified':a.stage=='full','audit_source_sha256':sha256(Path(__file__))}
    (out/f'{a.stage}-pair-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
