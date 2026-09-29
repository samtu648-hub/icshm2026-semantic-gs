#!/usr/bin/env python3
"""Fixed3k/6k OOF comparison after all three GPU fits and fresh audits pass."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.quality_expert_duration import validate_recipe
from src.quality_experts import validate_probability
from src.quality_static_training import TrainingLabels
from src.quality_fusion import pipeline
from src.quality_semantic_audit import independent_confusion, metrics
from src.quality_foundation_policy import changes
from src.semantic_protected_fusion import partition_training_names

CONFIG = 'experiments/runs/SEM-335-SCAFFOLD-EXPERT-DURATION/config.yaml'
CODE = ('scripts/evaluate_quality_expert_duration.py', 'src/quality_expert_duration.py', 'src/quality_protocol.py',
    'src/quality_experts.py', 'src/quality_static_training.py', 'src/quality_fusion.py', 'src/quality_semantic_audit.py',
    'src/quality_foundation_policy.py', 'src/semantic_protected_fusion.py', 'src/semantic2d.py')


def accepted_candidate(root, batch, part, cfg, control):
    run, audit = batch / ('full-'+part), batch / ('audit-'+part)
    r = json.loads((run/'result.json').read_text())
    a = json.loads((audit/'result.json').read_text())
    if r['experiment_id'] != cfg['experiment_id'] or r['status'] != 'completed_expert_fit_pending_independent_audit' or r['trained_steps'] != 6000 or not r.get('prefix_3000',{}).get('exact'):
        raise ValueError('Incomplete duration candidate/control reproduction')
    if r['binding']['config_sha256'] != sha256(root/CONFIG) or r['control_result_sha256'] != cfg['study']['controls'][part]['sha256'] or r['access'] != control['access']:
        raise ValueError('Candidate configuration or OOF differs')
    if a['status'] != 'completed_independent_expert_replay' or a['run_result_sha256'] != sha256(run/'result.json') or a['binding'] != r['binding']:
        raise ValueError('Candidate lacks matching independent audit')
    fresh = a['fresh_reloads']
    if len(fresh) != 2 or len({v['pid'] for v in fresh}) != 2 or any(not v['probabilities_exact'] or v['views'] != control['prediction_views'] for v in fresh):
        raise ValueError('Incomplete fresh probability replay')
    verify_hashes(run, r['artifacts'])
    verify_hashes(root, r['code_sha256'])
    return run, r, audit


def source_index(root, cfg, batch):
    parent_path = root/cfg['study']['baseline_fusion']
    if sha256(parent_path/'result.json') != cfg['study']['baseline_fusion_sha256']:
        raise ValueError('Matched fusion source differs')
    parent = json.loads((parent_path/'result.json').read_text())
    sources, proofs = {}, {}
    for rel, digest in parent['input_result_hashes'].items():
        path = root/rel
        if sha256(path/'result.json') != digest:
            raise ValueError('Baseline source result changed')
        r = json.loads((path/'result.json').read_text())
        part = r.get('access',{}).get('partition')
        if part not in cfg['partitions']:
            continue
        if r['status'] == 'completed_static_fit_pending_independent_evaluation':
            branch = 'static'
        elif r['status'] == 'completed_expert_fit_pending_independent_audit':
            branch = r['binding']['expert']
        else:
            continue
        verify_hashes(path, r['artifacts'])
        sources.setdefault(part,{})[branch] = (path, r)
        proofs[rel] = digest
    index = {}
    for part in cfg['partitions']:
        s = sources[part]
        if set(s) != {'static','fiveclass','deck','stay_cable'}:
            raise ValueError('Missing frozen fusion input')
        control_path, control = s['fiveclass']
        if str(control_path.relative_to(root)) != cfg['study']['controls'][part]['run']:
            raise ValueError('Wrong same-fold3k comparator')
        candidate_path, candidate, audit = accepted_candidate(root,batch,part,cfg,control)
        s['sixk'] = (candidate_path,candidate)
        for p in (candidate_path,audit):
            proofs[str(p.relative_to(root))] = sha256(p/'result.json')
        for name in control['prediction_views']:
            if name in index or any(r['access'] != control['access'] for _,r in s.values()):
                raise ValueError('OOF duplication or access mismatch')
            index[name] = (part,s)
    return parent,parent_path,index,proofs


def load_inputs(sources, name):
    path,r = sources['static']
    rel = 'predictions/'+name
    if sha256(path/rel) != r['artifacts'][rel]:
        raise ValueError('Static prediction changed')
    with Image.open(path/rel) as im:
        static = np.array(im)
    probs = {}
    for branch in ('fiveclass','sixk','deck','stay_cable'):
        path,r = sources[branch]
        rel = 'probabilities/'+Path(name).stem+'.npz'
        if sha256(path/rel) != r['artifacts'][rel]:
            raise ValueError('Probability source changed')
        with np.load(path/rel,allow_pickle=False) as z:
            value = z['probabilities']
        validate_probability(value,5 if branch in ('fiveclass','sixk') else 2,(989,1320))
        probs[branch] = value if branch in ('fiveclass','sixk') else value[1]
    return static,probs


def check_fixed_foundation_transition(static, old, new, p4, thresholds):
    # Independent simplified construction: the frozen binary cable step
    # overwrites the entire old0/2 lane, so only original class4 can differ.
    if thresholds != {'fiveclass_lane':.8,'cable':.75,'foundation':.8,'deck':.98}:
        raise ValueError('Not the registered unchanged fusion thresholds')
    expected = old.copy()
    roi = static == 4
    expected[roi] = np.where(p4[roi] < .8,0,4)
    if not np.array_equal(expected,new) or any(not np.array_equal(old == c,new == c) for c in (1,2,3)):
        raise ValueError('Independent fixed fusion/protected class check failed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('batch','artifact-root','output'):
        p.add_argument('--'+k,type=Path,required=True)
    a = p.parse_args()
    a.batch = a.batch.resolve()
    a.output.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    cfg = json.loads((ROOT/CONFIG).read_text())
    record = {'experiment_id':cfg['experiment_id'],'status':'initializing_fixed_oof_comparison', 'seed':cfg['seed'],
        'config_sha256':sha256(ROOT/CONFIG),'code_sha256':{n:sha256(ROOT/n) for n in CODE},
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'threshold_search':False,'development_labels_decoded':False,'automatic_promotion':False,'whole_chain_spatial_validation':False}
    try:
        validate_recipe(cfg,json.loads((ROOT/cfg['study']['baseline_config']).read_text()))
        parent,pp,index,proofs = source_index(ROOT,cfg,a.batch)
        protocol = json.loads((ROOT/cfg['protocol']).read_text())
        verify_hashes(ROOT,protocol['input_hashes'])
        frozen = {v['path']:v['sha256'] for v in json.loads((ROOT/protocol['frozen_manifest']).read_text())['artifacts']}
        verify_hashes(a.artifact_root,frozen)
        split = json.loads((ROOT/'experiments/splits/sem_001_scaffold_static_v1.json').read_text())
        if set(index) != set(split['semantic_train_labeled']) or set(index)&set(split['semantic_validation_labeled']):
            raise ValueError('Comparison must cover219 OOF and no developer views')
        oof = json.loads((ROOT/'experiments/splits/sem_005_oof_spatial_v1.json').read_text())
        if sha256(ROOT/cfg['label_manifest']) != oof['label_manifest_sha256']:
            raise ValueError('Target label manifest changed')
        label_manifest = json.loads((ROOT/cfg['label_manifest']).read_text())
        roles = partition_training_names(sorted(index),expected_count=219)
        thresholds = parent['thresholds']
        record.update(input_result_hashes=proofs,input_hashes=protocol['input_hashes'],thresholds=thresholds,
            parent_result_sha256=cfg['study']['baseline_fusion_sha256'])
        for name in (*CODE,CONFIG):
            dest = a.output/'code'/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/name,dest)
        write_new_json(a.output/'fixed_protocol.json', {**record,'roles':roles,'target_labels_read':False})
        seal = sha256(a.output/'fixed_protocol.json')
        evaluation = {}
        for role in ('calibration','check'):
            labels = TrainingLabels(a.artifact_root/cfg['labels'],roles[role],label_manifest,cfg['resolution'])
            names = ('raw_3k','raw_6k','full_3k','full_6k')
            pooled = {n:np.zeros((5,5),np.int64) for n in names}
            costs = dict(changed=0,corrected=0,harmed=0,wrong_to_wrong=0)
            groups,per_view = {},{}
            for name in roles[role]:
                part,sources = index[name]
                static,probs = load_inputs(sources,name)
                old = pipeline(static,probs['fiveclass'],probs['stay_cable'],probs['deck'],thresholds)
                new = pipeline(static,probs['sixk'],probs['stay_cable'],probs['deck'],thresholds)
                rel = role+'/deck/'+name
                if sha256(pp/rel) != parent['artifacts'][rel]:
                    raise ValueError('Parent saved fusion changed')
                with Image.open(pp/rel) as im:
                    if not np.array_equal(old['deck'],np.asarray(im)):
                        raise ValueError('3k control fusion does not reproduce matched baseline')
                check_fixed_foundation_transition(static,old['deck'],new['deck'],probs['sixk'][4],thresholds)
                target = labels.read(name)
                maps = dict(raw_3k=probs['fiveclass'].argmax(0).astype(np.uint8),raw_6k=probs['sixk'].argmax(0).astype(np.uint8),full_3k=old['deck'],full_6k=new['deck'])
                matrices = {n:independent_confusion(target,v) for n,v in maps.items()}
                c = changes(old['deck'],new['deck'],target)
                for k in costs: costs[k] += c[k]
                group = groups.setdefault(part,{n:np.zeros((5,5),np.int64) for n in names})
                for k in names: pooled[k] += matrices[k]; group[k] += matrices[k]
                per_view[name] = {'partition':part,'metrics':{k:metrics(v) for k,v in matrices.items()},'changes':c}
                for stage,value in new.items():
                    dest = a.output/role/stage/name; dest.parent.mkdir(parents=True,exist_ok=True); Image.fromarray(value).save(dest)
                dest = a.output/role/'modified'/name; dest.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray((old['deck'] != new['deck']).astype(np.uint8)*255).save(dest)
            if pooled['full_3k'].tolist() != parent['evaluation'][role]['metrics']['confusion']:
                raise ValueError('Pooled control does not reproduce parent')
            evaluation[role] = {'views':roles[role],'metrics':{k:metrics(v) for k,v in pooled.items()},'changes':costs,
                'per_partition':{k:{n:metrics(m) for n,m in v.items()} for k,v in groups.items()}}
            write_new_json(a.output/role/'per_view.json',per_view)
            print(json.dumps({'role':role,'mIoU':{k:metrics(v)['mIoU_with_background'] for k,v in pooled.items()},'changes':costs}),flush=True)
        if sha256(a.output/'fixed_protocol.json') != seal or sha256(ROOT/CONFIG) != record['config_sha256']:
            raise ValueError('Fixed comparison changed after labels opened')
        verify_hashes(ROOT,record['code_sha256']); verify_hashes(ROOT,protocol['input_hashes']); verify_hashes(a.artifact_root,frozen)
        for rel,digest in proofs.items():
            if sha256(ROOT/rel/'result.json') != digest: raise ValueError('Source evidence changed during comparison')
        record.update(status='completed_fixed_219_oof_duration_comparison_requires_review',evaluation=evaluation,
            fixed_protocol_sha256=seal,frozen_verified=len(frozen),independent_fusion_constructions=219,
            artifacts={str(v.relative_to(a.output)):sha256(v) for v in a.output.rglob('*') if v.is_file()})
    except BaseException as exc:
        record.update(status='duration_comparison_failed_not_method_rejection',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['seconds'] = time.monotonic()-started
        write_new_json(a.output/'result.json',record)
        print(json.dumps({'status':record['status'],'output':str(a.output)}),flush=True)


if __name__ == '__main__':
    main()
