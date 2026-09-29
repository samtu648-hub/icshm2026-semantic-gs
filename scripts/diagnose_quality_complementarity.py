#!/usr/bin/env python3
"""SEM354: original108 only, saved same-fold proposals; no model inference."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.quality_mask2former_final import FinalContext
from src.quality_static_training import TrainingLabels, semantic_access
from src.quality_semantic_audit import metrics
from src.quality_complementarity import BRANCHES, SCOPES, positive_proposal, diagnose, independent_diagnose
from scripts.diagnose_quality_raw_fusion import read_image, read_probability

CONFIG = 'experiments/runs/SEM-354-COMPLEMENTARITY-ORACLE-DIAG/config.yaml'
CODE = ['scripts/diagnose_quality_complementarity.py', 'src/quality_complementarity.py',
        'scripts/audit_quality_complementarity.py', 'src/quality_protocol.py',
        'src/quality_static_training.py', 'src/quality_semantic_audit.py',
        'src/quality_mask2former_final.py', 'src/quality_experts.py',
        'scripts/diagnose_quality_raw_fusion.py']


def read(path):
    return json.loads(Path(path).read_text())


def checked_image(root, record, relative):
    verify_hashes(root, {relative: record['artifacts'][relative]})
    value = read_image(root/relative)
    if value.shape != (989, 1320) or value.dtype != np.uint8:
        raise ValueError('Expected native single-channel uint8 prediction')
    return value


def sources_and_roles(c, ctx):
    sources, proof = {}, {}
    rgb_access = read(ctx.parent.rgb_run/'access.json')
    for fold, names in c['views'].items():
        expected = semantic_access(ctx.parent.split, ctx.parent.oof, rgb_access, fold)
        sources[fold], proof[fold] = {}, {}
        for kind, rel in c['sources'][fold].items():
            root = ROOT/rel
            rec = read(root/'result.json')
            sources[fold][kind] = (root, rec)
            binding = rec['binding']
            identity = binding.get('rgb_identity', binding.get('identity'))
            if identity != ctx.binding['rgb_identity'] or binding['seed'] != c['seed']:
                raise ValueError('RGB identity or seed mismatch')
            if kind == 'fullframe':
                five = read(ROOT/c['sources'][fold]['five']/'result.json')
                full_config = ROOT/'experiments/runs/SEM-346-FULLFRAME-CONTEXT-DIAG/config.yaml'
                verify_hashes(ROOT, {str(full_config.relative_to(ROOT)): rec['config_sha256'], **rec['code_sha256']})
                full_protocol = read(full_config)
                if (binding['partition'] != fold or sorted(rec['views']) != sorted(names)
                        or binding != five['binding']
                        or full_protocol['batch']+'/full-'+fold != c['sources'][fold]['five']):
                    raise ValueError('Fullframe must reuse exact same-fold Mask2Former')
                access = five['access']
                model_hashes = {'same_as_five': five['artifacts']['model.pt']}
            else:
                access = rec['access']
                if any(access[k] != expected[k] for k in ('partition', 'train', 'prediction', 'guard')):
                    raise ValueError('Mismatched spatial train/prediction/guard roles')
                if kind == 'five':
                    models = ['model.pt']
                elif kind == 'static':
                    models = ['semantic_field.pt']
                elif kind == 'b2':
                    models = ['expert-3000.pt', 'expert-6000.pt']
                else:
                    models = ['expert.pt']
                model_hashes = {n: rec['artifacts'][n] for n in models}
                verify_hashes(root, model_hashes)
                for role in ('class_weight_views', 'labels_allowed_during_fit'):
                    if role in access and set(access[role]) - set(expected['train']):
                        raise ValueError('Class frequency/fit labels escape training role')
            if set(names) & set(access['train']) or not set(names) <= set(access['prediction']):
                raise ValueError('Target label leakage or missing prediction role')
            proof[fold][kind] = {'result': str(root.relative_to(ROOT)/'result.json'),
                                  'result_sha256': sha256(root/'result.json'), 'model_sha256': model_hashes,
                                  'train_count': len(access['train']), 'prediction_count': len(access['prediction']),
                                  'calibration_count': len(names), 'target_train_overlap': [], 'rgb_identity_exact': True}
    crossroot = ROOT/c['crossview']
    prep, pred, audit = [read(crossroot/x/'result.json') for x in ('prepare', 'predict', 'audit')]
    seal = read(crossroot/'predict/source_roles_sealed.json')
    if (prep['identity'] != ctx.binding['rgb_identity'] or seal['folds'] != ctx.parent.oof['folds']
            or sha256(crossroot/'predict/result.json') != audit['prediction_sha256'] or not audit['independent_native_reconstruction']):
        raise ValueError('Crossview provenance or independent audit differs')
    crossrows = {x['target']: x for x in pred['predictions']}
    plans = {x['target']: x for x in prep['plans']}
    for fold, names in c['views'].items():
        train = set(sources[fold]['five'][1]['access']['train'])
        refs = set()
        for name in names:
            row = crossrows[name]
            if row['fold'] != int(fold[-1]) or row['role'] != 'calibration' or plans[name]['fold'] != int(fold[-1]):
                raise ValueError('Crossview target role differs')
            if not set(plans[name]['sources']) <= train:
                raise ValueError('Crossview candidate reference leaks target/guard')
            selected = {row[k] for k in ('first_source', 'second_source') if row[k] is not None}
            if not selected <= train:
                raise ValueError('Crossview chosen reference leaks target/guard')
            refs |= selected
        proof[fold]['crossview'] = {'selected_reference_names': sorted(refs),
            'references_inside_same_fold_train': True, 'historical_check_names_as_sources': sorted(refs & set(prep['roles']['check'])),
            'new_source_label_decodes': 0, 'composite_selection_used_historical108': True}
    leakage = {}
    for outer, names in c['views'].items():
        leakage[outer] = {inner: sorted(set(names) & set(sources[inner]['five'][1]['access']['train']))
                          for inner in c['views'] if inner != outer}
    proof['naive_meta_crossfold_outer_label_exposure'] = leakage
    proof['composite_policy_labels'] = 'm2f_fused and crossview_base/veto policies previously selected using108; diagnostic only'
    return sources, (crossroot, prep, pred), proof


def load_view(c, sources, cross, parent, fold, name):
    source = sources[fold]
    root, rec = source['five']
    five = read_probability(root, rec, 'probabilities/'+Path(name).stem+'.npz', 5)
    comparison = ROOT/c['comparison']
    raw = checked_image(comparison, parent, 'calibration/mask2former_6k/raw/'+name)
    if not np.array_equal(raw, five.argmax(0)):
        raise ValueError('Raw argmax mismatch')
    maps = {'static': checked_image(*source['static'], 'predictions/'+name)}
    for step in (3000, 6000):
        maps[f'b2_{step//1000}k'] = checked_image(*source['b2'], f'step-{step}/predictions/'+name)
    for kind, cid in [('deck', 1), ('stay_cable', 2)]:
        maps[kind+'_positive'] = positive_proposal(checked_image(*source[kind], 'predictions/'+name), cid)
    root, rec = source['fullframe']
    rel = Path(name).stem+'.npz'
    maps['fullframe'] = read_probability(root, rec, rel, 5).argmax(0).astype(np.uint8)
    maps['m2f_fused'] = checked_image(comparison, parent, 'calibration/mask2former_6k/calibrated/'+name)
    root, prep, pred = cross
    base = prep['baseline'][name]
    verify_hashes(ROOT, {base['path']: base['sha256']})
    maps['crossview_base'] = read_image(ROOT/base['path'])
    maps['crossview_veto'] = checked_image(root/'predict', pred, 'veto/'+name)
    selected = raw.copy()
    selected[(raw == 3) & (maps['static'] == 4) & (five[4] >= .1) & ((five[3]-five[4]) <= .05)] = 4
    return raw, selected, maps, five


def aggregate(rows):
    out = {}
    for control in ('raw', 'selected'):
        cms = sum(np.array(row[control]['confusion'], dtype=np.int64) for row in rows)
        branches = {}
        for key in BRANCHES:
            cube = sum(np.array(row[control]['source_cubes'][key], dtype=np.int64) for row in rows)
            correct = [int(sum(cube[t, p, t] for p in range(5) if p != t)) for t in range(5)]
            harm = [int(sum(cube[t, t, q] for q in range(5) if q != t)) for t in range(5)]
            common = [int(sum(cube[t, p, q] for p in range(5) if p != t for q in range(6) if q != t)) for t in range(5)]
            branches[key] = {'corrected_by_class': correct, 'harmed_by_class_if_all_proposals_taken': harm,
                             'common_error_or_abstain_by_class': common, 'source_cube': cube.tolist()}
        scopes = {}
        for scope in SCOPES:
            repair = sum(np.array(row[control]['scopes'][scope]['recovered_confusion'], dtype=np.int64) for row in rows)
            oracle = cms-repair+np.diag(repair.sum(1))
            scopes[scope] = {'recovered_pixels': int(repair.sum()), 'recovered_confusion': repair.tolist(),
                            'metrics_truth_assisted_only': metrics(oracle),
                            'eligible_pixels': sum(row[control]['scopes'][scope]['eligible_pixels'] for row in rows),
                            'correct_control_pixels_exposed': np.sum([row[control]['scopes'][scope]['correct_control_pixels_exposed'] for row in rows], axis=0).tolist(),
                            'delta_miou_pp': 100*(metrics(oracle)['mIoU_with_background']-metrics(cms)['mIoU_with_background'])}
        joint = np.zeros((5, 5, 512), np.int64)
        for row in rows:
            for gt, base, bits, n in row[control]['wrong_correctness_joint']:
                joint[gt, base, bits] += n
        pair, unique = {}, {}
        for i, key in enumerate(BRANCHES):
            unique[key] = joint[:, :, 1 << i].sum(1).tolist()
            pair[key] = {other: int(joint[:, :, [b for b in range(512) if b & (1 << i) and b & (1 << j)]].sum())
                         for j, other in enumerate(BRANCHES)}
        out[control] = {'metrics': metrics(cms), 'raw_error_pixels': int(cms.sum()-np.trace(cms)),
                        'branches': branches, 'scopes': scopes, 'pair_recoverable_overlap': pair,
                        'unique_recovery_by_class': unique, 'common_error_confusion': joint[:, :, 0].tolist()}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay-from', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT/'outputs/quality/complementarity'):
        parser.error('Use an isolated complementarity directory')
    output.mkdir(parents=True, exist_ok=False)
    c = read(ROOT/CONFIG)
    started = time.monotonic()
    record = {k: c[k] for k in ('experiment_id', 'git_commit', 'seed', 'dataset_manifest', 'split_id')}
    record.update(status='started', started_utc=datetime.now(timezone.utc).isoformat(),
        config_sha256=sha256(ROOT/CONFIG), code_sha256={p: sha256(ROOT/p) for p in CODE},
        new_training=False, new_inference=False, new_predictions=False, learned_fusion_trained=False,
        check_labels_decoded=0, development_labels_decoded=0, truth_assisted_diagnostic_only=True)
    try:
        verify_hashes(ROOT, c['input_hashes'])
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != c['git_commit']:
            raise ValueError('Git commit drift')
        if list(BRANCHES) != c['branches'] or list(SCOPES) != list(c['scopes']):
            raise ValueError('Unregistered sources or oracle scopes')
        write_new_json(output/'started.json', record)
        for path in [CONFIG, *CODE]:
            dst = output/'code'/path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/path, dst)
        ctx = FinalContext(ROOT, args.artifact_root, ROOT/c['final_batch'])
        parent = read(ROOT/c['comparison']/'result.json')
        history = read(ROOT/c['comparison']/'calibration/per_view.json')
        selected_history = read(ROOT/c['selected_calibration_counts'])
        names = [n for ns in c['views'].values() for n in ns]
        if (len(set(names)) != 108 or sorted(names) != sorted(parent['roles']['calibration'])
                or set(names) & set(parent['roles']['check']) or set(names) & set(ctx.access['prediction'])):
            raise ValueError('Wrong calibration membership')
        source, cross, proof = sources_and_roles(c, ctx)
        write_new_json(output/'provenance_before_gt.json', proof)
        labels = TrainingLabels(args.artifact_root/'r2/sem_001/labels', names, ctx.parent.label_manifest, [1320, 989])
        oldrows = read(args.replay_from/'per_view.json') if args.replay_from else None
        if args.replay_from:
            oldrecord = read(args.replay_from/'result.json')
            if oldrecord['config_sha256'] != record['config_sha256'] or oldrecord['code_sha256'] != record['code_sha256']:
                raise ValueError('Replay requires exact original implementation/config seal')
        rows = []
        for fold, foldnames in c['views'].items():
            for name in foldnames:
                if (time.monotonic()-started > c['resources']['maximum_seconds_per_stage']
                        or shutil.disk_usage(ROOT).free/2**30 < c['resources']['minimum_free_gib']):
                    raise OSError('Registered time/disk budget reached')
                raw, selected, maps, five = load_view(c, source, cross, parent, fold, name)
                gt = labels.read(name)
                method = independent_diagnose if args.replay_from else diagnose
                row = {'name': name, 'fold': fold}
                for control, base in [('raw', raw), ('selected', selected)]:
                    row[control] = method(base, raw, maps, five, gt)
                if (row['raw']['confusion'] != history[name]['metrics']['mask2former_6k/raw']['confusion']
                        or row['selected']['confusion'] != selected_history[name]['curves'][1]['confusion']
                        or np.array(row['raw']['source_cubes']['m2f_fused']).sum(1).tolist() != [x+[0] for x in history[name]['metrics']['mask2former_6k/calibrated']['confusion']]):
                    raise ValueError('Historical raw/selected/fused confusion mismatch')
                if oldrows is not None and row != oldrows[len(rows)]:
                    raise ValueError('Independent complete pixel recount differs')
                rows.append(row)
                if len(rows) % 6 == 0:
                    print(json.dumps({'stage': 'replay' if args.replay_from else 'diagnose', 'views': len(rows), 'total': 108}), flush=True)
        if sorted(labels.accessed) != sorted(names):
            raise ValueError('Label access incomplete')
        groups = {f: aggregate([r for r in rows if r['fold'] == f]) for f in c['views']}
        pooled = aggregate(rows)
        ctx.finish()
        verify_hashes(ROOT, c['input_hashes'])
        verify_hashes(ROOT, {CONFIG: record['config_sha256'], **record['code_sha256']})
        write_new_json(output/'per_view.json', rows)
        gain = pooled['raw']['scopes']['all']['delta_miou_pp']
        screen = gain >= c['fusion_screen']['pooled_all_oracle_gain_pp_at_least'] and all(
            g['raw']['scopes']['all']['delta_miou_pp'] >= c['fusion_screen']['each_fold_all_oracle_gain_pp_at_least'] for g in groups.values())
        record.update(status='completed_samefold108_oracle_diagnostic', pooled=pooled, per_fold=groups,
            calibration_labels_decoded=sorted(labels.accessed), rgb_verified=293, frozen_verified=18,
            independent_pixel_recount=bool(args.replay_from), fusion_protocol_headroom_screen=screen,
            per_view_sha256=sha256(output/'per_view.json'), provenance_sha256=sha256(output/'provenance_before_gt.json'))
        if args.replay_from:
            if oldrecord['pooled'] != pooled or oldrecord['per_fold'] != groups:
                raise ValueError('Replay summary differs')
            record['replay_source_sha256'] = sha256(args.replay_from/'result.json')
        size = sum(p.stat().st_size for p in (ROOT/'outputs/quality/complementarity').rglob('*') if p.is_file())
        if size > c['resources']['maximum_total_new_mib']*2**20:
            raise OSError('Output budget exceeded; preserve all results')
    except BaseException as exc:
        record.update(status='failed_complementarity_preserved', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['seconds'] = time.monotonic()-started
        write_new_json(output/'result.json', record)
        print(json.dumps({'status': record['status'], 'seconds': record['seconds']}), flush=True)


if __name__ == '__main__':
    main()
