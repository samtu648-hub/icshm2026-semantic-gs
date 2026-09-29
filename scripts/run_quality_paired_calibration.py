#!/usr/bin/env python3
"""SEM337 fixed paired-model calibration and fresh CPU audit, without GPU training."""
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
from scripts.evaluate_quality_expert_pair import source_index, load_inputs, CODE as PAIR_CODE
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.quality_static_training import TrainingLabels
from src.quality_semantic_audit import independent_confusion, metrics
from src.quality_foundation_policy import replace_foundation, changes
from src.quality_fusion import pipeline, confusion
from src.quality_paired_calibration import select_recall_preserving, independent_map, LabelRelease
from src.semantic_protected_fusion import partition_training_names

CONFIG = 'experiments/runs/SEM-337-PAIRED-FUSION-CALIBRATION/config.yaml'
CODE = tuple(sorted(set((*PAIR_CODE, 'scripts/run_quality_paired_calibration.py', 'src/quality_paired_calibration.py'))))
KEYS = ('fixed_3k', 'fixed_6k', 'adapted_3k', 'adapted_6k')
CONTRASTS = {'duration_fixed': ('fixed_3k', 'fixed_6k'),
    'policy_3k': ('fixed_3k', 'adapted_3k'), 'policy_6k': ('fixed_6k', 'adapted_6k'),
    'duration_adapted': ('adapted_3k', 'adapted_6k'), 'joint_vs_3k': ('fixed_3k', 'adapted_6k')}


class Inputs:
    def __init__(self, artifact_root, output):
        self.cfg = json.loads((ROOT / CONFIG).read_text())
        c = self.cfg
        self.hashes = {CONFIG: sha256(ROOT / CONFIG), **{n: sha256(ROOT/n) for n in CODE}}
        for name in ('pair_config', 'grid_config', 'parent_result'):
            self.hashes[c[name]] = c[name+'_sha256']
        verify_hashes(ROOT, self.hashes)
        self.pc = json.loads((ROOT/c['pair_config']).read_text())
        self.parent = json.loads((ROOT/c['parent_result']).read_text())
        grid_cfg = json.loads((ROOT/c['grid_config']).read_text())
        grid = next(s['thresholds'] for s in grid_cfg['stages'] if s['name'] == 'foundation')
        if grid != c['threshold_grid'] or c['fixed_thresholds'] != self.parent['thresholds'] or c['endpoints'] != [3000, 6000]:
            raise ValueError('Grid or fixed endpoints differ')
        if (self.parent['status'] != 'completed_fixed_paired_219_oof_duration_comparison_requires_review'
                or self.parent['seed'] != c['seed'] or c['pixel_space'] != 'PINHOLE' or c['resolution'] != [1320, 989]):
            raise ValueError('Unaccepted native paired source')
        _, _, self.index, self.proofs = source_index(ROOT, self.pc, ROOT/c['pair_batch'])
        if self.proofs != self.parent['input_result_hashes']:
            raise ValueError('Different source population')
        protocol = json.loads((ROOT/self.pc['protocol']).read_text())
        self.hashes.update(protocol['input_hashes'])
        oof_path = 'experiments/splits/sem_005_oof_spatial_v1.json'
        verify_hashes(ROOT, self.hashes)
        oof = json.loads((ROOT/oof_path).read_text())
        self.hashes[self.pc['label_manifest']] = oof['label_manifest_sha256']
        self.manifest = json.loads((ROOT/self.pc['label_manifest']).read_text())
        split = json.loads((ROOT/'experiments/splits/sem_001_scaffold_static_v1.json').read_text())
        if set(self.index) != set(split['semantic_train_labeled']) or set(self.index) & set(split['semantic_validation_labeled']):
            raise ValueError('OOF target coverage or development isolation error')
        self.roles = partition_training_names(sorted(self.index), expected_count=219)
        if {k: len(v) for k, v in self.roles.items()} != {'calibration': 108, 'check': 111}:
            raise ValueError('Role split differs')
        for role in self.roles:
            if sorted(self.roles[role]) != sorted(self.parent['evaluation'][role]['views']):
                raise ValueError('Parent role mismatch')
        self.frozen = {r['path']: r['sha256'] for r in json.loads((ROOT/protocol['frozen_manifest']).read_text())['artifacts']}
        self.artifact_root = artifact_root
        self.release = LabelRelease(output)
        self.access = {}
        self.finish()

    def labels(self, role):
        self.release.require(role)
        reader = TrainingLabels(self.artifact_root/self.pc['labels'], self.roles[role], self.manifest, self.cfg['resolution'])
        self.access[role] = reader.accessed
        return reader

    def finish(self):
        verify_hashes(ROOT, self.hashes)
        verify_hashes(self.artifact_root, self.frozen)
        for path, digest in self.proofs.items():
            if sha256(ROOT/path/'result.json') != digest:
                raise ValueError('Source result changed')


def fixed_maps(static, probs, thresholds):
    return {key: pipeline(static, probs[branch], probs['stay_cable'], probs['deck'], thresholds)['deck']
        for key, branch in (('fixed_3k', 'fiveclass'), ('fixed_6k', 'sixk'))}


def calculate_curves(source, independent=False):
    grid = source.cfg['threshold_grid']
    matrices = {e: [np.zeros((5, 5), np.int64) for _ in grid] for e in ('3k', '6k')}
    groups = {}
    labels = source.labels('calibration')
    for count, name in enumerate(source.roles['calibration']):
        part, sources = source.index[name]
        static, probs = load_inputs(sources, name)
        base = fixed_maps(static, probs, source.cfg['fixed_thresholds']) if not independent else None
        gt = labels.read(name)
        group = groups.setdefault(part, {e: [np.zeros((5, 5), np.int64) for _ in grid] for e in matrices})
        for endpoint, branch in (('3k', 'fiveclass'), ('6k', 'sixk')):
            for j, threshold in enumerate(grid):
                if independent:
                    prediction = independent_map(static, probs[branch][4], probs['stay_cable'], probs['deck'], threshold)
                else:
                    prediction = replace_foundation(base['fixed_'+endpoint], static, probs[branch][4], threshold)
                matrix = confusion(gt, prediction)
                matrices[endpoint][j] += matrix
                group[endpoint][j] += matrix
        if count % 24 == 0:
            print(json.dumps({'stage': 'audit_curves' if independent else 'calibration', 'views': count+1}), flush=True)
    return {e: [{'threshold': t, 'metrics': metrics(m),
        'per_partition': {part: metrics(g[e][j]) for part, g in groups.items()}}
        for j, (t, m) in enumerate(zip(grid, matrices[e]))] for e in matrices}


def new_totals():
    return {'matrices': {k: np.zeros((5, 5), np.int64) for k in KEYS},
        'changes': {k: dict(changed=0, corrected=0, harmed=0, wrong_to_wrong=0) for k in CONTRASTS}}


def accumulate(total, matrices, costs):
    for k in KEYS:
        total['matrices'][k] += matrices[k]
    for k in costs:
        for n, value in costs[k].items():
            total['changes'][k][n] += value


def summarize(total):
    return {'metrics': {k: metrics(v) for k, v in total['matrices'].items()}, 'changes': total['changes']}


def evaluate(source, selected, output, audit=False):
    evaluation = {}
    for role in ('calibration', 'check'):
        labels = source.labels(role)
        totals, groups, views = new_totals(), {}, {}
        for name in source.roles[role]:
            part, sources = source.index[name]
            static, probs = load_inputs(sources, name)
            fixed = fixed_maps(static, probs, source.cfg['fixed_thresholds']) if not audit else {}
            maps = {}
            for key in KEYS:
                endpoint = key.split('_')[1]
                branch = 'fiveclass' if endpoint == '3k' else 'sixk'
                threshold = .8 if key.startswith('fixed') else selected[endpoint]
                independent = independent_map(static, probs[branch][4], probs['stay_cable'], probs['deck'], threshold)
                actual = independent if audit else (fixed[key] if key.startswith('fixed') else
                    replace_foundation(fixed['fixed_'+endpoint], static, probs[branch][4], threshold))
                if not np.array_equal(actual, independent):
                    raise ValueError('Independent native construction differs')
                maps[key] = actual
                if key.startswith('adapted'):
                    path = output/role/key/name
                    if audit:
                        with Image.open(path) as im:
                            if not np.array_equal(np.asarray(im), actual):
                                raise ValueError('Fresh CPU replay differs from saved prediction')
                    else:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(actual).save(path)
            for endpoint, branch in (('3k', 'fiveclass'), ('6k', 'sixk')):
                old, new = maps['fixed_'+endpoint], maps['adapted_'+endpoint]
                if any(not np.array_equal(old == c, new == c) for c in (1, 2, 3)) or np.any((old != new) & (static != 4)):
                    raise ValueError('Protected class or eligibility changed')
                indices = np.flatnonzero(old != new).astype(np.uint32)
                evidence = {'flat_index': indices, 'old_class': old.ravel()[indices],
                    'new_class': new.ravel()[indices], 'foundation_probability': probs[branch][4].ravel()[indices]}
                path = output/role/('evidence_'+endpoint)/(Path(name).stem+'.npz')
                if audit:
                    with np.load(path, allow_pickle=False) as z:
                        if set(z.files) != set(evidence) or any(not np.array_equal(z[k], v) for k, v in evidence.items()):
                            raise ValueError('Changed pixel evidence differs')
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, **evidence)
            gt = labels.read(name)
            matrices = {k: independent_confusion(gt, p) for k, p in maps.items()}
            costs = {k: changes(maps[a], maps[b], gt) for k, (a, b) in CONTRASTS.items()}
            for c in costs.values():
                if c['changed'] != c['corrected'] + c['harmed'] + c['wrong_to_wrong']:
                    raise ValueError('Modification accounting differs')
            accumulate(totals, matrices, costs)
            accumulate(groups.setdefault(part, new_totals()), matrices, costs)
            views[name] = {'partition': part, 'metrics': {k: metrics(m) for k, m in matrices.items()}, 'changes': costs}
        summary = summarize(totals)
        for k in ('3k', '6k'):
            if summary['metrics']['fixed_'+k] != source.parent['evaluation'][role]['metrics']['full_'+k]:
                raise ValueError('SEM336 paired baseline did not reproduce exactly')
        evaluation[role] = {**summary, 'views': source.roles[role],
            'per_partition': {k: summarize(v) for k, v in groups.items()}}
        path = output/role/'per_view.json'
        if audit:
            if views != json.loads(path.read_text()):
                raise ValueError('Fresh per-view scoring differs')
        else:
            write_new_json(path, views)
        print(json.dumps({'stage': 'audit_evaluation' if audit else 'evaluation', 'role': role,
            'mIoU': {k: v['mIoU_with_background'] for k, v in summary['metrics'].items()}}), flush=True)
    return evaluation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--audit', type=Path, help='Read completed run and write only a new audit result')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    record = {'experiment_id': 'SEM-337-PAIRED-FUSION-CALIBRATION', 'status': 'initializing',
        'config_sha256': sha256(ROOT/CONFIG), 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'seed': 20260805, 'development_labels_decoded': False, 'automatic_promotion': False,
        'gpu_training': False, 'new_gpu_rendering': False, 'whole_chain_spatial_validation': False}
    source = None
    try:
        output = args.audit if args.audit else args.output
        source = Inputs(args.artifact_root, output)
        record.update(input_hashes=source.hashes, input_result_hashes=source.proofs,
            parent_result_sha256=source.cfg['parent_result_sha256'])
        write_new_json(args.output/'started.json', record)
        if args.audit:
            original = json.loads((output/'result.json').read_text())
            if original['status'] != 'completed_paired_calibration_pending_independent_audit' or original['input_hashes'] != source.hashes:
                raise ValueError('Run incomplete or source changed before independent audit')
            verify_hashes(output, original['artifacts'])
            seal = json.loads((output/'calibration_sealed.json').read_text())
            if sha256(output/'calibration_sealed.json') != original['calibration_sealed_sha256']:
                raise ValueError('Calibration seal changed')
            curves = calculate_curves(source, independent=True)
            if curves != seal['curves']:
                raise ValueError('Independent full calibration curves differ')
            reference = curves['3k'][source.cfg['threshold_grid'].index(.8)]['metrics']
            if reference != seal['reference']:
                raise ValueError('Recall reference differs')
            selected = {e: select_recall_preserving(rows, reference)['threshold'] for e, rows in curves.items()}
            if selected != seal['selected']:
                raise ValueError('Calibration selection differs')
            source.release.seal()
            evaluation = evaluate(source, selected, output, audit=True)
            if evaluation != original['evaluation']:
                raise ValueError('Independent complete scoring differs')
            verify_hashes(output, original['artifacts'])
            record.update(status='completed_independent_paired_calibration_audit',
                run_result_sha256=sha256(output/'result.json'), views_replayed=219,
                predictions_replayed=438, pixel_evidence_replayed=438, all_calibration_curves_recounted=True)
        else:
            if shutil.disk_usage(args.output).free < source.cfg['minimum_free_gib']*1024**3:
                raise ValueError('Less than registered free disk reserve')
            for name in (*CODE, CONFIG):
                dest = args.output/'code'/name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT/name, dest)
            write_new_json(args.output/'protocol_sealed.json', {**record, 'config': source.cfg,
                'roles': source.roles, 'current_run_target_labels_decoded': False,
                'evidence_reason': 'Restore/reject only originally-static foundation using sealed endpoint threshold; labels are never a decision input. Pixel indices row-major on1320x989 PINHOLE; probabilities/source result bindings identify evidence.'})
            curves = calculate_curves(source)
            reference = curves['3k'][source.cfg['threshold_grid'].index(.8)]['metrics']
            if reference != source.parent['evaluation']['calibration']['metrics']['full_3k']:
                raise ValueError('Recall reference not paired3k')
            selected = {e: select_recall_preserving(rows, reference)['threshold'] for e, rows in curves.items()}
            write_new_json(args.output/'calibration_sealed.json', {'curves': curves, 'reference': reference,
                'selected': selected, 'config_sha256': record['config_sha256'],
                'current_run_check_labels_decoded': False, 'development_labels_decoded': False,
                'calibration_views': sorted(source.access['calibration']), 'source_hashes': source.proofs})
            source.release.seal()
            print(json.dumps({'stage': 'calibration_sealed', 'selected': selected,
                'reference_recall': reference['per_class']['foundation']['recall']}), flush=True)
            evaluation = evaluate(source, selected, args.output)
            record.update(status='completed_paired_calibration_pending_independent_audit',
                selected=selected, evaluation=evaluation, calibration_sealed_sha256=source.release.seal_hash)
        source.release.require('check')
        source.finish()
        record.update(frozen_verified=len(source.frozen), rgb_source_artifacts_unchanged=True,
            target_label_access={r: sorted(names) for r, names in source.access.items()},
            artifacts={str(f.relative_to(args.output)): sha256(f) for f in args.output.rglob('*') if f.is_file()})
    except BaseException as exc:
        record.update(status='paired_calibration_failed_not_method_rejection', error=f'{type(exc).__name__}: {exc}')
        if source is not None:
            record['target_label_access'] = {r: sorted(names) for r, names in source.access.items()}
        raise
    finally:
        record['seconds'] = time.monotonic()-start
        write_new_json(args.output/'result.json', record)
        print(json.dumps({'status': record['status'], 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
