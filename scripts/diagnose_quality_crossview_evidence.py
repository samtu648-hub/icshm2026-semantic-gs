#!/usr/bin/env python3
"""SEM339: source-only evidence, fresh independent audit, calibration-only description."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
import numpy as np
import torch
from run_quality_crossview_migration import Context, load_plan, map_reader
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.quality_crossview_migration import SourceLabels, class_map, verify_trace
from src.quality_crossview_evidence import (STATES, source_sample, evidence_states,
    source_status, counted, compare_evidence, snapshot_real_sources)
from src.semantic_source_boundary import foundation_distances, sample_boundary_distance
from src.quality_static_training import TrainingLabels
from src.quality_semantic_audit import independent_confusion, metrics

EXP = 'SEM-339-CROSSVIEW-EVIDENCE-DIAG'
CONFIG = 'experiments/runs/'+EXP+'/config.yaml'


class Inputs:
    def __init__(self, args):
        self.args = args; self.cfg = json.loads((ROOT/CONFIG).read_text()); c = self.cfg
        if c['experiment_id'] != EXP or sha256(ROOT/c['parent_config']) != c['parent_config_sha256']:
            raise ValueError('Different registered parent')
        self.parent_root = ROOT/c['parent_root']
        self.parents = {}
        for stage, digest in c['parent_results_sha256'].items():
            p = self.parent_root/stage/'result.json'
            if sha256(p) != digest: raise ValueError('Parent result changed: '+stage)
            self.parents[stage] = json.loads(p.read_text())
        if (self.parents['batch']['status'] != 'completed_crossview_migration_batch_requires_review'
                or self.parents['batch']['score_sha256'] != c['parent_results_sha256']['score']
                or self.parents['audit']['prediction_sha256'] != c['parent_results_sha256']['predict']
                or self.parents['score']['audit_sha256'] != c['parent_results_sha256']['audit']):
            raise ValueError('Parent completion/audit chain differs')
        old_args = argparse.Namespace(batch=self.parent_root, data_root=args.data_root, artifact_root=args.artifact_root)
        self.ctx = Context(old_args); self.plan = load_plan(self.ctx)
        self.names = self.ctx.roles['calibration']
        if len(self.names) != 108 or set(self.names) & set(self.ctx.roles['check']):
            raise ValueError('Not exact calibration scope')
        self.records = {r['target']:r for r in self.parents['predict']['predictions']}
        verify_hashes(self.parent_root/'predict', self.parents['predict']['artifacts'])
        self.maps = map_reader(self.ctx, self.parents['match'])
        self.sources = SourceLabels(self.ctx.folds, args.artifact_root/self.ctx.cfg['labels'],
                                   self.ctx.labels, self.cfg['resolution'])
        self.distance_fields = {}

    def finish(self):
        self.ctx.finish(); load_plan(self.ctx)
        for stage, digest in self.cfg['parent_results_sha256'].items():
            if sha256(self.parent_root/stage/'result.json') != digest:
                raise ValueError('Parent changed during diagnostic')

    def source(self, rec, prefix):
        name = rec[prefix+'_source']
        if name is None: return None
        pose = self.ctx.poses[name]
        return {'name': name, 'pose': pose, 'camera': self.ctx.cameras[pose.camera_id],
                'labels': self.sources.read(rec['fold'], rec['target'], name),
                'maps': self.maps(rec[prefix+'_pair_index'])}

    def prediction(self, name, arm):
        if arm == 'baseline':
            path, digest = self.ctx.base_path(name)
        else:
            rel = arm+'/'+name; path = self.parent_root/'predict'/rel
            digest = self.parents['predict']['artifacts'][rel]
        return class_map(path, digest, (989, 1320))


def point_list(indices, shape):
    yy, xx = np.unravel_index(indices, shape)
    return np.stack((xx+.5, yy+.5), -1)[None]


def one_view(inp, name, independent=False):
    rec = inp.records[name]; ctx = inp.ctx
    pose = ctx.poses[name]; camera = ctx.cameras[pose.camera_id]; filters = ctx.cfg['filters']
    baseline, dense, current = (inp.prediction(name, arm) for arm in ('baseline', 'dense', 'veto'))
    first, second = inp.source(rec, 'first'), inp.source(rec, 'second')
    state = np.full(current.shape, 255, np.uint8)
    indices = np.flatnonzero(current == 0)
    chunk = inp.cfg['resource']['points_per_chunk']
    for start in range(0, len(indices), chunk):
        part = indices[start:start+chunk]; pts = point_list(part, current.shape)
        a = source_sample(pts, first, pose, camera, filters, independent)
        b = source_sample(pts, second, pose, camera, filters, independent)
        codes = evidence_states(a, b, first is not None, second is not None)
        state.ravel()[part] = codes.ravel()
    attempted = np.flatnonzero(baseline != dense).astype(np.uint32)
    potential = np.flatnonzero(state == 4).astype(np.uint32)
    result = {'background_state': state, 'foundation_index': attempted, 'potential_deck_index': potential}
    samples = {}
    for purpose, selected in (('foundation', attempted), ('potential_deck', potential)):
        pts = point_list(selected, current.shape)
        for prefix, source in (('first', first), ('second', second)):
            sample = source_sample(pts, source, pose, camera, filters, independent, details=True)
            samples[(purpose, prefix)] = sample
            for key, value in sample.items():
                result[purpose+'_'+prefix+'_'+key] = value.reshape(-1, 2) if key == 'xy' else value.ravel()
    # Rebuild the parent's exact attempted-deletion trace, without target truth.
    trace = {'flat_index': attempted, 'old_class': baseline.ravel()[attempted],
             'dense_class': dense.ravel()[attempted], 'veto_class': current.ravel()[attempted]}
    for prefix in ('first', 'second'):
        sample = samples[('foundation', prefix)]
        trace[prefix+'_source_xy'] = sample['xy'].reshape(-1, 2)
        trace[prefix+'_reliable'] = sample['reliable'].ravel()
        trace[prefix+'_label'] = sample['label'].ravel()
    path = inp.parent_root/'predict/evidence'/Path(name).with_suffix('.npz')
    with np.load(path, allow_pickle=False) as z: verify_trace(dict(z), trace)
    result['foundation_final_class'] = trace['veto_class']
    result['foundation_second_status'] = source_status(samples[('foundation', 'second')], second is not None).ravel()
    distances = np.empty(0, np.float64)
    if attempted.size:
        if first is None: raise ValueError('Deletion without first source')
        if first['name'] not in inp.distance_fields:
            inp.distance_fields[first['name']] = foundation_distances(first['labels'])
        labels, distances, valid = sample_boundary_distance(samples[('foundation', 'first')]['xy'], first['labels'],
            inp.distance_fields[first['name']], independent)
        if not valid.all() or (labels != 0).any(): raise ValueError('Invalid deletion-boundary sample')
    result['foundation_source_distance'] = distances.ravel()
    for prefix in ('first', 'second'):
        sample = samples[('potential_deck', prefix)]
        if not sample['reliable'].all() or (sample['label'] != 1).any():
            raise ValueError('Potential deck evidence not jointly reliable')
    return result


def load_own(args, stage):
    path = args.output_root/stage/'result.json'
    result = json.loads(path.read_text())
    if result.get('experiment_id') != EXP or result.get('config_sha256') != sha256(ROOT/CONFIG) or 'failed' in result['status']:
        raise ValueError('Unaccepted diagnostic '+stage)
    verify_hashes(args.output_root/'evidence/source_snapshot', result['source_sha256'])
    verify_hashes(ROOT, result['source_sha256'])
    return result


def evidence_stage(inp, output, audit=False):
    args = inp.args
    old = load_own(args, 'evidence') if audit else None
    if old:
        verify_hashes(args.output_root/'evidence', old['artifacts'])
        sources = old['source_sha256']
    else:
        sources = snapshot_real_sources(ROOT, output/'source_snapshot',
            [CONFIG, inp.cfg['parent_config'], str(Path(__file__).resolve().relative_to(ROOT)), 'src/quality_crossview_evidence.py'],
            [getattr(module, '__file__', None) for module in tuple(sys.modules.values())])
    write_new_json(output/'access_sealed.json', {'targets': inp.names, 'target_role': 'historical108 calibration',
        'selected_sources': [{k: inp.records[n][k] for k in ('target', 'fold', 'first_source', 'second_source')} for n in inp.names],
        'target_labels_decoded': False, 'source_sha256': sources, 'parent_results_sha256': inp.cfg['parent_results_sha256']})
    rows = []
    for name in inp.names:
        values = one_view(inp, name, independent=audit)
        rel = Path(name).with_suffix('.npz').name
        if audit:
            with np.load(args.output_root/'evidence'/rel, allow_pickle=False) as z: compare_evidence(dict(z), values)
        else:
            np.savez_compressed(output/rel, **values)
        state = values['background_state']; rec = inp.records[name]
        row = {'target': name, 'fold': rec['fold'], 'first_source': rec['first_source'], 'second_source': rec['second_source'],
               'foundation_attempts': len(values['foundation_index']), 'potential_deck': len(values['potential_deck_index']),
               'background_states': np.bincount(state[state != 255], minlength=len(STATES)).tolist()}
        if audit and row != old['targets'][len(rows)]: raise ValueError('Fresh descriptive totals differ')
        rows.append(row)
        if len(rows)%9 == 0: print(json.dumps({'stage': 'audit' if audit else 'evidence', 'views': len(rows), 'total': 108}), flush=True)
    if audit and inp.sources.access != old['source_label_access']: raise ValueError('Source role access differs')
    return {'status': 'completed_independent_source_evidence_audit' if audit else 'completed_source_only_evidence',
            'source_sha256': sources, 'targets': rows, 'source_label_access': inp.sources.access,
            'evidence_sha256': sha256(args.output_root/'evidence/result.json') if audit else None,
            'target_labels_decoded': 0, 'independent_native_and_boundary': audit}


def describe(inp, output):
    args = inp.args; evidence = load_own(args, 'evidence'); audit = load_own(args, 'audit')
    if audit['status'] != 'completed_independent_source_evidence_audit' or audit['evidence_sha256'] != sha256(args.output_root/'evidence/result.json'):
        raise ValueError('Fresh full evidence audit required before target labels')
    verify_hashes(args.output_root/'evidence', evidence['artifacts'])
    write_new_json(output/'target_scoring_sealed.json', {'target_names': inp.names, 'selection': False,
        'evidence_sha256': sha256(args.output_root/'evidence/result.json'), 'audit_sha256': sha256(args.output_root/'audit/result.json')})
    labels = TrainingLabels(args.artifact_root/inp.ctx.cfg['labels'], inp.names, inp.ctx.labels, inp.cfg['resolution'])
    rows = []; limits = inp.cfg['boundary_upper_bounds_source_pixels']; conf_limits = inp.cfg['confidence_upper_bounds']
    matrices = {k: np.zeros((5, 5), np.int64) for k in ('baseline', 'dense', 'veto')}
    for name in inp.names:
        with np.load(args.output_root/'evidence'/Path(name).with_suffix('.npz'), allow_pickle=False) as z:
            values = dict(z)
        gt = labels.read(name); baseline = inp.prediction(name, 'baseline'); dense = inp.prediction(name, 'dense'); veto = inp.prediction(name, 'veto')
        for arm, prediction in (('baseline', baseline), ('dense', dense), ('veto', veto)):
            matrices[arm] += independent_confusion(gt, prediction)
        idx = values['foundation_index']; truth = gt.ravel()[idx]; final = values['foundation_final_class'] == 0
        distance = values['foundation_source_distance']; bins = np.searchsorted(limits, distance, side='left')
        confidence = np.minimum(values['foundation_first_forward_confidence'], values['foundation_first_reverse_confidence'])
        cbins = np.searchsorted(conf_limits, confidence, side='left')
        state = values['background_state']; bg = state != 255
        if not np.array_equal(bg, veto == 0): raise ValueError('Background evidence population differs')
        # No radius or threshold is selected, and no new semantic map is emitted.
        arrays = {'foundation_dense_distance_gt': counted(bins, truth, len(limits)+1),
                  'foundation_final_distance_gt': counted(bins[final], truth[final], len(limits)+1),
                  'foundation_vetoed_distance_gt': counted(bins[~final], truth[~final], len(limits)+1),
                  'foundation_final_second_status_gt': counted(values['foundation_second_status'][final], truth[final], 8),
                  'foundation_final_confidence_gt': counted(cbins[final], truth[final], len(conf_limits)+1),
                  'background_state_gt': counted(state[bg], gt[bg], len(STATES)),
                  'background_base0_state_gt': counted(state[bg & (baseline == 0)], gt[bg & (baseline == 0)], len(STATES)),
                  'background_deleted4_state_gt': counted(state[bg & (baseline == 4)], gt[bg & (baseline == 4)], len(STATES))}
        potential = values['potential_deck_index']
        pc = np.minimum.reduce([values['potential_deck_'+p+'_'+k] for p in ('first', 'second') for k in ('forward_confidence', 'reverse_confidence')])
        arrays['potential_deck_confidence_gt'] = counted(np.searchsorted(conf_limits, pc, side='left'), gt.ravel()[potential], len(conf_limits)+1)
        if not np.array_equal(arrays['background_state_gt'][4], arrays['potential_deck_confidence_gt'].sum(0)):
            raise ValueError('Potential support population differs')
        row = {'target': name, 'fold': inp.records[name]['fold'], 'counts': {k:v.tolist() for k,v in arrays.items()}}
        rows.append(row)
    def aggregate(items):
        return {key: np.sum([r['counts'][key] for r in items], axis=0).tolist() for key in rows[0]['counts']}
    summary = aggregate(rows)
    for arm, matrix in matrices.items():
        if metrics(matrix) != inp.parents['score']['evaluation']['calibration']['metrics'][arm]:
            raise ValueError('Parent calibration quality not independently reproduced')
    for key, contrast in (('foundation_dense_distance_gt', 'dense_vs_baseline'), ('foundation_final_distance_gt', 'veto_vs_baseline')):
        byclass = np.sum(summary[key], axis=0)
        old = inp.parents['score']['evaluation']['calibration']['changes'][contrast]
        if (int(byclass[0]), int(byclass[4]), int(byclass[1:4].sum())) != (old['corrected'], old['harmed'], old['wrong_to_wrong']):
            raise ValueError('Parent corrected/harmed counts not reproduced')
    write_new_json(output/'per_view.json', rows)
    return {'status': 'completed_calibration_evidence_diagnostic_no_prediction_change', 'summary': summary,
            'per_fold': {str(f): aggregate([r for r in rows if r['fold'] == f]) for f in sorted({r['fold'] for r in rows})},
            'state_names': STATES, 'second_source_status_names': ['no_source', 'outside_source_grid', 'unreliable', 'background', 'deck', 'stay_cable', 'tower', 'foundation'],
            'target_label_access': sorted(labels.accessed), 'source_sha256': evidence['source_sha256'],
            'evidence_sha256': sha256(args.output_root/'evidence/result.json'), 'audit_sha256': sha256(args.output_root/'audit/result.json'),
            'parent_calibration_confusions_and_changes_exact': True, 'independent_counts_exact': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('evidence', 'audit', 'score', 'batch'), required=True)
    for key in ('data-root', 'artifact-root', 'output-root'): parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    for key in ('data_root', 'artifact_root', 'output_root'): setattr(args, key, getattr(args, key).resolve())
    output = args.output_root/args.mode; output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); started = time.monotonic(); inp = None
    result = {'experiment_id': EXP, 'config_sha256': sha256(ROOT/CONFIG), 'git_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
              'seed': 20260805, 'mode': args.mode, 'status': 'initializing', 'training': False, 'new_matching': False,
              'semantic_exports': False, 'threshold_search': False, 'check_target_decodes': 0, 'developer_target_decodes': 0, 'automatic_promotion': False}
    write_new_json(output/'started.json', result)
    try:
        inp = Inputs(args)
        if shutil.disk_usage(output).free < inp.cfg['resource']['minimum_free_disk_gib']*2**30: raise RuntimeError('Disk reserve')
        if args.mode == 'batch':
            events = []
            for stage in ('evidence', 'audit', 'score'):
                argv = [sys.executable, str(Path(__file__).resolve()), '--mode', stage, '--data-root', str(args.data_root),
                        '--artifact-root', str(args.artifact_root), '--output-root', str(args.output_root)]
                with (output/(stage+'.log')).open('x') as log:
                    start = time.time(); done = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                        timeout=inp.cfg['resource']['maximum_seconds_per_stage'])
                event = {'stage': stage, 'started_unix': start, 'ended_unix': time.time(), 'returncode': done.returncode}
                write_new_json(output/(stage+'-finished.json'), event); events.append(event)
                if done.returncode: raise RuntimeError('Diagnostic stopped at '+stage)
                print(json.dumps(event), flush=True)
            result.update(status='completed_evidence_diagnostic_batch', events=events, score_sha256=sha256(args.output_root/'score/result.json'))
        elif args.mode in ('evidence', 'audit'):
            result.update(evidence_stage(inp, output, audit=args.mode == 'audit'))
        else:
            result.update(describe(inp, output))
        inp.finish()
        if 'source_sha256' in result: verify_hashes(ROOT, result['source_sha256'])
        result.update(frozen_verified=18, parent_inputs_unchanged=True,
            artifacts={str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file()})
    except BaseException as exc:
        result.update(status='failed_diagnostic_engineering_not_method_rejection', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result['seconds'] = time.monotonic()-started; write_new_json(output/'result.json', result)
        print(json.dumps({k:result[k] for k in ('mode', 'status', 'seconds')}), flush=True)


if __name__ == '__main__': main()
