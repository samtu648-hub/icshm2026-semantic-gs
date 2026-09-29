#!/usr/bin/env python3
"""Seal accepted SEM368 own-fold219 outputs, then score against saved SEM341."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import resource
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from src.quality_protocol import sha256, verify_hashes, write_new_json, CLASSES
from src.quality_static_training import TrainingLabels
from src.quality_semantic_audit import independent_confusion, metrics


def read(p):
    return json.loads(Path(p).read_text())


def boundary_bins(gt):
    # Same preregistered two-sided four-neighbour definition, expressed separately.
    boundary = np.zeros(gt.shape, bool)
    for axis in (0, 1):
        front = [slice(None), slice(None)]
        back = [slice(None), slice(None)]
        front[axis], back[axis] = slice(1, None), slice(None, -1)
        difference = gt[tuple(front)] != gt[tuple(back)]
        boundary[tuple(front)] |= difference
        boundary[tuple(back)] |= difference
    distance = ndi.distance_transform_edt(~boundary) if boundary.any() else np.full(gt.shape, np.inf)
    return np.searchsorted([2, 8, 32], distance, side='left')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=Path, required=True)
    p.add_argument('--artifact-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output = a.output.resolve()
    if not a.output.is_relative_to(ROOT/'outputs/quality/expert_scale'):
        p.error('Isolated scale evaluation output required')
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    config_path = ROOT/'experiments/runs/SEM-368-EXPERT-INPUT-SCALE/config.yaml'
    c = read(config_path)
    record = dict(experiment_id=c['experiment_id'], git_commit=c['git_commit'], seed=c['seed'],
                  dataset_manifest=c['dataset_manifest'], split_id=c['split_id'],
                  config_sha256=sha256(config_path), code_sha256=sha256(Path(__file__)),
                  status='started', new_training=0, new_inference=0, new74_quality_reads=0)
    dependencies = ['src/quality_protocol.py', 'src/quality_static_training.py', 'src/quality_semantic_audit.py']
    record['dependency_sha256'] = {f: sha256(ROOT/f) for f in dependencies}
    for f in ['scripts/evaluate_quality_expert_scale.py', str(config_path.relative_to(ROOT)), *dependencies]:
        dest = a.output/'code'/f
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/f, dest)
    write_new_json(a.output/'started.json', record)
    try:
        queue = read(a.batch/'queue/result.json')
        if queue['status'] != 'completed_mask2former_threefold_acceptance_pending_comparison' or queue['config_sha256'] != record['config_sha256']:
            raise ValueError('Complete accepted threefold queue required before labels')
        verify_hashes(ROOT, queue['code_sha256'])
        files = {}; ownfold = {}; source = {}
        for fold, base in c['baseline_roles'].items():
            oldfolder = ROOT/base['run']
            verify_hashes(oldfolder, {'result.json': base['result_sha256']})
            oldrecord = read(oldfolder/'result.json')
            folder = a.batch/('full-'+fold)
            r = read(folder/'result.json')
            audit = read(a.batch/('audit-'+fold)/'result.json')
            if (r['access'] != base['access'] or r['trained_steps'] != 6000 or r['binding']['network_input'] != 864
                    or audit['status'] != 'completed_mask2former_fold_acceptance' or not audit['two_fresh_exact']
                    or audit['run_result_sha256'] != sha256(folder/'result.json')):
                raise ValueError('Original roles or fresh acceptance differ')
            if (r['sampling_sha256'] != oldrecord['sampling_sha256']
                    or r['sampled_crop_class_counts'] != oldrecord['sampled_crop_class_counts']):
                raise ValueError('Native sample trace or class exposure not exactly matched to SEM341')
            verify_hashes(folder, r['artifacts'])
            source[fold] = (oldfolder, oldrecord, folder, r)
            for name in r['prediction_views']:
                if name in ownfold or name in r['access']['train']:
                    raise ValueError('Repeated or leaking target')
                ownfold[name] = fold
                for rel in ['predictions/'+name, 'probabilities/'+Path(name).stem+'.npz']:
                    files[str((folder/rel).resolve().relative_to(a.batch.resolve()))] = r['artifacts'][rel]
        diag_config = read(ROOT/'experiments/runs/SEM-367-EXPERT-ERROR-STRATA/config.yaml')
        if sorted(ownfold) != diag_config['views'] or set(ownfold) & set(diag_config['forbidden_target_views']):
            raise ValueError('Not exact original219 or forbidden74 included')
        # All candidate predictions are sealed before constructing any label reader.
        write_new_json(a.output/'predictions_sealed.json', dict(files=files, views=sorted(ownfold), unix=time.time()))
        labels = TrainingLabels(a.artifact_root/'r2/sem_001/labels', sorted(ownfold),
                               read(ROOT/'experiments/runs/SEM-001/label_manifest.json'), [1320, 989])
        previous = {r['name']: r for r in read(ROOT/'outputs/quality/expert_diagnostic/SEM-367-r1/diagnosis.json')['rows']}
        rows = []
        for name, fold in sorted(ownfold.items()):
            oldfolder, old, folder, r = source[fold]
            verify_hashes(oldfolder, {'predictions/'+name: old['artifacts']['predictions/'+name]})
            gt = labels.read(name)
            with Image.open(oldfolder/'predictions'/name) as im:
                baseline = np.array(im)
            with Image.open(folder/'predictions'/name) as im:
                candidate = np.array(im)
            cms = [independent_confusion(gt, pred) for pred in [baseline, candidate]]
            if cms[0].tolist() != np.array(previous[name]['boundary_counts'])[..., 0].sum(2).tolist():
                raise ValueError('Saved baseline confusion not exact SEM367')
            cube = np.bincount((gt.astype(np.int64)*25+baseline.astype(np.int64)*5+candidate).ravel(), minlength=125).reshape(5, 5, 5)
            if not np.array_equal(cube.sum(2), cms[0]) or not np.array_equal(cube.sum(1), cms[1]):
                raise ValueError('Transition cube conservation')
            bins = boundary_bins(gt)
            boundary = []
            for t in range(5):
                boundary.append([[int(np.count_nonzero((gt == t) & (bins == b) & m)) for m in
                                  [np.ones(gt.shape, bool), baseline != gt, candidate != gt]] for b in range(4)])
            if np.array(boundary)[..., :2].tolist() != np.array(previous[name]['boundary_counts']).sum(1)[..., :2].tolist():
                raise ValueError('Independent boundary definition differs from SEM367')
            components = []
            for t in range(1, 5):
                mask = gt == t
                cc, n = ndi.label(mask, np.ones((3, 3)))
                if not n:
                    continue
                ids = np.arange(1, n+1)
                area = np.bincount(cc.ravel())[1:]
                rad = ndi.maximum(ndi.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1], cc, ids)
                error = [ndi.sum(pred != gt, cc, ids).astype(np.int64) for pred in [baseline, candidate]]
                for ab in range(4):
                    for rb in range(4):
                        keep = (np.searchsorted([64, 1024, 16384], area) == ab) & (np.searchsorted([2, 4, 8], rad) == rb)
                        components.append([t, ab, rb, int(keep.sum()), int(area[keep].sum()), *[int(e[keep].sum()) for e in error]])
            if [v[:6] for v in components] != [v[:6] for v in previous[name]['components']]:
                raise ValueError('Baseline component partition differs from SEM367')
            row = dict(name=name, fold=fold, baseline=cms[0].tolist(), candidate=cms[1].tolist(),
                       transition=cube.tolist(), boundary=boundary, components=components)
            rows.append(row)
        oldroles = read(ROOT/'outputs/quality/mask2former/SEM-341-r1/comparison/result.json')['roles']
        groups = {'all219': sorted(ownfold), **{f: [n for n in ownfold if ownfold[n] == f] for f in source},
                  'historical108': oldroles['calibration'], 'historical111': oldroles['check']}
        summary = {}
        for group, names in groups.items():
            part = [r for r in rows if r['name'] in names]
            if len(part) != len(names):
                raise ValueError('Historical subgroup mismatch')
            cube = np.sum([np.array(r['transition']) for r in part], axis=0)
            mats = [cube.sum(2), cube.sum(1)]
            ms = [metrics(m) for m in mats]
            for m, metric in zip(mats, ms):
                values = [Fraction(int(m[t, t]), int(m[t].sum()+m[:, t].sum()-m[t, t])) for t in range(5) if m[t].sum()+m[:, t].sum()-m[t,t]]
                if abs(float(sum(values)/len(values))-metric['mIoU_with_background']) > 1e-14:
                    raise ValueError('Rational mIoU mismatch')
            corrected = [int(sum(cube[t, q, t] for q in range(5) if q != t)) for t in range(5)]
            harmed = [int(sum(cube[t, t, q] for q in range(5) if q != t)) for t in range(5)]
            summary[group] = dict(views=len(part), baseline=ms[0], candidate=ms[1],
                delta_miou_pp=100*(ms[1]['mIoU_with_background']-ms[0]['mIoU_with_background']),
                corrected_by_class=corrected, harmed_by_class=harmed,
                boundary=np.sum([np.array(r['boundary']) for r in part], axis=0).tolist())
        write_new_json(a.output/'per_view.json', rows)
        write_new_json(a.output/'summary.json', summary)
        verify_hashes(a.batch, files)
        verify_hashes(ROOT, record['dependency_sha256'])
        record.update(status='completed_fixed_219_evaluation', views=219,
                      primary_delta_miou_pp=summary['all219']['delta_miou_pp'],
                      per_view_sha256=sha256(a.output/'per_view.json'), summary_sha256=sha256(a.output/'summary.json'))
    except BaseException as exc:
        record.update(status='failed_fixed_219_evaluation', error=repr(exc))
        raise
    finally:
        record.update(seconds=time.monotonic()-start, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        write_new_json(a.output/'result.json', record)


if __name__ == '__main__':
    main()
