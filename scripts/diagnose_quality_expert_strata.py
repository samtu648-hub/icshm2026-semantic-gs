#!/usr/bin/env python3
"""SEM367: saved own-fold predictions and exact historical crop exposure; no fit."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import resource
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy import ndimage as ndi
from scripts.diagnose_quality_oof_selector_bottleneck import provenance
from scripts.run_quality_selector_weight_ablation import sources, read
from scripts.diagnose_quality_complementarity import checked_image
from scripts.diagnose_quality_raw_fusion import read_probability
from scripts.run_quality_final_selector import freeze
from src.quality_static_training import TrainingLabels
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.quality_selector_bottleneck import source_counts

CONFIG = 'experiments/runs/SEM-367-EXPERT-ERROR-STRATA/config.yaml'


def boundary_bins(gt):
    boundary = np.zeros(gt.shape, bool)
    dy = gt[1:] != gt[:-1]
    dx = gt[:, 1:] != gt[:, :-1]
    boundary[1:] |= dy
    boundary[:-1] |= dy
    boundary[:, 1:] |= dx
    boundary[:, :-1] |= dx
    distance = ndi.distance_transform_edt(~boundary) if boundary.any() else np.full(gt.shape, np.inf)
    return np.searchsorted([2, 8, 32], distance, side='left')


def crop_coverage(samples, shape=(989, 1320), crop=768):
    delta = np.zeros((shape[0]+1, shape[1]+1), np.int32)
    for s in samples:
        y, x = s['top'], s['left']
        if not (0 <= y <= shape[0]-crop and 0 <= x <= shape[1]-crop):
            raise ValueError('Crop outside native grid')
        delta[y, x] += 1
        delta[y+crop, x] -= 1
        delta[y, x+crop] -= 1
        delta[y+crop, x+crop] += 1
    counts = delta.cumsum(0).cumsum(1)[:-1, :-1]
    if counts.min() < 0 or counts.max() > len(samples) or counts.sum() != len(samples)*crop*crop:
        raise ValueError('Rectangle exposure conservation failed')
    return counts


def stratified_counts(gt, pred, bins, common, high):
    # GT x prediction x stratum x status(total/raw-error/common-error/high-confidence-error)
    key = (gt.astype(np.int64)*5+pred)*4+bins
    masks = [np.ones(gt.shape, bool), gt != pred, common, (gt != pred) & high]
    out = np.stack([np.bincount(key[m], minlength=100).reshape(5, 5, 4) for m in masks], -1)
    # Independent class masks catch indexing/marginal mistakes without a duplicate heavy pass.
    for t in range(5):
        for b in range(4):
            mask = (gt == t) & (bins == b)
            expected = [np.count_nonzero(mask & m) for m in masks]
            if out[t, :, b].sum(0).tolist() != expected:
                raise ValueError('Independent stratum marginal mismatch')
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    a.output = a.output.resolve()
    if not a.output.is_relative_to(ROOT/'outputs/quality/expert_diagnostic'):
        parser.error('Use a new isolated expert diagnostic output')
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    c = read(ROOT/CONFIG)
    pc = read(ROOT/c['parent_config'])
    code = [str(Path(__file__).resolve().relative_to(ROOT)),
            'scripts/diagnose_quality_oof_selector_bottleneck.py',
            'scripts/run_quality_selector_weight_ablation.py',
            'scripts/diagnose_quality_complementarity.py', 'scripts/diagnose_quality_raw_fusion.py',
            'scripts/run_quality_final_selector.py', 'src/quality_static_training.py',
            'src/quality_protocol.py', 'src/quality_selector_bottleneck.py']
    rec = {k: c[k] for k in ['experiment_id', 'git_commit', 'seed', 'dataset_manifest', 'split_id']}
    rec.update(config_sha256=sha256(ROOT/CONFIG), code_sha256={f: sha256(ROOT/f) for f in code},
               status='started', new_training=0, new_inference=0, new_74_quality_reads=0)
    write_new_json(a.output/'started.json', rec)
    for f in [CONFIG, *code]:
        dst = a.output/'code'/f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/f, dst)
    try:
        verify_hashes(ROOT, c['input_hashes'])
        if c['views'] != pc['views'] or c['forbidden_target_views'] != pc['forbidden_target_views']:
            raise ValueError('Diagnostic label role drift')
        folds, bindings, frozen = provenance(pc, a.artifact_root)
        labels = TrainingLabels(a.artifact_root/'r2/sem_001/labels', c['views'],
                               read(ROOT/'experiments/runs/SEM-001/label_manifest.json'), [1320, 989])
        old = {r['name']: r for r in read(ROOT/'outputs/quality/selector_bottleneck/SEM-364-r1/diagnosis.json')['rows']}
        src = {f: sources(pc, f) for f in pc['sources']}
        logs = {}; samples = {}; train = {}; coverage_totals = {}
        for fold, s in src.items():
            folder, r = s['five']
            verify_hashes(folder, {'training.jsonl': r['artifacts']['training.jsonl']})
            lines = [json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
            if [v['step'] for v in lines] != list(range(1, 6001)):
                raise ValueError('Successful update sequence drift')
            groups = defaultdict(list)
            for line in lines:
                groups[line['sample']['name']].append(line['sample'])
            if set(groups) != set(r['access']['train']):
                raise ValueError('Training sample roles differ')
            samples[fold] = groups
            logs[fold] = sha256(folder/'training.jsonl')
            train[fold] = []
            coverage_totals[fold] = np.zeros(5, np.int64)
        rows = []
        for i, name in enumerate(c['views']):
            gt = labels.read(name)
            fold = folds[name]
            s = src[fold]
            prob = read_probability(*s['five'], f'probabilities/{Path(name).stem}.npz', 5)
            raw = checked_image(*s['five'], 'predictions/'+name)
            if not np.array_equal(prob.argmax(0), raw):
                raise ValueError('Raw PNG and saved probability differ')
            maps = [checked_image(*s['static'], 'predictions/'+name)]
            maps += [checked_image(*s['b2'], f'step-{step}/predictions/{name}') for step in [3000, 6000]]
            for kind, cid in [('deck', 1), ('stay_cable', 2)]:
                m = checked_image(*s[kind], 'predictions/'+name)
                maps.append(np.where(m == 1, cid, 5))
            hist, _, recover = source_counts(gt, raw, maps)
            if hist.tolist() != old[name]['source_error_mask_counts']:
                raise ValueError('SEM364 exact common/recoverable counts differ')
            common = (raw != gt) & ~recover.reshape(gt.shape)
            high = prob.max(0) >= c['statistics']['confidence_cut']
            bins = boundary_bins(gt)
            bc = stratified_counts(gt, raw, bins, common, high)
            components = []
            for t in range(1, 5):
                mask = gt == t
                cc, n = ndi.label(mask, np.ones((3, 3)))
                if not n:
                    continue
                ids = np.arange(1, n+1)
                area = np.bincount(cc.ravel())[1:]
                # Pad the frame for a defined inradius for truncated components.
                edt = ndi.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
                radius = ndi.maximum(edt, cc, ids)
                errors = ndi.sum(raw != gt, cc, ids).astype(np.int64)
                commons = ndi.sum(common, cc, ids).astype(np.int64)
                for area_bin in range(4):
                    for radius_bin in range(4):
                        keep = ((np.searchsorted([64, 1024, 16384], area, side='left') == area_bin) &
                                (np.searchsorted([2, 4, 8], radius, side='left') == radius_bin))
                        components.append([t, area_bin, radius_bin, int(keep.sum()), int(area[keep].sum()),
                                           int(errors[keep].sum()), int(commons[keep].sum())])
            for tfold, groups in samples.items():
                if name not in groups:
                    continue
                visits = len(groups[name])
                exposure = crop_coverage(groups[name])
                # An independent set of scalar point/rectangle incidence checks.
                ys = [0, 1, 221, 494, 767, 988]
                xs = [0, 1, 552, 660, 767, 1319]
                for y in ys:
                    for x in xs:
                        if exposure[y, x] != sum(q['top'] <= y < q['top']+768 and q['left'] <= x < q['left']+768 for q in groups[name]):
                            raise ValueError('Independent crop incidence check failed')
                data = np.zeros((5, 4, 4), np.float64)
                for t in range(5):
                    for b in range(4):
                        take = (gt == t) & (bins == b)
                        e = exposure[take]
                        data[t, b] = [len(e), np.count_nonzero(e == 0), e.sum(), e.sum()/visits]
                        coverage_totals[tfold][t] += int(e.sum())
                train[tfold].append(dict(name=name, visits=visits, boundary_exposure=data.tolist()))
            rows.append(dict(name=name, fold=fold, boundary_counts=bc.tolist(), components=components))
            if (i+1) % 20 == 0:
                print(f'Processed {i+1}/219; elapsed {time.monotonic()-started:.1f}s; free {shutil.disk_usage(ROOT).free/2**30:.2f} GiB', flush=True)
        for fold in src:
            if coverage_totals[fold].tolist() != src[fold]['five'][1]['sampled_crop_class_counts']:
                raise ValueError('Exact full training crop class counts differ')
        groups = {'all219': c['views'], **{f: [n for n in c['views'] if folds[n] == f] for f in src}}
        summary = {g: dict(views=len(ns), boundary_counts=np.sum([np.array(r['boundary_counts']) for r in rows if r['name'] in ns], axis=0).tolist()) for g, ns in groups.items()}
        result = dict(rows=rows, training=train, summary=summary, source_bindings=bindings,
                      log_hashes=logs, boundary_count_axes=['GT', 'prediction', 'distance_bin', 'total/error/common/high_conf_error'],
                      component_columns=['class', 'area_bin', 'max_inradius_bin', 'components', 'pixels', 'errors', 'common_errors'],
                      training_columns=['pixels', 'zero_exposure_pixels', 'total_exposures', 'exposures_divided_by_view_visits'])
        write_new_json(a.output/'diagnosis.json', result)
        verify_hashes(ROOT, {CONFIG: rec['config_sha256'], **c['input_hashes'], **rec['code_sha256']})
        freeze(read(ROOT/pc['freeze_config']), a.artifact_root)
        rec.update(status='completed_readonly_diagnosis', seconds=time.monotonic()-started,
                   peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                   frozen=frozen, labels_read=len(labels.accessed), original_oof_count_checks=219,
                   training_crop_counts_exact=True, diagnosis_sha256=sha256(a.output/'diagnosis.json'))
        write_new_json(a.output/'result.json', rec)
        print(json.dumps({k: rec[k] for k in ['status', 'seconds', 'peak_rss_mib', 'labels_read']}, indent=2))
    except BaseException as exc:
        write_new_json(a.output/'failure.json', dict(type=type(exc).__name__, message=str(exc), seconds=time.monotonic()-started))
        raise


if __name__ == '__main__':
    main()
