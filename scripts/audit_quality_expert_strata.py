#!/usr/bin/env python3
"""Independent integer audit plus fixed frame-location stratification, no models."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.quality_static_training import TrainingLabels


def read(p):
    return json.loads(Path(p).read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--artifact-root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output = a.output.resolve()
    if not a.output.is_relative_to(ROOT/'outputs/quality/expert_diagnostic'):
        p.error('Isolated audit directory required')
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    config_path = ROOT/'experiments/runs/SEM-367-EXPERT-ERROR-STRATA/audit_config.yaml'
    ac = read(config_path)
    c = read(ROOT/ac['source_config'])
    pc = read(ROOT/c['parent_config'])
    result = read(a.source/'result.json')
    verify_hashes(a.source, {'diagnosis.json': result['diagnosis_sha256']})
    verify_hashes(ROOT, {ac['source_config']: result['config_sha256'], **result['code_sha256'], **c['input_hashes']})
    d = read(a.source/'diagnosis.json')
    old = {r['name']: r for r in read(ROOT/'outputs/quality/selector_bottleneck/SEM-364-r1/diagnosis.json')['rows']}
    labels = TrainingLabels(a.artifact_root/'r2/sem_001/labels', c['views'],
                           read(ROOT/'experiments/runs/SEM-001/label_manifest.json'), [1320, 989])
    if sorted(r['name'] for r in d['rows']) != c['views']:
        raise ValueError('219 unique roles differ')
    yy, xx = np.indices((989, 1320))
    distance = np.minimum.reduce([yy, xx, 988-yy, 1319-xx])
    frame_bins = np.searchsorted(ac['frame_distance_px'], distance, side='left')
    records = []
    for r in d['rows']:
        name, fold = r['name'], r['fold']
        gt = labels.read(name)
        folder = ROOT/pc['sources'][fold]['five']
        saved = read(folder/'result.json')
        path = 'predictions/'+name
        verify_hashes(folder, {path: saved['artifacts'][path]})
        with Image.open(folder/path) as im:
            raw = np.array(im)
        # Scalar class masking is independent of primary GT*5+pred bincount.
        cm = np.array([[np.count_nonzero((gt == t) & (raw == q)) for q in range(5)] for t in range(5)])
        b = np.array(r['boundary_counts'], np.int64)
        if not np.array_equal(b[..., 0].sum(2), cm):
            raise ValueError('Pixel confusion mismatch')
        hist = np.array(old[name]['source_error_mask_counts'])
        if not np.array_equal(b[..., 1].sum((1, 2)), hist.sum(1)) or not np.array_equal(b[..., 2].sum((1, 2)), hist[:, 0]):
            raise ValueError('SEM364 class error/common totals differ')
        for t in range(1, 5):
            rows = [v for v in r['components'] if v[0] == t]
            total = [sum(v[j] for v in rows) for j in [4, 5, 6]]
            if total != [int(cm[t].sum()), int(hist[t].sum()), int(hist[t, 0])]:
                raise ValueError('Component partition differs')
        frame = np.array([[[np.count_nonzero((gt == t) & (raw == q) & (frame_bins == k))
                            for k in range(4)] for q in range(5)] for t in range(5)])
        if not np.array_equal(frame.sum(2), cm):
            raise ValueError('Frame partition differs')
        records.append(dict(name=name, fold=fold, frame_confusion=frame.tolist()))
    for fold, rows in d['training'].items():
        saved = read(ROOT/pc['sources'][fold]['five']/'result.json')
        if sorted(r['name'] for r in rows) != sorted(saved['access']['train']):
            raise ValueError('Training-only roles differ')
        totals = [sum(round(row['boundary_exposure'][t][b][2]) for row in rows for b in range(4)) for t in range(5)]
        if totals != saved['sampled_crop_class_counts']:
            raise ValueError('Crop exposure class totals mismatch')
    write_new_json(a.output/'frame_strata.json', records)
    write_new_json(a.output/'result.json', dict(status='passed_independent_counts_and_frame_diagnosis',
        experiment_id=c['experiment_id'], git_commit=c['git_commit'], seed=c['seed'],
        dataset_manifest=c['dataset_manifest'], split_id=c['split_id'],
        config_sha256=sha256(config_path), code_sha256=sha256(Path(__file__)),
        source_sha256=sha256(a.source/'diagnosis.json'), frame_sha256=sha256(a.output/'frame_strata.json'),
        views=len(records), seconds=time.monotonic()-start,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        new_training=0, new_inference=0, new_74_reads=0))
    print('Independent 219-view count/component/training audit and frame diagnosis passed', flush=True)


if __name__ == '__main__':
    main()
