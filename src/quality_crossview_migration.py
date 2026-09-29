"""Strict per-target source roles and unchanged native SEM329 review composition."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from src.quality_protocol import sha256, verify_hashes
from src.quality_static_training import TrainingLabels
from src.semantic_dense_review import native_prediction, native_support
from src.semantic_second_source_veto import second_source, review_second_source


def require_source(folds, fold_id, target, source):
    fold = next(f for f in folds if f['fold'] == fold_id)
    train, held, guard = (set(fold[k]) for k in ('train', 'oof', 'guard'))
    if train & held or train & guard or held & guard or target not in held:
        raise ValueError('Invalid OOF source roles')
    if source == target or source not in train or source in held or source in guard:
        raise ValueError('Target/guard/out-of-fold source prohibited')


def class_map(path, expected_hash, shape):
    if sha256(path) != expected_hash:
        raise ValueError('Class map source changed')
    with Image.open(path) as im:
        arr = np.array(im)
    if arr.shape != shape or arr.dtype != np.uint8 or arr.max() > 4:
        raise ValueError('Class map grid or class IDs differ')
    return arr


class SourceLabels:
    """Check target-specific fold eligibility before any decode or cache reuse."""
    def __init__(self, folds, root, manifest, resolution):
        self.folds = folds
        self.readers = {f['fold']: TrainingLabels(root, f['train'], manifest, resolution) for f in folds}
        self.access = []
        self.cache = {}

    def read(self, fold, target, source):
        require_source(self.folds, fold, target, source)
        key = (fold, source)
        if key not in self.cache:
            self.cache[key] = self.readers[fold].read(source)
        self.access.append({'fold': fold, 'target': target, 'source': source, 'role': 'same_fold_train_label'})
        return self.cache[key]


def migrated_prediction(baseline, target, first, second, pose, camera, filters, independent=False):
    """Evidence dicts hold maps/labels/poses; never accept target GT."""
    if baseline.dtype != np.uint8 or baseline.ndim != 2 or not baseline.size or baseline.max() > 4:
        raise ValueError('Invalid baseline IDs')
    if baseline.shape != (camera.height, camera.width):
        raise ValueError('Native camera grid differs')
    selected = target['selected_source']
    second_name = second_source(target)
    for name, evidence in ((selected, first), (second_name, second)):
        if (name is None) != (evidence is None) or (evidence is not None and name != evidence['name']):
            raise ValueError('Evidence does not match pose-ordered source')
    dense = baseline.copy()
    stats = {'reliable_pixels': 0, 'outside_source_labels': 0, 'changed_pixels': 0}
    if first is not None:
        dense, stats = native_prediction(baseline, first['labels'], first['maps'], pose, first['pose'],
            camera, first['camera'], filters, independent=independent)
    veto, veto_stats = review_second_source(baseline, dense,
        source_labels=second['labels'] if second else None, maps=second['maps'] if second else None,
        target_pose=pose, source_pose=second['pose'] if second else None, target_camera=camera,
        source_camera=second['camera'] if second else None, filters=filters, independent=independent)
    for out in (dense, veto):
        changed = baseline != out
        if np.any(changed & ((baseline != 4) | (out != 0))) or any(not np.array_equal(out == c, baseline == c) for c in (1, 2, 3)):
            raise ValueError('Migration changed protected classes or added baseline foundation')
    return dense, veto, {'dense': stats, 'veto': veto_stats}


def evidence_trace(baseline, dense, veto, first, second, pose, camera, filters, independent=False):
    """Compact native evidence for every attempted dense deletion, including vetoes."""
    indices = np.flatnonzero(baseline != dense).astype(np.uint32)
    yy, xx = np.unravel_index(indices, baseline.shape)
    points = np.stack((xx+.5, yy+.5), axis=-1)[None]
    trace = {'flat_index': indices, 'old_class': baseline.ravel()[indices],
        'dense_class': dense.ravel()[indices], 'veto_class': veto.ravel()[indices]}
    for role, src in (('first', first), ('second', second)):
        if src is None or not indices.size:
            trace[role+'_source_xy'] = np.full((indices.size, 2), np.nan)
            trace[role+'_reliable'] = np.zeros(indices.size, bool)
            trace[role+'_label'] = np.full(indices.size, 255, np.uint8)
            continue
        xy, support = native_support(points, src['maps'], pose, src['pose'], camera, src['camera'], filters, independent)
        xy, support = xy.reshape(-1, 2), support.ravel()
        sc = src['camera']
        valid = np.isfinite(xy).all(1) & (xy[:, 0] >= 0) & (xy[:, 0] < sc.width) & (xy[:, 1] >= 0) & (xy[:, 1] < sc.height)
        pixels = np.floor(np.nan_to_num(xy, nan=0, posinf=0, neginf=0)).astype(np.int64)
        sampled = np.full(indices.size, 255, np.uint8)
        sampled[valid] = src['labels'][pixels[valid, 1], pixels[valid, 0]]
        trace[role+'_source_xy'], trace[role+'_reliable'], trace[role+'_label'] = xy, support & valid, sampled
    if indices.size and not np.all(trace['first_reliable'] & (trace['first_label'] == 0)):
        raise ValueError('Deletion lacks required reliable background evidence')
    restored = trace['veto_class'] == 4
    if not np.array_equal(restored, trace['second_reliable'] & (trace['second_label'] == 4)):
        raise ValueError('Veto lacks required reliable foundation evidence')
    return trace


def verify_trace(expected, actual):
    if set(expected) != set(actual):
        raise ValueError('Evidence schema differs')
    for key in expected:
        if key.endswith('_source_xy'):
            okay = np.allclose(expected[key], actual[key], atol=2e-5, rtol=0, equal_nan=True)
        else:
            okay = np.array_equal(expected[key], actual[key])
        if not okay:
            raise ValueError('Independent evidence differs: '+key)
