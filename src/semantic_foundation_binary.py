"""SEM-270: dedicated foundation task with an unrestricted 0/4 research lane."""
from __future__ import annotations

import random
import numpy as np

ARMS = ('five_control', 'binary_candidate')


def target_labels(label, arm):
    if arm not in ARMS or label.dtype != np.uint8 or label.ndim != 2:
        raise ValueError('Invalid task or label grid')
    if not label.size or label.max() > 4:
        raise ValueError('Invalid official class ID')
    return label.copy() if arm == ARMS[0] else (label == 4).astype(np.uint8)


def fuse_foundation(base, probability, *, strength=1.0):
    """Strength 0 is exact bypass; deployed strength 1 uses fixed p>=0.5.

    This is a replacement specialist, NOT a zero-initialized learned residual.
    Historical foundation rejection pixels are intentionally not protected.
    """
    if base.dtype != np.uint8 or base.ndim != 2 or not base.size or base.max() > 4:
        raise ValueError('Invalid baseline classes')
    p = np.asarray(probability)
    if p.shape != base.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid foundation probability')
    if strength not in (0.0, 1.0):
        raise ValueError('Only disabled equivalence or fixed full deployment allowed')
    new = base.copy()
    if strength:
        lane = np.isin(base, (0, 4))
        new[lane] = np.where(p[lane] >= .5, 4, 0).astype(np.uint8)
    return new


def training_plan(names, seed, fold, steps, crop, shape=(989, 1320)):
    if not names or len(names) != len(set(names)) or not 0 < crop <= min(shape):
        raise ValueError('Invalid training plan')
    rng = random.Random(seed + fold * 100003)
    return [{'name': names[rng.randrange(len(names))],
             'top': rng.randint(0, shape[0] - crop),
             'left': rng.randint(0, shape[1] - crop), 'flip': rng.random() < .5}
            for _ in range(steps)]


def validate_fold(fold, universe, centers, radius):
    sets = [set(fold[k]) for k in ('train', 'oof', 'guard')]
    if any(len(fold[k]) != len(s) for k, s in zip(('train','oof','guard'), sets)):
        raise ValueError('Duplicate fold access')
    a, b, c = sets
    if not a or not b or a & b or a & c or b & c or a | b | c != set(universe):
        raise ValueError('Fold isolation failed')
    distances = np.linalg.norm(np.array([centers[n] for n in a])[:,None] -
                               np.array([centers[n] for n in b])[None], axis=-1)
    if distances.min() < radius - 1e-9:
        raise ValueError('Spatial guard violated')


def quality_gates(baseline, candidate, control, gates, *, developer=False):
    b, c = [m['per_class']['foundation'] for m in (baseline, candidate)]
    checks = {
        'foundation_iou_gain': c['iou'] >= b['iou'] + gates['foundation_iou_gain'],
        'foundation_precision_gain': c['precision'] >= b['precision'] + gates['foundation_precision_gain'],
        'foundation_recall_retention': c['recall'] >= b['recall'] - gates['maximum_recall_drop'],
        'foundation_recall_floor': c['recall'] >= gates['developer_recall_floor' if developer else 'recall_floor'],
        'miou_nondecreasing': candidate['mIoU_with_background'] >= baseline['mIoU_with_background'],
    }
    if control is not None:
        checks['candidate_beats_matched_control_iou'] = c['iou'] > control['per_class']['foundation']['iou']
    return checks
