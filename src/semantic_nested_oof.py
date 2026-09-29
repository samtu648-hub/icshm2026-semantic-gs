"""Nested teacher access contracts; no images or labels are needed for planning."""
from __future__ import annotations

import numpy as np


def validate_inner(outer, inner, centers, radius):
    allowed, forbidden = set(outer['train']), set(outer['oof']) | set(outer['guard'])
    train, targets, guard = (set(inner[k]) for k in ('train', 'oof', 'guard'))
    if any(len(inner[k]) != len(set(inner[k])) for k in ('train', 'oof', 'guard')):
        raise ValueError('Duplicate inner access')
    if train & targets or train & guard or targets & guard:
        raise ValueError('Inner fitting/target/guard overlap')
    if train | targets | guard != allowed or (train | targets | guard) & forbidden:
        raise ValueError('Inner teacher leaks outside outer training set')
    if not train or not targets:
        raise ValueError('Empty inner train or prediction set')
    distance = min(float(np.linalg.norm(np.asarray(centers[a])-centers[b]))
                   for a in train for b in targets)
    if distance < radius:
        raise ValueError('Inner spatial guard violated')
    return distance


def nested_plan(folds, centers, seed, radius):
    """Two fixed spatial clusters within EACH outer training set, plus guards."""
    # Share the original deterministic camera-only clustering implementation.
    from scripts.create_sem005_oof_split import deterministic_clusters
    result = []
    for outer in folds:
        names = sorted(outer['train'])
        points = np.asarray([centers[n] for n in names], np.float64)
        assignments, cluster_centers, initial = deterministic_clusters(points, seed, count=2)
        inners = []
        for index in range(2):
            targets = sorted(n for n, k in zip(names, assignments) if k == index)
            remaining = sorted(set(names) - set(targets))
            guard = [n for n in remaining if min(np.linalg.norm(np.asarray(centers[n])-centers[t])
                                                for t in targets) < radius]
            inner = {'inner_fold': index, 'train': sorted(set(remaining)-set(guard)),
                     'oof': targets, 'guard': guard}
            inner['minimum_train_target_distance'] = validate_inner(outer, inner, centers, radius)
            inners.append(inner)
        if sorted(n for i in inners for n in i['oof']) != names:
            raise ValueError('Nested predictions do not cover outer train exactly once')
        result.append({'outer_fold': outer['fold'], 'inners': inners,
                       'cluster_centers': cluster_centers.tolist(), 'initial_indices': initial})
    return result


def validate_teacher_metadata(metadata, outer, inner, kind):
    expected = {'experiment_id': 'SEM-263-NESTED-POINT-OOF', 'outer_fold': outer['fold'],
                'inner_fold': inner['inner_fold'], 'kind': kind,
                'train': inner['train'], 'oof': inner['oof'], 'guard': inner['guard']}
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError('Wrong nested teacher provenance')
    if set(metadata['train']) - set(outer['train']):
        raise ValueError('Nested teacher has forbidden training supervision')


def common_sampling_base(static):
    """An arm-independent sampling frame, with no learned 2D teacher access.

    Keep the inherited static deck boundary; static stay_cable becomes background
    only in the SAMPLER, not in any baseline/prediction or protected mask.
    """
    result = static.copy()
    result[result == 2] = 0
    return result
