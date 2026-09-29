#!/usr/bin/env python3
"""Audit only the 108 reused calibration targets, not check/developer metrics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.semantic2d import sha256, read_json, write_json
from src.semantic_protected_fusion import (CLASS_NAMES, partition_training_names,
    local_confusion, metrics_from_confusion)
from src.semantic_refinement import mutable_lane, class_boundary_distance
from src.semantic_render_domain import semantic_rgb


def png(path):
    return np.asarray(Image.open(path))


def manifest_paths(root, filename, list_key, path_key, hash_key):
    records = read_json(root / filename)[list_key]
    return {r['name']: (root / r[path_key], r[hash_key]) for r in records}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    out, art = args.output_dir, args.artifact_root
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    cfgpath = ROOT / 'experiments/runs/SEM-260-ERROR-BUDGET/config.yaml'
    config = read_json(cfgpath)
    split = read_json(ROOT / config['semantic_split'])
    partition = partition_training_names(split['semantic_train_labeled'], expected_count=219)
    names = partition['calibration']
    pred = manifest_paths(art / config['parents']['sem160'], 'prediction_manifest.json',
                          'predictions', 'prediction', 'prediction_sha256')
    static = manifest_paths(art / config['parents']['sem001'], 'training_sem001_render_manifest.json',
                            'rendered', 'semantic_png', 'sha256')
    rendered = manifest_paths(art / config['parents']['rgb'], 'rendered_dataset_manifest.json',
                              'rendered_images', 'path', 'sha256')
    labels = art / 'r2/sem_001/labels'
    totals, views, access = {}, [], []
    cm = np.zeros((5, 5), np.int64)
    oracle_cm = cm.copy()
    for name in names:
        paths = [pred[name], static[name], rendered[name]]
        for path, digest in paths:
            if sha256(path) != digest:
                raise ValueError(f'Input hash changed: {path}')
        gt, base, s0, rgb = [png(path) for path in (labels/name, pred[name][0], static[name][0], rendered[name][0])]
        lane = mutable_lane(base, s0)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        edge = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
        c = local_confusion(gt, base)
        cm += c
        oracle = base.copy()
        oracle[lane] = (gt[lane] == 1).astype(np.uint8)
        oracle_cm += local_confusion(gt, oracle)
        strata = {}
        for class_id, classname in enumerate(CLASS_NAMES):
            distance = class_boundary_distance(gt, class_id)
            for direction, mask in [('fn', (gt == class_id) & (base != class_id)),
                                    ('fp', (gt != class_id) & (base == class_id))]:
                key = f'{classname}_{direction}'
                row = {'pixels': int(mask.sum()), 'mutable_pixels': int((mask & lane).sum()),
                       'protected_pixels': int((mask & ~lane).sum()),
                       'boundary4_pixels': int((mask & (distance <= 4)).sum()),
                       'boundary8_pixels': int((mask & (distance <= 8)).sum()),
                       'boundary16_pixels': int((mask & (distance <= 16)).sum()),
                       'render_edge_sum': float(edge[mask].sum(dtype=np.float64))}
                strata[key] = row
                total = totals.setdefault(key, {k: 0 for k in row})
                for k, v in row.items():
                    total[k] += v
        views.append({'name': name, 'strata': strata, 'confusion': c.tolist()})
        access.append({'name': name, 'role': 'calibration_diagnostic_target',
                       'label_sha256': sha256(labels/name),
                       'inputs': [digest for _, digest in paths]})
    selected = sorted(views, key=lambda r: (-r['strata']['deck_fn']['mutable_pixels'], r['name']))[:6]
    panel_paths = []
    for row in selected:
        name = row['name']
        gt, base, s0, rgb = [png(path) for path in (labels/name, pred[name][0], static[name][0], rendered[name][0])]
        lane = mutable_lane(base, s0)
        error = np.zeros_like(rgb)
        error[(gt != base) & lane] = (255, 80, 40)
        error[(gt != base) & ~lane] = (180, 80, 255)
        columns = [rgb, semantic_rgb(gt), semantic_rgb(base), error]
        titles = ['Frozen rendered RGB', 'GT (diagnostic only)', 'SEM-160 OOF baseline', 'Orange mutable / purple protected errors']
        panel = Image.new('RGB', (4 * base.shape[1], base.shape[0] + 32), 'white')
        draw = ImageDraw.Draw(panel)
        for i, (col, title) in enumerate(zip(columns, titles)):
            panel.paste(Image.fromarray(col), (i*base.shape[1], 32))
            draw.text((i*base.shape[1]+5, 8), name+' '+title, fill='black')
        path = out / (Path(name).stem+'_panel.png')
        panel.save(path)
        ys, xs = np.where((gt == 1) | (base == 1))
        if len(xs):
            y0, y1 = max(0, int(ys.min())-16), min(base.shape[0], int(ys.max())+17)
            x0, x1 = max(0, int(xs.min())-16), min(base.shape[1], int(xs.max())+17)
            zoom = np.concatenate([c[y0:y1, x0:x1] for c in columns], axis=1)
            zpath = out/(Path(name).stem+'_deck_zoom.png')
            Image.fromarray(zoom).save(zpath)
            panel_paths.append({'path': zpath.name, 'sha256': sha256(zpath)})
        panel_paths.append({'path': path.name, 'sha256': sha256(path)})
    result = {'experiment_id': config['experiment_id'], 'status': 'completed_diagnostic',
        'repository_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        'config_sha256': sha256(cfgpath), 'code_sha256': sha256(Path(__file__)),
        'split_sha256': sha256(ROOT/config['semantic_split']), 'seed': 0,
        'views': len(names), 'metrics': metrics_from_confusion(cm),
        'oracle_mutable_bg_deck_metrics': metrics_from_confusion(oracle_cm),
        'strata': totals, 'per_view': views, 'access': access, 'panels': panel_paths,
        'check_target_labels_read_this_run': [], 'developer_target_labels_read_this_run': [],
        'interpretation': 'Reused calibration and inherited SEM-001: descriptive incremental-OOF evidence only. RGB gradient is not a semantic visibility classifier. Oracle is unattainable GT-assisted headroom, not a candidate.',
        'seconds': time.perf_counter()-started,
        'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    write_json(out/'result.json', result)
    print(json.dumps({k:result[k] for k in ('views','metrics','oracle_mutable_bg_deck_metrics','strata','seconds','peak_rss_mib')}), flush=True)


if __name__ == '__main__':
    main()
