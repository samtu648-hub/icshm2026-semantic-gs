#!/usr/bin/env python3
"""Prepare SEM354's next-protocol role draft from camera metadata only; no fit."""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.create_rgb_split import parse_colmap_images, kmeans_labels
from src.quality_protocol import sha256, write_new_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    c = json.loads((ROOT/'experiments/runs/SEM-354-COMPLEMENTARITY-ORACLE-DIAG/config.yaml').read_text())
    meta = Path('outputs/quality/runs/RGB-110-SCAFFOLD-CTRL-full-r1/scene/sparse/0/images.txt')
    cams = {x.name: x for x in parse_colmap_images(ROOT/meta)}
    radius = json.loads((ROOT/'experiments/splits/sem_005_oof_spatial_v1.json').read_text())['guard_policy']['radius']
    result = {}
    for fold, names in c['views'].items():
        records = [cams[n] for n in names]
        labels = kmeans_labels(records, 3, c['seed'])
        blocks = [[x.name for x, group in zip(records, labels) if group == j] for j in range(3)]
        outer = []
        for i, evaluation in enumerate(blocks):
            pool = [n for n in names if n not in evaluation]
            guard = [n for n in pool if min(math.dist(cams[n].center, cams[t].center) for t in evaluation) < radius]
            pool = [n for n in pool if n not in guard]
            inner = []
            for j in range(3):
                if j == i:
                    continue
                selection = [n for n in blocks[j] if n in pool]
                candidates = [n for n in pool if n not in selection]
                ig = [n for n in candidates if min(math.dist(cams[n].center, cams[t].center) for t in selection) < radius]
                fit = [n for n in candidates if n not in ig]
                assert fit and selection and not set(fit) & (set(evaluation) | set(selection))
                inner.append({'selection': selection, 'fit': fit, 'guard': ig})
            outer.append({'evaluation': evaluation, 'refit': pool, 'outer_guard': guard, 'inner': inner})
        result[fold] = {'blocks': blocks, 'outer_roles': outer}
    write_new_json(a.output, {'status': 'draft_metadata_only_no_fusion_fit', 'seed': c['seed'],
        'camera_metadata': str(meta), 'camera_sha256': sha256(ROOT/meta),
        'allocation_code_sha256': sha256(ROOT/'scripts/create_rgb_split.py'), 'guard_radius': radius,
        'method': 'same existing base fold; deterministic3means official camera centres; nested2way inner selection with fixed inherited distance guards',
        'roles': result})


if __name__ == '__main__':
    main()
