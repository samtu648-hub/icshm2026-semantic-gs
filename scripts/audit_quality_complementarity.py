#!/usr/bin/env python3
"""Independent integer/rational audit of SEM354 saved counts (no image/GT decode)."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, write_new_json


def scores(cm):
    values = []
    for k in range(5):
        den = sum(cm[k])+sum(row[k] for row in cm)-cm[k][k]
        if den:
            values.append(Fraction(cm[k][k], den))
    return sum(values)/len(values)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--replay', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    read = lambda path: json.loads(path.read_text())
    cfg = read(ROOT/'experiments/runs/SEM-354-COMPLEMENTARITY-ORACLE-DIAG/config.yaml')
    r, replay = read(a.source/'result.json'), read(a.replay/'result.json')
    rows = read(a.source/'per_view.json')
    if (not replay.get('independent_pixel_recount') or replay['replay_source_sha256'] != sha256(a.source/'result.json')
            or r['per_view_sha256'] != sha256(a.source/'per_view.json')
            or rows != read(a.replay/'per_view.json')):
        raise ValueError('Missing exact independent full-pixel evidence')
    nscopes = ncubes = 0
    for row in rows:
        for base in ('raw', 'selected'):
            data = row[base]
            cm = data['confusion']
            joint = data['wrong_correctness_joint']
            for t in range(5):
                for b in range(5):
                    if t != b and sum(n for y, p, bits, n in joint if y == t and p == b) != cm[t][b]:
                        raise ValueError('Joint wrong-pixel partition incomplete')
            for index, key in enumerate(cfg['branches']):
                cube = data['source_cubes'][key]
                for t in range(5):
                    for b in range(5):
                        if sum(cube[t][b]) != cm[t][b]:
                            raise ValueError('Branch/control marginal differs')
                        if t != b and sum(n for y, p, bits, n in joint if y == t and p == b and bits & (1 << index)) != cube[t][b][t]:
                            raise ValueError('Joint source correctness differs from transition tensor')
                ncubes += 1
            for scope, x in data['scopes'].items():
                repair = x['recovered_confusion']
                expected = [[cm[t][b]-repair[t][b]+(sum(repair[t]) if t == b else 0) for b in range(5)] for t in range(5)]
                if expected != x['oracle_confusion'] or any(repair[k][k] for k in range(5)):
                    raise ValueError('Repair algebra or zero-harm definition fails')
                if any(not 0 <= repair[t][b] <= cm[t][b] for t in range(5) for b in range(5)):
                    raise ValueError('Invented recoverable pixels')
                if scores(expected) < scores(cm):
                    raise ValueError('Correct-choice oracle decreased mIoU')
                if scope in ('core', 'core_context', 'all', 'lock_deck_stay_cable', 'background_foundation'):
                    mask = (1 << (5 if scope == 'core' else 6 if scope == 'core_context' else 9))-1
                    expected_repair = [[0]*5 for _ in range(5)]
                    for t, b, bits, n in joint:
                        permit = bool(bits & mask)
                        if scope == 'lock_deck_stay_cable':
                            permit &= t in (0, 3, 4) and b in (0, 3, 4)
                        if scope == 'background_foundation':
                            permit &= t in (0, 4) and b in (0, 4)
                        if permit:
                            expected_repair[t][b] += n
                    if expected_repair != repair:
                        raise ValueError('Oracle coverage differs from independent correctness joint')
                nscopes += 1
    checks = 0
    for group, group_rows in [('pooled', rows)] + [(f, [x for x in rows if x['fold'] == f]) for f in cfg['views']]:
        data = r['pooled'] if group == 'pooled' else r['per_fold'][group]
        for base in ('raw', 'selected'):
            cm = [[sum(x[base]['confusion'][t][b] for x in group_rows) for b in range(5)] for t in range(5)]
            if cm != data[base]['metrics']['confusion']:
                raise ValueError('Grouped control mismatch')
            for scope in cfg['scopes']:
                repair = [[sum(x[base]['scopes'][scope]['recovered_confusion'][t][b] for x in group_rows) for b in range(5)] for t in range(5)]
                oracle = [[cm[t][b]-repair[t][b]+(sum(repair[t]) if t == b else 0) for b in range(5)] for t in range(5)]
                result = data[base]['scopes'][scope]
                delta = 100*float(scores(oracle)-scores(cm))
                if (oracle != result['metrics_truth_assisted_only']['confusion'] or repair != result['recovered_confusion']
                        or abs(delta-result['delta_miou_pp']) > 1e-11):
                    raise ValueError('Grouped rational metric mismatch')
                checks += 1
    out = {'experiment_id': cfg['experiment_id'], 'status': 'passed_integer_fraction_complementarity_audit',
        'source_sha256': sha256(a.source/'result.json'), 'replay_sha256': sha256(a.replay/'result.json'),
        'views': len(rows), 'source_cube_checks': ncubes, 'view_control_scope_checks': nscopes,
        'pooled_fold_control_scope_fraction_checks': checks, 'new_label_decodes': 0}
    write_new_json(a.output, out)
    print(json.dumps(out))


if __name__ == '__main__':
    main()
