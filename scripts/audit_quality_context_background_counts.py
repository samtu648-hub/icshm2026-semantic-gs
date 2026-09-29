#!/usr/bin/env python3
"""Independently recompute SEM347 metrics and acceptance from saved integer counts."""
import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def score(cm):
    ious = []
    for i in range(5):
        tp = cm[i][i]
        union = sum(cm[i]) + sum(row[i] for row in cm) - tp
        ious.append(Fraction(tp, union))
    return sum(ious) / 5, ious[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    r = json.loads((a.run / 'result.json').read_text())
    rows = json.loads((a.run / 'per_view.json').read_text())
    assert digest(a.run / 'per_view.json') == r['per_view_sha256']
    assert r['status'] == 'completed_finite_calibration_no_selection'
    assert len(rows) == len({v['name'] for v in rows}) == 108
    def aggregate(subset, baseline, threshold):
        matrix = [[0] * 5 for _ in range(5)]
        counts = [0] * 5
        for row in subset:
            truth = row['candidates'][threshold]
            for i in range(5):
                counts[i] += truth[i]
                for j in range(5):
                    matrix[i][j] += row['baseline_confusions'][baseline][i][j]
                matrix[i][2] -= truth[i]
                matrix[i][0] += truth[i]
        return matrix, counts
    checks = 0
    for t, eligibility in r['eligibility'].items():
        passes = {}
        for fold, g in r['per_fold'].items():
            subset = [v for v in rows if v['fold'] == fold]
            for b in ['raw', 'selected']:
                cm, truth = aggregate(subset, b, t)
                expected = g['metrics'][b][t]
                assert cm == expected['confusion'] and truth == g['truth_counts'][t]
                m, cable = score(cm)
                assert abs(float(m) - expected['mIoU_with_background']) < 1e-15
                assert abs(float(cable) - expected['per_class']['stay_cable']['IoU']) < 1e-15
                old_m, old_c = score(aggregate(subset, b, 'off')[0])
                passes[fold + '/' + b] = m >= old_m and cable >= old_c
                checks += 1
        assert passes == eligibility['per_fold_control_pass']
        assert all(passes.values()) == eligibility['all_pass']
        for b in ['raw', 'selected']:
            cm, truth = aggregate(rows, b, t)
            assert cm == r['metrics'][b][t]['confusion']
            assert truth == eligibility['truth_counts']
        assert eligibility['nonzero'] == (sum(truth) > 0)
        assert eligibility['recall_loss_requires_user_review'] == (truth[2] > 0)
    assert not any(e['all_pass'] and e['nonzero'] for e in r['eligibility'].values())
    result = dict(status='passed_independent_integer_count_and_exact_fraction_audit',
                  experiment_id=r['experiment_id'], config_sha256=r['config_sha256'],
                  result_sha256=digest(a.run / 'result.json'),
                  audit_code_sha256=digest(Path(__file__)), views=108, fold_control_checks=checks,
                  nonzero_candidates_passed=0, outcome='retain_off_close_finite_context_threshold_line')
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
