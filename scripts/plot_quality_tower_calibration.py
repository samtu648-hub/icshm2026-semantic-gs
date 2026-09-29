#!/usr/bin/env python3
"""Plot SEM353 saved counts only; no candidate selection or label access."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    result_path = args.run / 'result.json'
    result = json.loads(result_path.read_text())
    rows = json.loads((args.run / 'per_view.json').read_text())
    if result['status'] != 'completed_fixed108_tower_calibration_no_selection':
        raise ValueError('Completed calibration required')
    args.output.mkdir(parents=True, exist_ok=False)
    tags = list(result['eligibility'])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for fold, group in result['per_fold'].items():
        curves = group['metrics']['raw']
        axes[0, 0].plot(tags[:5], [100 * (curves[t]['mIoU_with_background'] - curves['off']['mIoU_with_background']) for t in tags[:5]], marker='o', label=fold)
    axes[0, 0].set_title('Small margins: raw mIoU by semantic OOF fold')
    axes[0, 0].set_ylabel('Change (percentage points)')
    for base, curves in result['metrics'].items():
        axes[0, 1].plot(tags, [100 * (curves[t]['mIoU_with_background'] - curves['off']['mIoU_with_background']) for t in tags], marker='o', linestyle='--' if base == 'selected' else '-', label=base)
    axes[0, 1].set_title('All six fixed candidates: pooled mIoU')
    axes[0, 1].set_ylabel('Change vs respective control (pp)')
    x = np.arange(5)
    for i, (field, label) in enumerate([('corrected', 'True tower recovered'), ('harmed', 'True stay_cable lost'), ('wrong_to_wrong', 'Other wrong class')]):
        axes[1, 0].bar(x + (i - 1) * .25, [result['eligibility'][t][field] for t in tags[1:]], width=.25, label=label)
    axes[1, 0].set_xticks(x, tags[1:])
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_ylabel('Pixels (log scale)')
    axes[1, 0].set_title('Every change is original raw stay_cable to tower')
    curves = result['metrics']['raw']
    for cls in ['tower', 'stay_cable']:
        axes[1, 1].plot(tags, [100 * (curves[t]['per_class'][cls]['recall'] - curves['off']['per_class'][cls]['recall']) for t in tags], marker='o', label=cls)
    axes[1, 1].set_title('Tower recall gain always costs true stay_cable')
    axes[1, 1].set_ylabel('Recall change (pp)')
    for ax in axes.flat:
        ax.set_xlabel('Maximum probability gap (categorical spacing)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    for ax in [axes[0, 0], axes[0, 1], axes[1, 1]]:
        ax.axhline(0, color='black', linewidth=.6)
    fig.suptitle('SEM353 | reused 108 calibration views | no selection / no blind claim\n0.05, 0.10, 0.20 pass the finite IoU screen; 0.40 and 1.00 fail', fontsize=13)
    fig.savefig(args.output / 'tower_calibration.png', dpi=160)
    plt.close(fig)
    evidence = {'experiment_id': result['experiment_id'], 'config_sha256': result['config_sha256'], 'run_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(), 'new_labels_decoded': 0, 'automatic_selection': False, 'concentration': {}}
    for tag in tags[1:]:
        harm = sorted(rows, key=lambda row: row['candidates'][tag][2], reverse=True)
        evidence['concentration'][tag] = {
            'views_harmed': sum(row['candidates'][tag][2] > 0 for row in rows),
            'views_corrected': sum(row['candidates'][tag][3] > 0 for row in rows),
            'highest_harm_five': [{'name': row['name'], 'fold': row['fold'], 'truth_counts': row['candidates'][tag]} for row in harm[:5]],
        }
    (args.output / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')


if __name__ == '__main__':
    main()
