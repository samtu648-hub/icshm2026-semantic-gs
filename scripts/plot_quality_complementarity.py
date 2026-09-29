#!/usr/bin/env python3
"""SEM354 scientific figure from audited integer summaries only."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR', '/tmp/sem354-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, write_new_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    r, audit = [json.loads(x.read_text()) for x in (a.source, a.audit)]
    if audit['source_sha256'] != sha256(a.source):
        raise ValueError('Figure requires accepted source hash')
    a.output.mkdir(parents=True, exist_ok=False)
    d = r['pooled']['raw']
    names = ['background', 'deck', 'stay_cable', 'tower', 'foundation']
    recovered = np.array(d['scopes']['all']['recovered_confusion']).sum(1)
    common = np.array(d['common_error_confusion']).sum(1)
    fractions = recovered/(recovered+common)*100
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={'width_ratios': [1, 1.25]})
    ax = axes[0]
    ax.barh(names, fractions, color='#307495', label='Other source has correct class')
    ax.barh(names, 100-fractions, left=fractions, color='#d4d7db', label='No source has correct class')
    for i, v in enumerate(fractions):
        ax.text(v/2, i, f'{v:.1f}%', ha='center', va='center', color='white', fontsize=10)
    ax.set(xlim=(0, 100), xlabel='Share of raw errors in GT class (%)', ylabel='GT class', title='Exact pixel recovery headroom: all nine sources')
    ax.invert_yaxis()
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.17), fontsize=8, frameon=False)
    keys = ['core', 'core_context', 'all', 'lock_deck_stay_cable', 'background_foundation', 'foundation_proposal', 'approved_foundation05', 'tower_proposal']
    labels = ['Core five sources', 'Core + same-model fullframe', 'All nine sources', 'Preserve deck / stay_cable masks', 'Only background / foundation', 'Existing foundation proposal lane', 'Approved foundation 0.05 lane', 'Unapproved tower proposal lane']
    values = [d['scopes'][k]['delta_miou_pp'] for k in keys]
    ax = axes[1]
    ax.barh(labels, values, color='#307495')
    for i, v in enumerate(values):
        ax.text(v+.07, i, f'+{v:.3f}', va='center', fontsize=9)
    ax.invert_yaxis()
    ax.set(xlim=(0, 6.3), xlabel='Truth-assisted correct-choice mIoU gain (pp)', ylabel='Available sources / modification eligibility', title='Diagnostic oracle gains relative to raw')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('SEM354 · historical 108 calibration views · NOT deployable performance', fontsize=13)
    fig.text(.5, .01, 'Source: SEM354 independent pixel recount + integer audit · 2026-09-14 · shared RGB, one seed · raw mIoU 92.913092%', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .94), w_pad=2)
    fig.savefig(a.output/'complementarity.png', dpi=180)
    write_new_json(a.output/'result.json', {'source_sha256': sha256(a.source), 'audit_sha256': sha256(a.audit),
        'figure_sha256': sha256(a.output/'complementarity.png'), 'new_label_decodes': 0})


if __name__ == '__main__':
    main()
