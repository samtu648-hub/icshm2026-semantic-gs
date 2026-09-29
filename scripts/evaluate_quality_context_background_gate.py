#!/usr/bin/env python3
"""SEM347 finite preregistered calibration; never selects or exports a policy."""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.quality_static_training import TrainingLabels
from src.quality_fusion import confusion
from src.quality_semantic_audit import metrics
from src.quality_context_background_gate import background_gate

CONFIG = 'experiments/runs/SEM-347-CONTEXT-BACKGROUND-GATE/config.yaml'


def read(path):
    return json.loads(path.read_text())


def image(path):
    with Image.open(path) as im:
        return np.array(im)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    if not a.output.resolve().is_relative_to(ROOT / 'outputs/quality/cable_diagnostic'):
        parser.error('Use a new cable_diagnostic output directory')
    a.output.mkdir(parents=True, exist_ok=False)
    c = read(ROOT / CONFIG)
    r = {k: c[k] for k in ('experiment_id', 'git_commit', 'seed', 'dataset_manifest', 'split_id')}
    r.update(status='started', started_utc=datetime.now(timezone.utc).isoformat(),
             config_sha256=sha256(ROOT / CONFIG), code_sha256=c['code_sha256'],
             selected_threshold=None, automatic_adoption=False, new_prediction_outputs=False,
             new_training=False, new_inference=False, check_labels_decoded=0, development_labels_decoded=0)
    try:
        verify_hashes(ROOT, c['input_hashes'])
        verify_hashes(ROOT, c['code_sha256'])
        for name in (CONFIG, *c['code_sha256']):
            dest = a.output / 'code' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)
        write_new_json(a.output / 'started.json', r)
        pc = read(ROOT / c['preflight_config'])
        export_dir = ROOT / pc['rgb_export']
        verify_hashes(export_dir, {'result.json': pc['rgb_export_result_sha256']})
        rgb = {e['name']: e['sha256'] for e in read(export_dir / 'result.json')['rendered_images']}
        frozen = {e['path']: e['sha256'] for e in read(ROOT / c['frozen_manifest'])['artifacts']}

        def guard():
            if shutil.disk_usage(ROOT).free < 100 * 1024 ** 3:
                raise OSError('Preserve 100GiB free space')
            verify_hashes(export_dir / 'images', rgb)
            verify_hashes(a.artifact_root, frozen)

        guard()
        parent = read(ROOT / c['comparison'] / 'result.json')
        selected = read(ROOT / c['selected'] / 'result.json')
        queue = read(ROOT / c['fullframe_batch'] / 'result.json')
        if queue['status'] != 'completed_fullframe108_and_exact_fresh_replay':
            raise ValueError('Fullframe source incomplete')
        names = [n for v in c['views'].values() for n in v]
        if len(names) != 108 or len(set(names)) != 108:
            raise ValueError('Invalid calibration role')
        labels = TrainingLabels(a.artifact_root / 'r2/sem_001/labels', names,
                                read(ROOT / pc['label_manifest']), [1320, 989])
        per = []
        groups = {}
        for fold, fold_names in c['views'].items():
            base = ROOT / c['fullframe_batch'] / ('full-' + fold)
            source = read(base / 'result.json')
            fresh = read(ROOT / c['fullframe_batch'] / ('replay-' + fold) / 'result.json')
            if (source['views'] != fold_names or source['binding']['partition'] != fold or
                    not fresh['replay_exact'] or fresh['replay_source_sha256'] != sha256(base / 'result.json')):
                raise ValueError('Wrong fold or missing fresh replay')
            verify_hashes(base, source['artifacts'])
            acc = {b: {t: np.zeros((5, 5), np.int64) for t in c['candidates']}
                   for b in ('raw', 'selected')}
            counts = {t: np.zeros(5, np.int64) for t in c['candidates']}
            for name in fold_names:
                raw_rel = 'calibration/mask2former_6k/raw/' + name
                sel_rel = 'calibration/predictions/' + name
                verify_hashes(ROOT / c['comparison'], {raw_rel: parent['artifacts'][raw_rel]})
                verify_hashes(ROOT / c['selected'], {sel_rel: selected['artifacts'][sel_rel]})
                raw = image(ROOT / c['comparison'] / raw_rel)
                sel = image(ROOT / c['selected'] / sel_rel)
                if np.any((raw != sel) & ~((raw == 3) & (sel == 4))):
                    raise ValueError('Selected control exceeds sealed foundation lane')
                with np.load(base / (Path(name).stem + '.npz'), allow_pickle=False) as z:
                    probability = z['probabilities']
                if (probability.dtype != np.float32 or not np.isfinite(probability).all() or
                        probability.min() < 0 or probability.max() > 1 or
                        not np.allclose(probability.sum(0), 1, atol=2e-6, rtol=0)):
                    raise ValueError('Invalid saved normalized probabilities')
                # All decisions are label-free. Only this exact 108-view role is decoded.
                outputs = {t: background_gate(raw, probability, v) for t, v in c['candidates'].items()}
                target = labels.read(name)
                baselines = {b: confusion(target, x) for b, x in [('raw', raw), ('selected', sel)]}
                row = {'name': name, 'fold': fold, 'baseline_confusions': {b: x.tolist() for b, x in baselines.items()}, 'candidates': {}}
                for t, out in outputs.items():
                    changed = raw != out
                    truth = np.bincount(target[changed], minlength=5)
                    if np.any(changed & ~((raw == 2) & (out == 0))):
                        raise ValueError('Invalid transition')
                    counts[t] += truth
                    for b, control in [('raw', raw), ('selected', sel)]:
                        candidate = control.copy()
                        candidate[changed] = 0
                        cm = confusion(target, candidate)
                        algebra = baselines[b].copy()
                        algebra[:, 2] -= truth
                        algebra[:, 0] += truth
                        if not np.array_equal(cm, algebra):
                            raise ValueError('Pixel recount differs from transition algebra')
                        if any(not np.array_equal(control == i, candidate == i) for i in (1, 3, 4)):
                            raise ValueError('Protected class changed')
                        acc[b][t] += cm
                    row['candidates'][t] = truth.tolist()
                per.append(row)
            groups[fold] = {'truth_counts': {t: x.tolist() for t, x in counts.items()},
                            'metrics': {b: {t: metrics(x) for t, x in v.items()} for b, v in acc.items()}}
            print(f'{fold}: {len(fold_names)} calibration views counted', flush=True)
        totals = {b: {t: metrics(sum(np.array(g['metrics'][b][t]['confusion']) for g in groups.values()))
                      for t in c['candidates']} for b in ('raw', 'selected')}
        for b in totals:
            if totals[b]['off'] != selected['evaluation']['calibration']['metrics'][b]:
                raise ValueError('Historical calibration control mismatch')
        eligibility = {}
        for t in c['candidates']:
            details = {}
            for f, g in groups.items():
                for b in totals:
                    original, candidate = g['metrics'][b]['off'], g['metrics'][b][t]
                    details[f + '/' + b] = all(candidate[k] >= original[k] - 1e-12 for k in ['mIoU_with_background']) and candidate['per_class']['stay_cable']['IoU'] >= original['per_class']['stay_cable']['IoU'] - 1e-12
            truth = np.sum([g['truth_counts'][t] for g in groups.values()], axis=0).tolist()
            eligibility[t] = {'per_fold_control_pass': details, 'all_pass': all(details.values()),
                              'truth_counts': truth, 'nonzero': sum(truth) > 0,
                              'recall_loss_requires_user_review': truth[2] > 0}
        guard()
        write_new_json(a.output / 'per_view.json', per)
        r.update(status='completed_finite_calibration_no_selection', per_fold=groups, metrics=totals,
                 eligibility=eligibility, calibration_views=len(per), rgb_exports_verified=len(rgb),
                 frozen_verified=len(frozen), per_view_sha256=sha256(a.output / 'per_view.json'))
    except BaseException as exc:
        r.update(status='failed_finite_calibration', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_new_json(a.output / 'result.json', r)


if __name__ == '__main__':
    main()
