#!/usr/bin/env python3
"""SEM360: evaluate saved heads on original fit images; never fit a model."""
import argparse
import hashlib
from pathlib import Path
import resource
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_quality_selector_weight_ablation import read, load, sources, provenance
import numpy as np
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.quality_mask2former_final import FinalContext
from src.quality_static_training import TrainingLabels
from src.quality_learned_selector import predict, sample_rows, feature_rows
from src.quality_semantic_audit import metrics
from src.quality_bottleneck import core_counts
from scripts.audit_quality_selector_weight_ablation import reference_rows, local_reference, proposals

CONFIG = 'experiments/runs/SEM-360-SELECTOR-FIT-GAP-DIAG/config.yaml'
CODE = ['scripts/diagnose_quality_selector_fit_gap.py', 'src/quality_bottleneck.py']


def reference_prediction(data, model):
    allowed = proposals(data)
    data = (*data[:4], local_reference(data[0]))
    out = data[0].ravel().copy()
    if bool(model['fallback']):
        return out.reshape(data[0].shape)
    beta = model['coef']/model['scale']
    offset = -np.dot(model['mean'], beta)
    scores = np.full(allowed.shape, -np.inf)
    for cid in range(5):
        pixels = np.where(allowed[cid])[0]
        for start in range(0, len(pixels), 32768):
            ix = pixels[start:start+32768]
            scores[cid, ix] = reference_rows(data, ix, np.full(len(ix), cid))@beta+offset
    top = np.max(scores, axis=0)
    accept = (top > 0) & (np.sum(scores == top, axis=0) == 1)
    out[accept] = np.argmax(scores, axis=0)[accept]
    return out.reshape(data[0].shape)


def cube(gt, raw, pred, reference=False):
    if reference:
        out = np.zeros((5, 5, 5), np.int64)
        for t in range(5):
            keep = gt == t
            np.add.at(out[t], (raw[keep], pred[keep]), 1)
        return out
    return np.bincount(gt.astype(np.int64).ravel()*25+raw.ravel()*5+pred.ravel(), minlength=125).reshape(5, 5, 5)


def summary(cubes):
    c = sum(np.asarray(v, np.int64) for v in cubes)
    raw, pred = metrics(c.sum(2)), metrics(c.sum(1))
    correct = [int(sum(c[t,p,t] for p in range(5) if p != t)) for t in range(5)]
    harm = [int(sum(c[t,t,p] for p in range(5) if p != t)) for t in range(5)]
    return dict(cube=c.tolist(), raw=raw, selector=pred,
                delta_miou_pp=100*(pred['mIoU_with_background']-raw['mIoU_with_background']),
                corrected_by_class=correct, harmed_by_class=harm,
                net_corrected=sum(correct)-sum(harm))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--artifact-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference-from', type=Path)
    a = p.parse_args()
    a.output = a.output.resolve()
    if not a.output.is_relative_to(ROOT/'outputs/quality/fit_gap'):
        p.error('Isolated fit_gap output required')
    a.output.mkdir(parents=True, exist_ok=False)
    c = read(ROOT/CONFIG)
    pc = read(ROOT/c['parent_config'])
    parent = ROOT/c['parent_batch']
    old = read(parent/'result.json')
    started = time.time()
    prior_seconds = read(a.reference_from/'result.json')['seconds'] if a.reference_from else 0
    code = {n:sha256(ROOT/n) for n in CODE}
    rec = {k:c[k] for k in ('experiment_id','git_commit','seed','dataset_manifest','split_id')}
    rec.update(config_sha256=sha256(ROOT/CONFIG), code_sha256=code, status='started', new_fits=0, new_111_or74_scores=0)
    write_new_json(a.output/'started.json', rec)
    for n in [CONFIG,*CODE]:
        dst = a.output/'code'/n
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/n,dst)

    def guard():
        used = sum(f.stat().st_size for f in a.output.parent.rglob('*') if f.is_file())
        if time.time()-started+prior_seconds > c['resources']['maximum_seconds'] or used > c['resources']['maximum_output_mib']*2**20 or shutil.disk_usage(ROOT).free < c['resources']['minimum_free_gib']*2**30:
            raise RuntimeError('Registered diagnostic budget reached')

    try:
        verify_hashes(ROOT, c['input_hashes'])
        verify_hashes(ROOT, old['code_sha256'])
        verify_hashes(parent, read(parent/'predictions_sealed.json')['files'])
        if old['config_sha256'] != sha256(ROOT/c['parent_config']) or c['roles'] != pc['roles'] or c['views'] != pc['views']:
            raise ValueError('Parent protocol drift')
        ctx = FinalContext(ROOT,a.artifact_root,ROOT/c['final_batch'])
        proof = provenance(pc,ctx)
        lm = read(ROOT/'experiments/runs/SEM-001/label_manifest.json')
        heldout = read(parent/'per_view.json')
        complement = {r['name']:r for r in read(ROOT/c['complementarity']/'per_view.json')}
        previous = {(r['head'],r['name']):r for r in read(a.reference_from/'per_view.json')} if a.reference_from else {}
        rows, heads = [], {}
        for role in c['roles']:
            folder = parent/role['id']
            fit = read(folder/'fit.json')
            if fit['role'] != role or fit['fit_labels_accessed'] != sorted(role['fit']):
                raise ValueError('Fit whitelist drift')
            with np.load(folder/'model.npz',allow_pickle=False) as z:
                model = {k:z[k] for k in z.files}
            labels = TrainingLabels(a.artifact_root/'r2/sem_001/labels',role['fit'],lm,[1320,989])
            s = sources(pc,role['fold'])
            part, mse, zero = [], [], []
            for name in role['fit']:
                guard()
                data = load(s,name)
                pred = reference_prediction(data,model) if a.reference_from else predict(data,model)
                # Prediction is fixed and computed before this diagnostic accesses GT.
                gt = labels.read(name)
                raw = data[0]
                if a.reference_from:
                    allowed = proposals(data)
                    pool = np.where(allowed.any(0))[0]
                    token = hashlib.sha256(f"{c['seed']}/{role['fold']}/{name}".encode()).digest()
                    rng = np.random.Generator(np.random.PCG64(int.from_bytes(token[:8],'little')))
                    if len(pool) > pc['learner']['max_pixels_per_view']:
                        pool = np.sort(rng.choice(pool,pc['learner']['max_pixels_per_view'],replace=False))
                    cls, pos = np.where(allowed[:,pool])
                    ix = pool[pos]
                else:
                    ix, cls = sample_rows(data,c['seed'],role['fold'],name,pc['learner']['max_pixels_per_view'])
                digest = hashlib.sha256(np.column_stack([ix,cls]).astype('<i8').tobytes()).hexdigest()
                journal = next(j for j in fit['sampling'] if j['name'] == name)
                if digest != journal['sample_sha256'] or len(ix) != journal['rows'] or len(np.unique(ix)) != journal['pixels']:
                    raise ValueError('Original sampling drift')
                x = reference_rows(data,ix,cls) if a.reference_from else feature_rows(data,ix,cls)
                score = ((x-model['mean'])/model['scale'])@model['coef']
                truth = gt.ravel()[ix]
                y = (truth == cls).astype(float)-(truth == raw.ravel()[ix]).astype(float)
                if len(ix):
                    mse.append(float(np.mean((score-y)**2)))
                    zero.append(float(np.mean(y*y)))
                full = cube(gt,raw,pred,bool(a.reference_from))
                sampled = np.unique(ix)
                sub = cube(gt.ravel()[sampled],raw.ravel()[sampled],pred.ravel()[sampled],bool(a.reference_from))
                if full.sum(2).tolist() != complement[name]['raw']['confusion']:
                    raise ValueError('Fit raw comparator drift')
                row = dict(head=role['id'],fold=role['fold'],name=name,prediction_array_sha256=hashlib.sha256(pred.tobytes()).hexdigest(),sample_sha256=digest,
                           full_cube=full.tolist(),sample_cube=sub.tolist(),unsampled_cube=(full-sub).tolist(),candidate_rows=len(ix),sample_mse=float(np.mean((score-y)**2)) if len(ix) else None,zero_score_mse=float(np.mean(y*y)) if len(ix) else None)
                if a.reference_from:
                    expected = previous[(role['id'],name)]
                    for key in row:
                        if key in ('sample_mse','zero_score_mse'):
                            if not np.isclose(row[key],expected[key],atol=1e-12,rtol=1e-12):
                                raise ValueError('Reference sample loss differs')
                        elif row[key] != expected[key]:
                            raise ValueError('Reference prediction/counts differ: '+key)
                rows.append(row)
                part.append(row)
                del data,x,pred,gt
            fit_summary = summary(r['full_cube'] for r in part)
            eval_rows = [r for r in heldout if r['head'] == role['id']]
            if sorted(r['name'] for r in eval_rows) != sorted(role['evaluation']):
                raise ValueError('Heldout membership differs')
            evaluation = summary(r['gt_raw_selector_cube'] for r in eval_rows)
            if abs(evaluation['delta_miou_pp']-old['heads'][role['id']]['delta_miou_pp']) > 1e-12:
                raise ValueError('Saved heldout score differs')
            for ns,stat in [(role['fit'],fit_summary),(role['evaluation'],evaluation)]:
                comp = core_counts([complement[n] for n in ns])
                stat['core_recoverable_by_class'] = comp['core_recovered']
                stat['unused_core_by_class'] = [a-b for a,b in zip(comp['core_recovered'],stat['corrected_by_class'])]
                if min(stat['unused_core_by_class']) < 0:
                    raise ValueError('Correction outside core oracle')
            heads[role['id']] = dict(role=role,fit=fit_summary,evaluation=evaluation,
                sampled_fit=summary(r['sample_cube'] for r in part),unsampled_fit=summary(r['unsampled_cube'] for r in part),
                fit_sample_mse=float(np.mean(mse)),zero_score_sample_mse=float(np.mean(zero)),
                fit_minus_evaluation_delta_pp=fit_summary['delta_miou_pp']-evaluation['delta_miou_pp'],fit_labels_accessed=sorted(labels.accessed))
            print(role['id'],'fit/eval delta pp',round(fit_summary['delta_miou_pp'],6),round(evaluation['delta_miou_pp'],6),flush=True)
        diagnosis = dict(heads=heads,fit_visits=len(rows),fit_unique_views=len({r['name'] for r in rows}),heldout_unique_views=len(heldout),
            fit_visit_pooled_descriptive_only=summary(r['full_cube'] for r in rows),heldout_pooled=summary(r['gt_raw_selector_cube'] for r in heldout),
            provenance=proof,boundary='In-sample fit diagnostics and historically reused development heldout; no new model, selection, independent evaluation or deployable oracle')
        write_new_json(a.output/'per_view.json',rows)
        write_new_json(a.output/'diagnosis.json',diagnosis)
        ctx.finish()
        verify_hashes(ROOT,{CONFIG:rec['config_sha256'],**code,**c['input_hashes'],**old['code_sha256']})
        verify_hashes(parent,read(parent/'predictions_sealed.json')['files'])
        guard()
        rec.update(status='completed_fit_gap_diagnostic',seconds=time.time()-started,combined_stage_seconds=time.time()-started+prior_seconds,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            reference_full_fit_passed=bool(a.reference_from),fit_visits=len(rows),frozen_rgb_verified=293,frozen_artifacts_verified=18,
            artifacts={n:sha256(a.output/n) for n in ('per_view.json','diagnosis.json')})
        write_new_json(a.output/'result.json',rec)
        print(rec['status'],rec['seconds'],flush=True)
    except BaseException as exc:
        write_new_json(a.output/'failure.json',dict(type=type(exc).__name__,message=str(exc),seconds=time.time()-started))
        raise


if __name__ == '__main__':
    main()
