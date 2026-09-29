#!/usr/bin/env python3
"""SEM358 JSON-only diagnosis; all models/GT/probability decoders remain unused."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,verify_hashes,write_new_json
from src.quality_mask2former_final import FinalContext
from src.quality_static_training import semantic_access
from src.quality_bottleneck import window_stats,core_counts,role_shift,distribution
CONFIG='experiments/runs/SEM-358-SAVED-BOTTLENECK-DIAG/config.yaml'
CODE=['scripts/diagnose_quality_saved_bottleneck.py','src/quality_bottleneck.py','tests/test_quality_bottleneck.py','src/quality_protocol.py','src/quality_semantic_audit.py','src/quality_mask2former_final.py','src/quality_static_training.py']
def read(path):return json.loads(Path(path).read_text())
def guard(output,c,started):
    used=sum(f.stat().st_size for stage in output.parent.glob('SEM-358*') for f in ([stage] if stage.is_file() else stage.rglob('*')) if f.is_file())
    if time.monotonic()-started>c['resources']['maximum_seconds'] or used>c['resources']['maximum_output_mib']*2**20 or shutil.disk_usage(ROOT).free<c['resources']['minimum_free_gib']*2**30:raise RuntimeError('Registered budget reached')
    return used

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--replay-from',type=Path);a=p.parse_args();a.output=a.output.resolve()
    if not a.output.is_relative_to(ROOT/'outputs/quality/bottleneck'):p.error('Isolated bottleneck output required')
    a.output.mkdir(parents=True,exist_ok=False);c=read(ROOT/CONFIG);started=time.monotonic();rec={k:c[k] for k in ('experiment_id','git_commit','seed','dataset_manifest','split_id')};rec.update(config_sha256=sha256(ROOT/CONFIG),code_sha256={n:sha256(ROOT/n) for n in CODE},status='started',training=False,model_inference=False,gt_probability_image_decodes=0,new_111_or74_scores=0)
    write_new_json(a.output/'started.json',rec)
    for n in [CONFIG,*CODE]:
        dst=a.output/'code'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dst)
    try:
        verify_hashes(ROOT,c['input_hashes'])
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=c['git_commit']:raise ValueError('Git drift')
        ctx=FinalContext(ROOT,a.artifact_root,ROOT/c['final_batch']);proof={};training={}
        for fold,paths in c['logs'].items():
            guard(a.output,c,started);base=read(ROOT/paths['result']);rows=[json.loads(line) for line in (ROOT/paths['log']).read_text().splitlines() if line.strip()]
            expected=semantic_access(ctx.parent.split,ctx.parent.oof,read(ctx.parent.rgb_run/'access.json'),fold)
            if any(base['access'][k]!=expected[k] for k in ('partition','train','prediction','guard')) or base['seed']!=c['seed'] or base['binding']['rgb_identity']!=ctx.binding['rgb_identity']:raise ValueError('Same-fold base provenance')
            if sha256(ROOT/paths['log'])!=base['artifacts']['training.jsonl'] or [r['step'] for r in rows]!=list(range(1,6001)):raise ValueError('Log integrity or missing steps')
            if any(r['sample']['name'] not in expected['train'] for r in rows):raise ValueError('Sample outside original train')
            if set(c['views'][fold]) & set(expected['train']):raise ValueError('Fold target in base fit')
            proof[fold]=dict(original_train_count=len(expected['train']),calibration_count=len(c['views'][fold]),target_fit_overlap=0,log_sha256=sha256(ROOT/paths['log']))
            windows={f'{lo:04d}-{hi:04d}':window_stats(rows[lo-1:hi]) for lo,hi in c['windows']}
            view_counter=Counter(r['sample']['name'] for r in rows)
            if sum(x['amp_skips'] for x in windows.values())!=base['amp_skips']:raise ValueError('AMP skip audit')
            if set(view_counter)!=set(expected['train']) or len(view_counter)!=base['unique_sampled_views']:raise ValueError('Saved full view coverage mismatch')
            full=np.array(base['class_counts'],np.int64);crop=np.array(base['sampled_crop_class_counts'],np.int64)
            if full.sum()!=len(expected['train'])*989*1320 or crop.sum()!=6000*768*768:raise ValueError('Class-count pixel totals')
            if any(abs(sum(r['losses'].values())-r['loss'])>1e-4 for r in rows):raise ValueError('Loss component sum')
            lr_error=max(abs(r['lr_factor']-(1-(r['step']-1)/6000)**.9) for r in rows)
            if lr_error>1e-12:raise ValueError('Logged learning-rate schedule differs')
            training[fold]=dict(windows=windows,view_counts=dict(sorted(view_counter.items())),view_count_range=[min(view_counter.values()),max(view_counter.values())],all_original_training_views_sampled=True,full_gt_counts=full.tolist(),sampled_crop_gt_counts=crop.tolist(),full_gt_fraction=(full/full.sum()).tolist(),crop_gt_fraction=(crop/crop.sum()).tolist(),crop_over_full_fraction=((crop/crop.sum())/(full/full.sum())).tolist(),amp_skips=base['amp_skips'],last_vs_penultimate_loss_mean_ratio=windows['5001-6000']['loss']['mean']/windows['4001-5000']['loss']['mean'],lr_schedule_max_error=lr_error,unseen_step_quality_available=False)
        cr=read(ROOT/c['complementarity']/'per_view.json');names=sum(c['views'].values(),[])
        if sorted(r['name'] for r in cr)!=sorted(names):raise ValueError('Original108 complementarity roles')
        raw_by_name={r['name']:np.array(r['raw']['confusion'],np.int64) for r in cr}
        groups={'pooled':names,**c['views']};comp={}
        for group,ns in groups.items():
            part=[r for r in cr if r['name'] in ns];d=core_counts(part);d['core_sources']={}
            for branch in ('static','b2_3k','b2_6k','deck_positive','stay_cable_positive'):
                cube=sum(np.array(r['raw']['source_cubes'][branch],np.int64) for r in part)
                if not np.array_equal(cube.sum(2),np.array(d['raw']['metrics']['confusion'])):raise ValueError('Source cube marginal')
                d['core_sources'][branch]=dict(recoverable_by_class=[int(sum(cube[t,p,t] for p in range(5) if p!=t)) for t in range(5)],potential_harms_by_class=[int(sum(cube[t,t,q] for q in range(5) if q!=t)) for t in range(5)])
            comp[group]=d
        selectors={};by_selector={}
        for eid,path in c['selectors'].items():
            guard(a.output,c,started);record=read(ROOT/path/'result.json');rows=read(ROOT/path/'per_view.json');by_selector[eid]={r['name']:r for r in rows}
            if sorted(by_selector[eid])!=sorted(names):raise ValueError('Selector membership')
            summaries={}
            for group,ns in groups.items():
                part=[by_selector[eid][n] for n in ns]
                for row in part:
                    if row['confusions']['raw']!=raw_by_name[row['name']].tolist():raise ValueError('Raw comparator drift')
                    cube=np.array(row['gt_raw_selector_cube'],np.int64)
                    if not np.array_equal(cube.sum(2),raw_by_name[row['name']]) or not np.array_equal(cube.sum(1),np.array(row['confusions']['selector'])):raise ValueError('Transition cube marginal')
                cube=sum(np.array(r['gt_raw_selector_cube'],np.int64) for r in part);cm=cube.sum(1)
                corrected=[int(sum(cube[t,p,t] for p in range(5) if p!=t)) for t in range(5)];harmed=[int(sum(cube[t,t,q] for q in range(5) if q!=t)) for t in range(5)]
                source=record['pooled'] if group=='pooled' else record['folds'][group]
                if corrected!=source['corrected_by_class'] or harmed!=source['harmed_by_class'] or cm.tolist()!=source['metrics']['selector']['confusion']:raise ValueError('Saved aggregate audit')
                if any(x>y for x,y in zip(corrected,comp[group]['core_recovered'])):raise ValueError('Learned correction escapes core potential')
                summaries[group]=dict(metrics=distribution(cm),corrected_by_class=corrected,harmed_by_class=harmed,unused_core_recoverable_by_class=[a-b for a,b in zip(comp[group]['core_recovered'],corrected)])
            selectors[eid]=summaries
        shifts={}
        for role in c['roles']:
            fit=sum(raw_by_name[n] for n in role['fit']);evaluation=sum(raw_by_name[n] for n in role['evaluation']);saved=read(ROOT/c['selectors']['357']/role['id']/'fit.json')
            if saved['role']!=role or saved['fit_raw_confusion']!=fit.tolist() or set(role['fit'])&(set(role['evaluation'])|set(role['guard'])):raise ValueError('Meta role or fit counts')
            shifts[role['id']]=dict(role=role,**role_shift(fit,evaluation),selector_delta_pp={eid:read(ROOT/path/'result.json')['heads'][role['id']]['delta_miou_pp'] for eid,path in c['selectors'].items()})
        result=dict(training=training,complementarity=comp,selectors=selectors,role_shift=shifts,provenance=proof,boundary='Saved adaptive development evidence only; no hypothetical new candidate or independent generalization claim')
        write_new_json(a.output/'diagnosis.json',result)
        if a.replay_from:
            old=read(a.replay_from/'result.json')
            if old['config_sha256']!=rec['config_sha256'] or old['code_sha256']!=rec['code_sha256'] or result!=read(a.replay_from/'diagnosis.json'):raise ValueError('Independent process replay differs')
        ctx.finish();verify_hashes(ROOT,{CONFIG:rec['config_sha256'],**rec['code_sha256'],**c['input_hashes']});used=guard(a.output,c,started)
        rec.update(status='completed_saved_diagnosis',diagnosis_sha256=sha256(a.output/'diagnosis.json'),seconds=time.monotonic()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,aggregate_output_bytes=used,frozen_rgb_verified=293,frozen_artifacts_verified=18,log_rows_audited=18000,windows_audited=18,views_audited=108,role_pairs=9,replay_exact=bool(a.replay_from))
        write_new_json(a.output/'result.json',rec);print(json.dumps({k:rec[k] for k in ('status','seconds','peak_rss_mib','log_rows_audited','replay_exact')},indent=2))
    except BaseException as exc:
        write_new_json(a.output/'failure.json',dict(type=type(exc).__name__,message=str(exc),seconds=time.monotonic()-started));raise
if __name__=='__main__':main()
