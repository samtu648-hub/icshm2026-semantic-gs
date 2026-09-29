#!/usr/bin/env python3
"""Post-seal error cohorts on the SAME108 calibration views; no new candidate."""
import argparse
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from audit_sem260_error_budget import manifest_paths,png
from src.semantic2d import read_json,write_json,sha256
from src.semantic_refinement import class_boundary_distance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('artifact-root','run-dir','output-dir'): p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    run=read_json(a.run_dir/'result.json');cfg=read_json(a.run_dir/'source_snapshot/config.yaml')
    if run['status']!='completed_calibration_gate_failed': raise ValueError('Only post-seal failed-run diagnosis')
    if sha256(a.run_dir/'calibration_sealed.json')!=run['calibration_sha256']: raise ValueError('Calibration seal changed')
    access=run['calibration']['target_access']
    if len(access)!=108 or len({r['name'] for r in access})!=108: raise ValueError('Not the same108 calibration')
    base=manifest_paths(a.artifact_root/cfg['parents']['sem160'],'prediction_manifest.json','predictions','prediction','prediction_sha256')
    predictions=read_json(a.run_dir/'prediction_manifest.json');results={arm:{} for arm in cfg['arm_names']}
    for r in access:
        name=r['name'];label=a.artifact_root/cfg['parents']['labels']/name
        if sha256(label)!=r['label_sha256'] or sha256(base[name][0])!=base[name][1]: raise ValueError('Parent/label changed')
        gt,old=png(label),png(base[name][0]);distance=class_boundary_distance(gt,4)
        for arm in cfg['arm_names']:
            rec=predictions[arm][name]
            if sha256(Path(rec['path']))!=rec['sha256']: raise ValueError('Sealed candidate changed')
            new=png(Path(rec['path']))
            masks={'lost_old_true_foundation':(gt==4)&(old==4)&(new==0),
                   'recovered_true_foundation':(gt==4)&(old==0)&(new==4),
                   'removed_background_false_positive':(gt==0)&(old==4)&(new==0),
                   'added_background_false_positive':(gt==0)&(old==0)&(new==4),
                   'all_foundation_false_negative':(gt==4)&(new!=4),
                   'all_foundation_false_positive':(gt!=4)&(new==4)}
            for key,mask in masks.items():
                stat=results[arm].setdefault(key,{'pixels':0,'within4':0,'within8':0,'within16':0})
                stat['pixels']+=int(mask.sum())
                for radius in (4,8,16):stat['within'+str(radius)]+=int((mask&(distance<=radius)).sum())
    for groups in results.values():
        for stat in groups.values():
            for radius in (4,8,16):stat['within'+str(radius)+'_fraction']=stat['within'+str(radius)]/stat['pixels'] if stat['pixels'] else None
    write_json(a.output_dir/'result.json',{'experiment_id':cfg['experiment_id'],'status':'completed_post_seal_calibration_diagnostic',
        'source_result_sha256':sha256(a.run_dir/'result.json'),'code_sha256':sha256(Path(__file__)),
        'scope':'Same108 previously scored calibration only; no new fit, threshold, inference or candidate',
        'label_access':access,'cohorts':results,'seconds':time.perf_counter()-started})
    print(results)


if __name__=='__main__':main()
