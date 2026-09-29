#!/usr/bin/env python3
"""Independent saved-count audit for SEM342/343; no labels or model replay."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import verify_hashes, sha256, write_new_json


def direct_metrics(cm, recorded):
    cm = np.asarray(cm, np.int64)
    ious = []
    for i, name in enumerate(("background", "deck", "stay_cable", "tower", "foundation")):
        tp = int(cm[i,i]); fp = int(cm[:,i].sum())-tp; fn = int(cm[i].sum())-tp
        expected = {"tp":tp,"fp":fp,"fn":fn,"IoU":tp/(tp+fp+fn) if tp+fp+fn else None,
                    "precision":tp/(tp+fp) if tp+fp else None,"recall":tp/(tp+fn) if tp+fn else None}
        if recorded['per_class'][name] != expected:
            raise ValueError("Independent metric fractions differ")
        ious.append(expected['IoU'])
    for key, values in (("mIoU_with_background",ious),("mIoU_without_background",ious[1:])):
        values=[v for v in values if v is not None]
        expected=sum(values)/len(values) if values else None
        actual=recorded[key]
        if expected is None and actual is not None or expected is not None and (actual is None or abs(expected-actual)>1e-14):
            raise ValueError("Independent mIoU differs")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostic',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); start=time.monotonic()
    a.output=a.output.resolve()
    if not a.output.is_relative_to(ROOT/'outputs/quality/raw_fusion'):
        p.error('Use a new raw_fusion output')
    a.output.mkdir(parents=True,exist_ok=False)
    record={'status':'auditing','script_sha256':sha256(Path(__file__)), 'new_label_decodes':0,'new_model_replays':0}
    try:
        d=json.loads((a.diagnostic/'result.json').read_text());c=json.loads((a.calibration/'result.json').read_text())
        for r,exp in ((d,'SEM-342-RAW-FUSION-DIAG'),(c,'SEM-343-RAW-FOUNDATION-GATE')):
            cfgpath=f'experiments/runs/{exp}/config.yaml'
            verify_hashes(ROOT,{cfgpath:r['config_sha256'],**r['code_sha256']})
            cfg=json.loads((ROOT/cfgpath).read_text());verify_hashes(ROOT,cfg['input_hashes'])
        if d['status']!='completed_readonly_complementarity_diagnostic' or c['status']!='completed_calibration_curves_pending_quality_review':
            raise ValueError('Incomplete source results')
        for role,e in d['evaluation'].items():
            path=a.diagnostic/f'{role}_per_view.json';verify_hashes(a.diagnostic,{path.name:d['artifacts'][path.name]})
            views=json.loads(path.read_text());sumcube=np.zeros((5,5,5),np.int64)
            for view in views.values():
                cube=np.array(view['cube_raw_fusion_truth']);sumcube+=cube
                for axis,key in ((1,'raw'),(0,'calibrated_fusion')):
                    cm=cube.sum(axis=axis).T
                    if cm.tolist()!=view[key]['confusion']:raise ValueError('Cube to confusion mismatch')
                    direct_metrics(cm,view[key])
                corr=sum(int(cube[old,new,new]) for old in range(5) for new in range(5) if old!=new)
                harm=sum(int(cube[old,new,old]) for old in range(5) for new in range(5) if old!=new)
                if (corr,harm)!=(view['corrected'],view['harmed']):raise ValueError('Complementarity counts differ')
            if sumcube.tolist()!=e['total']['cube_raw_fusion_truth']:raise ValueError('View sum differs')
            for k in ('per_partition','provenance','image_tiles'):
                if not np.array_equal(sum(np.array(v['cube_raw_fusion_truth']) for v in e[k].values()),sumcube):raise ValueError('Group sum differs')
        path=a.calibration/'calibration_per_view.json';verify_hashes(a.calibration,{path.name:c['calibration_per_view_sha256']})
        views=json.loads(path.read_text())
        if sorted(views)!=c['calibration_views'] or len(views)!=108:raise ValueError('Unexpected calibration access')
        groups={g:np.zeros((len(c['curves']),5,5),np.int64) for g in ('oof-0','oof-1','oof-2')}
        for view in views.values():
            baseline=np.array(view['curves'][0]['confusion'])
            for j,row in enumerate(view['curves']):
                counts=np.array(row['changed_truth_counts']);expected=baseline.copy()
                expected[:,3]-=counts;expected[:,4]+=counts
                if expected.tolist()!=row['confusion']:raise ValueError('Gate transition algebra differs')
                groups[view['partition']][j]+=expected
        for j,curve in enumerate(c['curves']):
            direct_metrics(sum(groups.values())[j],curve['metrics'])
            for group,cm in groups.items():
                if cm[j].tolist()!=curve['per_partition'][group]['metrics']['confusion']:raise ValueError('Gate fold sum differs')
                direct_metrics(cm[j],curve['per_partition'][group]['metrics'])
        cfg=json.loads((ROOT/'experiments/runs/SEM-343-RAW-FOUNDATION-GATE/config.yaml').read_text())
        if [v['maximum_margin'] for v in c['curves']]!=cfg['maximum_margin_grid']:raise ValueError('Preregistered grid differs')
        pc=json.loads((ROOT/cfg['preflight_config']).read_text())
        export=ROOT/pc['rgb_export'];verify_hashes(export,{'result.json':pc['rgb_export_result_sha256']})
        er=json.loads((export/'result.json').read_text());rgb={v['name']:v['sha256'] for v in er['rendered_images']}
        verify_hashes(export/'images',rgb)
        frozen={v['path']:v['sha256'] for v in json.loads((ROOT/cfg['frozen_manifest']).read_text())['artifacts']}
        verify_hashes(a.artifact_root,frozen)
        checkpoints={}
        for fold in range(3):
            base=ROOT/cfg['batch']/f'full-oof-{fold}';r=json.loads((base/'result.json').read_text())
            selected={name:h for name,h in r['artifacts'].items() if name.endswith(('.pt','.pth'))}
            verify_hashes(base,selected)
            checkpoints.update({str((base/n).relative_to(ROOT)):h for n,h in selected.items()})
        record.update(status='passed_independent_saved_count_and_provenance_audit',
            diagnostic_result_sha256=sha256(a.diagnostic/'result.json'),calibration_result_sha256=sha256(a.calibration/'result.json'),
            diagnostic_views=219,calibration_views=108,calibration_candidates=len(c['curves']),
            rgb_exports_verified=len(rgb),frozen_artifacts_verified=len(frozen),checkpoint_hashes=checkpoints)
    except BaseException as exc:
        record.update(status='failed_audit_preserved',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['seconds']=time.monotonic()-start
        write_new_json(a.output/'result.json',record)
        print(json.dumps({k:record[k] for k in ('status','seconds')}))


if __name__=='__main__':main()
