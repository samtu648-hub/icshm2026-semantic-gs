#!/usr/bin/env python3
"""Hash-bound SEM348 CPU soft-camera audit without target labels or inference."""
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import cv2
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.quality_cameras import read_camera_requests
from src.camera_adapter import SimpleRadialCamera, _redistortion_map
from src.quality_external_camera import official_labels
from src.quality_soft_camera import expert_soft, independent_linear
CONFIG = 'experiments/runs/SEM-348-SOFT-CAMERA-AUDIT/config.yaml'


def read(p):
    return json.loads(p.read_text())


def digest(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--replay', type=Path)
    a = p.parse_args()
    if not a.output.resolve().is_relative_to(ROOT/'outputs/quality/external_camera'):
        p.error('Use a new external_camera directory')
    a.output.mkdir(parents=True, exist_ok=False)
    c = read(ROOT/CONFIG)
    r = {k: c[k] for k in ['experiment_id', 'git_commit', 'seed', 'dataset_manifest', 'split_id']}
    r.update(status='started', started_utc=datetime.now(timezone.utc).isoformat(),
             config_sha256=sha256(ROOT/CONFIG), code_sha256=c['code_sha256'],
             opencv_version=cv2.__version__, numpy_version=np.__version__,
             target_labels_decoded=0, new_training=False, new_inference=False,
             quality_metrics=None, automatic_policy_change=False, replay=bool(a.replay))
    try:
        verify_hashes(ROOT, c['input_hashes']); verify_hashes(ROOT, c['code_sha256'])
        parent = read(ROOT/c['parent_config'])
        verify_hashes(a.artifact_root, parent['artifact_metadata_hashes'])
        cameras = read_camera_requests(a.artifact_root/parent['trajectory'])
        pc = read(ROOT/c['preflight_config']); export = ROOT/pc['rgb_export']
        verify_hashes(export, {'result.json': pc['rgb_export_result_sha256']})
        rgb = {e['name']: e['sha256'] for e in read(export/'result.json')['rendered_images']}
        frozen = {e['path']:e['sha256'] for e in read(ROOT/c['frozen_manifest'])['artifacts']}
        def guard():
            if shutil.disk_usage(ROOT).free < 100*1024**3: raise OSError('100GiB reserve')
            verify_hashes(export/'images', rgb); verify_hashes(a.artifact_root, frozen)
        guard()
        original = None
        if a.replay:
            original = read(a.replay/'result.json')
            assert original['status'] == 'passed_soft_camera_engineering_no_policy_adoption'
            assert original['config_sha256'] == r['config_sha256'] and original['code_sha256'] == r['code_sha256']
            verify_hashes(a.replay, original['artifacts'])
            r['replay_source_sha256'] = sha256(a.replay/'result.json')
        for n in [CONFIG, *c['code_sha256']]:
            dest=a.output/'code'/n; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/n,dest)
        write_new_json(a.output/'started.json',r)
        rows=[]; artifacts={}; max_error=0.; rgb_hashes={}
        for fold in range(3):
            base=ROOT/c['trajectory_batch']/f'infer-oof-{fold}'
            rr=ROOT/c['trajectory_batch']/f'render-oof-{fold}'
            source=read(base/'result.json'); render=read(rr/'result.json')
            assert source['status']==render['status']=='passed_external_camera_stage'
            assert source['render_result_sha256']==sha256(rr/'result.json')
            assert source['fold']==fold and source['binding']['partition']==f'oof-{fold}'
            assert [v['name'] for v in source['views']]==c['names']
            verify_hashes(base,source['artifacts']); verify_hashes(rr,render['artifacts'])
            for name in c['names']:
                cam=cameras[name]; k=cam.matrix
                assert cam.model=='SIMPLE_RADIAL' and [cam.width,cam.height]==c['resolution']
                camera=SimpleRadialCamera(cam.width,cam.height,k[0,0],k[0,2],k[1,2],cam.radial_k)
                with np.load(base/(Path(name).stem+'.npz'),allow_pickle=False) as z:
                    five=z['five']; assert np.array_equal(five.argmax(0),z['raw'])
                    soft,logs=expert_soft(five,camera); label=soft.argmax(0).astype(np.uint8)
                    assert np.array_equal(logs.argmax(0),label)
                    assert np.max(np.abs(np.exp(logs)-soft))<=c['numerical_atol']
                    maps=_redistortion_map(camera)
                    # Nearest probability gather witnesses original raw hard decision.
                    xx,yy=[np.rint(m).astype(np.int64) for m in maps]
                    valid=(xx>=0)&(xx<cam.width)&(yy>=0)&(yy<cam.height)
                    nearest=np.zeros_like(label); nearest[valid]=five[:,yy[valid],xx[valid]].argmax(0)
                    assert np.array_equal(nearest,z['official_raw'])
                    transitions={}
                    for arm in ['raw','selected','fused']:
                        assert np.array_equal(official_labels(z[arm],cam),z['official_'+arm])
                        transitions[arm]=np.bincount(z['official_'+arm].ravel().astype(np.int64)*5+label.ravel(),minlength=25).reshape(5,5).tolist()
                    row={'fold':fold,'name':name,'transitions_hard_to_expert_soft_argmax':transitions,
                         'probability_sha256':digest(soft),'log_probability_sha256':digest(logs),
                         'hard_argmax_sha256':digest(label),'simplex_max_error':float(np.max(np.abs(soft.sum(0)-1)))}
                    if a.replay:
                        independent=independent_linear(five,maps)
                        error=float(np.max(np.abs(independent-soft))); max_error=max(max_error,error)
                        assert error<=c['numerical_atol']
                        assert np.array_equal(independent.argmax(0),label)
                        assert row==original['views'][len(rows)]
                    elif name==c['sample_name']:
                        rel=f'soft-example-oof-{fold}.npz'
                        np.savez_compressed(a.output/rel,expert_probability=soft,expert_log_probability=logs,
                                            expert_soft_argmax=label,existing_raw=z['official_raw'],
                                            existing_selected=z['official_selected'],existing_fused=z['official_fused'])
                        artifacts[rel]=sha256(a.output/rel)
                with np.load(rr/(Path(name).stem+'.npz'),allow_pickle=False) as z:
                    values={v:digest(z[v]) for v in ['rgb','official_rgb','float_rgb_digest']}
                    if name in rgb_hashes: assert values==rgb_hashes[name]
                    rgb_hashes[name]=values
                rows.append(row)
            print(f'oof-{fold}: 24 fixed cameras checked',flush=True)
        totals={arm:sum(np.array(row['transitions_hard_to_expert_soft_argmax'][arm]) for row in rows).tolist() for arm in ['raw','selected','fused']}
        guard()
        r.update(status='passed_soft_camera_engineering_no_policy_adoption',views=rows,
                 artifacts=artifacts,totals=totals,rgb_exports_verified=219,frozen_verified=18,
                 trajectory_rgb_identical_across_folds=24,independent_max_abs_error=max_error if a.replay else None,
                 interpretation='Transitions are disagreement, not corrections/errors. Soft scores belong only to raw expert; no fused-policy probability is defined.')
    except BaseException as exc:
        r.update(status='failed_soft_camera_audit',error=f'{type(exc).__name__}: {exc}'); raise
    finally: write_new_json(a.output/'result.json',r)


if __name__=='__main__':main()
