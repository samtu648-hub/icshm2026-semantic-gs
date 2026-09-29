#!/usr/bin/env python3
"""SEM351 independent reconstruction of final-camera arrays; no target GT."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,verify_hashes,write_new_json
from src.quality_cameras import read_camera_requests
from src.camera_adapter import SimpleRadialCamera,_redistortion_map
from src.quality_mask2former_final import FinalContext
from src.quality_soft_camera import expert_soft,independent_linear
CONFIG='experiments/runs/SEM-351-FINAL-CAMERA-ACCEPTANCE/config.yaml'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['artifact-root','data-root','batch','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    c=json.loads((ROOT/CONFIG).read_text());read=lambda p:json.loads(p.read_text())
    r={k:c[k] for k in ['experiment_id','git_commit','seed','dataset_manifest','split_id']}
    r.update(status='started',config_sha256=sha256(ROOT/CONFIG),code_sha256=sha256(Path(__file__)),
             target_labels_decoded=False,quality_metrics=None,input_results={},views=[])
    try:
        verify_hashes(ROOT,c['input_hashes']);verify_hashes(ROOT,c['code_sha256'])
        verify_hashes(a.artifact_root,c['artifact_metadata_hashes']);verify_hashes(a.data_root,c['data_metadata_hashes'])
        ctx=FinalContext(ROOT,a.artifact_root,ROOT/c['final_batch'])
        cams={**read_camera_requests(a.data_root/c['official_metadata']),**read_camera_requests(a.artifact_root/c['trajectory'])}
        for stage in ['render','infer']:
            first=read(a.batch/stage/'result.json');fresh=read(a.batch/('fresh-'+stage)/'result.json')
            for folder,item in [(stage,first),('fresh-'+stage,fresh)]:
                if item['status']!='passed_final_camera_stage' or item['config_sha256']!=r['config_sha256']:
                    raise ValueError('Incomplete stage')
                if [v['name'] for v in item['views']]!=c['names']:raise ValueError('Incomplete view list')
                verify_hashes(a.batch/folder,item['artifacts']);r['input_results'][folder]=sha256(a.batch/folder/'result.json')
            if not fresh['fresh_replay_exact'] or fresh['repeat_from_result_sha256']!=r['input_results'][stage]:
                raise ValueError('Wrong fresh source')
            if first['binding']!=ctx.binding or fresh['binding']!=ctx.binding or first['views']!=fresh['views']:
                raise ValueError('Fresh provenance or counts differ')
        for name in c['names']:
            stem=Path(name).stem+'.npz';cam=cams[name];k=cam.matrix
            radial=SimpleRadialCamera(cam.width,cam.height,k[0,0],k[0,2],k[1,2],cam.radial_k)
            maps=_redistortion_map(radial);xx,yy=[np.rint(m).astype(np.int64) for m in maps]
            valid=(xx>=0)&(xx<cam.width)&(yy>=0)&(yy<cam.height)
            with np.load(a.batch/'render'/stem,allow_pickle=False) as rgb,np.load(a.batch/'infer'/stem,allow_pickle=False) as z:
                five,deck=z['five'],z['deck']
                for prob,nc in [(five,5),(deck,2)]:
                    if prob.shape!=(nc,cam.height,cam.width) or prob.dtype!=np.float32 or not np.isfinite(prob).all() or prob.min()<0 or not np.allclose(prob.sum(0),1,atol=2e-5,rtol=0):
                        raise ValueError('Invalid native probability tensor')
                raw=five.argmax(0).astype(np.uint8);fused=rgb['static'].copy()
                lane=(fused==0)|(fused==2);fused[lane]=np.where(five[2][lane]>=.35,2,0)
                previous=fused.copy();fused[(fused==4)&(five[4]<.1)]=0
                fused[(fused==0)&(previous==0)&(deck[1]>=.98)]=1
                selected=raw.copy();selected[(raw==3)&(rgb['static']==4)&(five[4]>=.1)&((five[3]-five[4])<=.05)]=4
                row={'name':name,'class_counts':{},'native_changes':int((selected!=raw).sum()),
                     'official_changes':int((z['official_selected']!=z['official_raw']).sum())}
                for arm,value in [('raw',raw),('selected',selected),('fused',fused)]:
                    if not np.array_equal(value,z[arm]):raise ValueError('Independent native reconstruction differs')
                    official=np.zeros(value.shape,np.uint8);official[valid]=value[yy[valid],xx[valid]]
                    if not np.array_equal(official,z['official_'+arm]):raise ValueError('Independent hard gather differs')
                    row['class_counts'][arm]=np.bincount(official.ravel(),minlength=5).tolist()
                for label in [0,1,2]:
                    if not np.array_equal(z['official_raw']==label,z['official_selected']==label):raise ValueError('Protected class differs')
                soft,logs=expert_soft(five,radial);independent=independent_linear(five,maps)
                err=float(np.abs(soft-independent).max())
                if err>2e-6 or not np.isfinite(logs).all():raise ValueError('Soft/log contract differs')
                soft_labels=soft.argmax(0).astype(np.uint8)
                row.update(soft_independent_max_abs_error=err,hard_soft_disagreements=int((soft_labels!=z['official_raw']).sum()),
                           hard_bg_to_soft_stay_cable=int(((z['official_raw']==0)&(soft_labels==2)).sum()),
                           hard_stay_cable_to_soft_bg=int(((z['official_raw']==2)&(soft_labels==0)).sum()))
                r['views'].append(row)
        ctx.finish()
        r.update(status='passed_independent_final_camera_audit',rgb_exports_verified=293,frozen_verified=18,
                 fresh_render_views=len(c['names']),fresh_inference_views=len(c['names']),
                 trajectory_views=len(c['trajectory_names']),soft_scope='raw expert only; original hard unchanged; no fused probabilities',
                 temporal_note='No GT or motion-aligned temporal metric; area/proposals are descriptive only.')
    except BaseException as exc:
        r.update(status='failed_final_camera_audit',error=f'{type(exc).__name__}: {exc}');raise
    finally:write_new_json(a.output/'result.json',r)


if __name__=='__main__':main()
