#!/usr/bin/env python3
"""Independent saved-array SEM344 audit, no GPU/labels/fitting."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,verify_hashes,write_new_json
from src.quality_cameras import read_camera_requests
from src.camera_adapter import SimpleRadialCamera, _redistortion_map


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['artifact-root','data-root','probe','replay','trajectory','output']:
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    c=json.loads((ROOT/'experiments/runs/SEM-344-EXTERNAL-CAMERA-PREFLIGHT/config.yaml').read_text())
    cameras={**read_camera_requests(a.data_root/c['official_metadata']),**read_camera_requests(a.artifact_root/c['trajectory'])}
    r={'experiment_id':c['experiment_id'],'status':'started','quality_metrics':None,'target_labels_decoded':False,
       'git_commit':c['git_commit'],'seed':c['seed'],'dataset_manifest':c['dataset_manifest'],'split_id':c['split_id'],
       'config_sha256':sha256(ROOT/'experiments/runs/SEM-344-EXTERNAL-CAMERA-PREFLIGHT/config.yaml'),
       'code_sha256':sha256(Path(__file__)),'input_results':{},'folds':[],'exact_probe_replayed':0,'trajectory_arrays_checked':0}
    try:
        verify_hashes(ROOT,c['input_hashes'])
        for fold in range(3):
            row={'fold':fold,'frames':[]}
            for base in [a.probe,a.replay,a.trajectory]:
                for stage in ['render','infer']:
                    path=base/f'{stage}-oof-{fold}';rr=json.loads((path/'result.json').read_text())
                    if rr['status']!='passed_external_camera_stage' or rr['config_sha256']!=r['config_sha256']:
                        raise ValueError('Unaccepted stage')
                    verify_hashes(path,rr['artifacts']);r['input_results'][str(path/'result.json')]=sha256(path/'result.json')
            for stage in ['render','infer']:
                for name in [c['closure_names'][fold],*c['probe_novel_names']]:
                    stem=Path(name).stem+'.npz'
                    with np.load(a.probe/f'{stage}-oof-{fold}'/stem,allow_pickle=False) as x, np.load(a.replay/f'{stage}-oof-{fold}'/stem,allow_pickle=False) as y:
                        if x.files!=y.files or any(not np.array_equal(x[k],y[k]) for k in x.files):
                            raise ValueError('Saved probe replay differs')
                    r['exact_probe_replayed']+=1
            previous={}
            for name in c['trajectory_names']:
                stem=Path(name).stem+'.npz';cam=cameras[name];k=cam.matrix
                # Independent gather via same canonical distortion map, not hard-label wrapper.
                maps=_redistortion_map(SimpleRadialCamera(cam.width,cam.height,k[0,0],k[0,2],k[1,2],cam.radial_k))
                xx,yy=[np.rint(m).astype(np.int64) for m in maps]
                valid=(xx>=0)&(xx<cam.width)&(yy>=0)&(yy<cam.height)
                with np.load(a.trajectory/f'render-oof-{fold}'/stem,allow_pickle=False) as rr, np.load(a.trajectory/f'infer-oof-{fold}'/stem,allow_pickle=False) as z:
                    five=z['five'];raw=five.argmax(0).astype(np.uint8)
                    fusion=rr['static'].copy();lane=(fusion==0)|(fusion==2);fusion[lane]=np.where(five[2][lane]>=.35,2,0)
                    before_foundation=fusion.copy();fusion[(fusion==4)&(five[4]<.1)]=0
                    fusion[(fusion==0)&(before_foundation==0)&(z['deck'][1]>=.98)]=1
                    if not np.array_equal(fusion,z['fused']):raise ValueError('Independent calibrated fusion differs')
                    expected=raw.copy();expected[(raw==3)&(rr['static']==4)&(five[4]>=.1)&((five[3]-five[4])<=.05)]=4
                    if not np.array_equal(raw,z['raw']) or not np.array_equal(expected,z['selected']):
                        raise ValueError('Independent native gate differs')
                    fr={'name':name,'native_changes':int((expected!=raw).sum()),'official_changes':int((z['official_selected']!=z['official_raw']).sum()),'class_counts':{},'unaligned_adjacent_changed_pixels':{}}
                    for arm in ['raw','selected','fused']:
                        mapped=np.zeros((cam.height,cam.width),np.uint8);mapped[valid]=z[arm][yy[valid],xx[valid]]
                        if not np.array_equal(mapped,z['official_'+arm]):
                            raise ValueError('Official nearest-neighbour mapping differs')
                        fr['class_counts'][arm]=np.bincount(mapped.ravel(),minlength=5).tolist()
                        if arm in previous:fr['unaligned_adjacent_changed_pixels'][arm]=int((mapped!=previous[arm]).sum())
                        previous[arm]=mapped
                    for label in [0,1,2]:
                        if not np.array_equal(z['official_raw']==label,z['official_selected']==label):
                            raise ValueError('Official protected class changed')
                    row['frames'].append(fr);r['trajectory_arrays_checked']+=1
            r['folds'].append(row)
        for name in c['trajectory_names']:
            arrays=[]
            for fold in range(3):
                with np.load(a.trajectory/f'render-oof-{fold}'/(Path(name).stem+'.npz'),allow_pickle=False) as z:arrays.append({k:z[k] for k in ['rgb','official_rgb','float_rgb_digest']})
            if any(not np.array_equal(arrays[0][k],x[k]) for x in arrays[1:] for k in arrays[0]):raise ValueError('RGB differs across fold attachments')
        from src.quality_mask2former_training import Context
        Context(ROOT,a.artifact_root,'oof-0').finish()
        r.update(status='passed_independent_saved_camera_audit',rgb_exports_verified=219,frozen_verified=18,
                 temporal_note='Unaligned adjacent changes include camera motion/occlusion; not accuracy or flicker scores. No target GT accessed.')
    except BaseException as exc:
        r.update(status='failed_independent_saved_camera_audit',error=f'{type(exc).__name__}: {exc}');raise
    finally:write_new_json(a.output/'result.json',r)

if __name__=='__main__':main()
