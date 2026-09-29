#!/usr/bin/env python3
"""Hash-bound SEM351 final-model camera-only inference. No training or target-data decoder."""
import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json
CONFIG = 'experiments/runs/SEM-351-FINAL-CAMERA-ACCEPTANCE/config.yaml'
CODE = ['scripts/run_quality_final_camera.py', 'src/quality_external_camera.py', 'src/quality_fusion.py',
        'src/quality_raw_foundation_gate.py', 'src/quality_scaffold.py', 'src/quality_semantic_fields.py',
        'src/quality_cameras.py', 'src/camera_adapter.py', 'src/quality_backbone.py', 'src/quality_experts.py',
        'src/quality_mask2former_final.py', 'src/quality_mask2former.py']


def read(path):
    return json.loads(path.read_text())


def checked_model(run, filename, ctx, kind):
    import torch
    r = read(run / 'result.json')
    verify_hashes(run, {filename: r['artifacts'][filename]})
    p = torch.load(run / filename, map_location='cpu', weights_only=True)
    if p['binding'] != r['binding'] or p['access'] != r['access'] or p['access']['partition'] != ctx.access['partition'] or p['access']['train'] != ctx.access['train']:
        raise ValueError('Model provenance or fold mismatch')
    if kind == 'five':
        if p['binding'] != ctx.binding or p['steps'] != 6000:
            raise ValueError('Different Mask2Former checkpoint')
    else:
        if p['binding']['identity'] != ctx.binding['rgb_identity'] or p['binding']['seed'] != ctx.cfg['seed']:
            raise ValueError('Static/deck RGB or seed mismatch')
        stepkey = 'trained_steps' if kind == 'static' else 'steps'
        if p[stepkey] != r['trained_steps']:
            raise ValueError('Checkpoint step mismatch')
    return p, r


def guard(a, c, ctx):
    from src.quality_mask2former_final import enforce_budget
    ctx.guard()
    return enforce_budget(a.batch, shutil.disk_usage(ROOT).free, pending_bytes=45*1024**2,
                          maximum_bytes=c['resource']['maximum_new_gib']*1024**3)


def save_arrays(out, name, arrays, repeat):
    import numpy as np
    path = out / (Path(name).stem + '.npz')
    if repeat:
        with np.load(repeat / path.name, allow_pickle=False) as z:
            if set(z.files) != set(arrays) or any(not np.array_equal(z[k], v) for k, v in arrays.items()):
                raise ValueError('Fresh-process array replay differs: ' + name)
    else:
        np.savez_compressed(path, **arrays)


def execute(a, c, rec):
    import numpy as np
    import torch
    from PIL import Image
    from src.quality_mask2former_final import FinalContext
    from src.quality_cameras import read_camera_requests
    from src.quality_external_camera import pinhole, official_labels, check_protected, projection_error
    from src.quality_raw_foundation_gate import apply_gate
    from src.quality_fusion import pipeline
    ctx = FinalContext(ROOT, a.artifact_root, ROOT/c['final_batch'])
    rec['runtime'] = ctx.setup_runtime()
    rec['binding'] = ctx.binding
    rec['access'] = ctx.access
    official = read_camera_requests(a.data_root / c['official_metadata'])
    novel = read_camera_requests(a.artifact_root / c['trajectory'])
    known = c['closure_names']
    names = c['names']
    cameras = {n: (novel if n.startswith('frame') else official)[n] for n in names}
    if any((cam.width,cam.height) != tuple(c['resolution']) or cam.model != 'SIMPLE_RADIAL' for cam in cameras.values()):
        raise ValueError('Wrong native official camera grid')
    repeat = a.repeat_from
    if repeat:
        r = read(repeat / 'result.json')
        if r['status'] != 'passed_final_camera_stage' or any(r[k] != rec[k] for k in ['config_sha256','code_sha256','stage']):
            raise ValueError('Replay input is not a matching completed stage')
        verify_hashes(repeat, r['artifacts'])
    if a.stage == 'render':
        from scripts.run_quality_rgb import source_manifest
        from src.quality_scaffold import ScaffoldQualityRenderer
        from src.quality_semantic_fields import PrimitiveSemanticField
        from src.quality_backbone import render_official
        source = a.artifact_root / c['scaffold_source']
        if source_manifest(source) != read(ctx.parent.rgb_run / 'source_manifest.json'):
            raise ValueError('Scaffold source differs from accepted RGB')
        sys.path[:0] = [str(source), str(source/'submodules/diff-gaussian-rasterization'), str(source/'submodules/simple-knn')]
        rgb_access = read(ctx.parent.rgb_run / 'access.json')
        renderer = ScaffoldQualityRenderer(source, ctx.parent.rgb_run/'model', 30000, train_camera_count=len(rgb_access['train']))
        if asdict(renderer.identity) != ctx.binding['rgb_identity']:
            raise ValueError('Frozen renderer identity differs')
        signature = renderer.rgb_signature()
        field = PrimitiveSemanticField(renderer)
        run = ROOT/c['static']
        payload, parent = checked_model(run, 'semantic_field.pt', ctx, 'static')
        field.load_state_dict(payload['field'], renderer)
        del payload
        saved_cameras = read_camera_requests(ctx.parent.rgb_run/'scene/sparse/0')
        for name, cam in cameras.items():
            guard(a, c, ctx)
            native_cam = pinhole(cam, renderer.official_to_model)
            if name in known:
                old = saved_cameras[name]
                if any(not np.allclose(x,y,atol=1e-12,rtol=0) for x,y in [(native_cam.matrix,old.matrix),(native_cam.pose.rotation,old.pose.rotation),(native_cam.pose.translation,old.pose.translation)]):
                    raise ValueError('Official/native known-pose metadata closure differs')
            before = renderer.render_pinhole(native_cam).rgb
            direct = render_official(renderer, cam)['rgb']
            with torch.no_grad():
                static = field.render_probabilities(renderer, native_cam).argmax(0).cpu().numpy().astype(np.uint8)
            after = renderer.render_pinhole(native_cam).rgb
            if not np.array_equal(before, after) or not np.array_equal(direct, render_official(renderer,cam)['rgb']):
                raise ValueError('RGB changed after semantic rendering')
            rgb = np.rint(before*255).astype(np.uint8)
            if name in known:
                for value, path in [(rgb,ctx.prediction_root/'images'/name), (static,run/'predictions'/name)]:
                    with Image.open(path) as im:
                        if not np.array_equal(value, np.array(im)):
                            raise ValueError('Saved native RGB/static closure differs: '+str(path))
            if name.startswith('frame'):
                oldroot = ROOT/c['cached_trajectory_rgb' if name in c['trajectory_names'] else 'cached_probe_rgb']
                with np.load(oldroot/(Path(name).stem+'.npz'),allow_pickle=False) as z:
                    if not np.array_equal(rgb,z['rgb']) or not np.array_equal(direct,z['official_rgb']):
                        raise ValueError('Frozen cached trajectory RGB differs: '+name)
                    if hashlib.sha256(before.tobytes()).digest() != z['float_rgb_digest'].tobytes():
                        raise ValueError('Cached float RGB differs: '+name)
            error = projection_error(cam, renderer.official_to_model, renderer.field_positions.detach().cpu().numpy()[::100])
            if error > 1e-8:
                raise ValueError('Official/model projection mismatch')
            arrays = {'rgb':rgb,'official_rgb':direct,'static':static,
                      'float_rgb_digest':np.frombuffer(hashlib.sha256(before.tobytes()).digest(),np.uint8)}
            save_arrays(a.output,name,arrays,repeat)
            rec['views'].append({'name':name,'projection_error_pixels':error,'rgb_before_after_exact':True,'known_closure':name in known})
        if renderer.rgb_signature() != signature:
            raise ValueError('RGB parameters changed')
        rec['rgb_signature'] = signature
        verify_hashes(run, {'semantic_field.pt':parent['artifacts']['semantic_field.pt']})
    else:
        from src.quality_mask2former import load_official_model, native_predict
        from src.quality_experts import load_expert, predict
        from src.semantic_cable_binary import read_rgb
        rendered = read(a.render_root/'result.json')
        if (rendered['status'] != 'passed_final_camera_stage' or rendered['stage'] != 'render'
                or rendered['config_sha256'] != rec['config_sha256']):
            raise ValueError('Different camera render stage')
        verify_hashes(a.render_root, rendered['artifacts'])
        rec['render_result_sha256'] = sha256(a.render_root/'result.json')
        run = ROOT/c['fit']
        payload, parent = checked_model(run,'model.pt',ctx,'five')
        model, _ = load_official_model(ctx.model_config(),ctx.runtime/'weights/model_final_6b4a3a.pkl')
        model.load_state_dict(payload['model'],strict=True);del payload
        model.cuda()
        five = {}
        for name in names:
            guard(a, c, ctx)
            with np.load(a.render_root/(Path(name).stem+'.npz'),allow_pickle=False) as z:
                five[name] = native_predict(model,z['rgb'])
            if name in known:
                with np.load(run/'probabilities'/(Path(name).stem+'.npz'),allow_pickle=False) as z:
                    if not np.array_equal(five[name],z['probabilities']):
                        raise ValueError('Saved Mask2Former probability closure differs')
        del model;torch.cuda.empty_cache()
        deckrun = ROOT/c['deck']
        payload, deckparent = checked_model(deckrun,'expert.pt',ctx,'deck')
        dc = read(ROOT/'experiments/runs/SEM-331-QUALITY-EXPERTS/config.yaml')
        base = a.artifact_root/dc['model']['relative_weights']
        verify_hashes(base,deckparent['binding']['model_hashes'])
        model = load_expert(base,c['seed'],1).cuda()
        model.load_state_dict(payload['state_dict'],strict=True);del payload
        for name, cam in cameras.items():
            guard(a, c, ctx)
            with np.load(a.render_root/(Path(name).stem+'.npz'),allow_pickle=False) as z:
                rgb,static = z['rgb'],z['static']
            deck = predict(model,rgb,crop=960,stride=480,amp_dtype=torch.bfloat16)
            if name in known:
                with np.load(deckrun/'probabilities'/(Path(name).stem+'.npz'),allow_pickle=False) as z:
                    if not np.array_equal(deck,z['probabilities']):
                        raise ValueError('Saved deck probability closure differs')
            p = five[name];raw = p.argmax(0).astype(np.uint8)
            fused = pipeline(static,p,np.zeros(raw.shape,np.float32),deck[1],c['thresholds'])['deck']
            selected = apply_gate(raw,fused,p,c['maximum_margin'])
            arrays = {'five':p,'deck':deck,'raw':raw,'selected':selected,'fused':fused}
            for arm in ['raw','selected','fused']:
                arrays['official_'+arm] = official_labels(arrays[arm],cam)
            changed = check_protected(raw,selected)
            official_changed = check_protected(arrays['official_raw'],arrays['official_selected'])
            save_arrays(a.output,name,arrays,repeat)
            rec['views'].append({'name':name,'native_changed':changed,'official_changed':official_changed,
                'native_class_counts':{arm:np.bincount(arrays[arm].ravel(),minlength=5).tolist() for arm in ['raw','selected','fused']},
                'official_class_counts':{arm:np.bincount(arrays['official_'+arm].ravel(),minlength=5).tolist() for arm in ['raw','selected','fused']}})
            print(json.dumps({'name':name,'native_changed':changed}),flush=True)
        verify_hashes(run,{'model.pt':parent['artifacts']['model.pt']})
        verify_hashes(deckrun,{'expert.pt':deckparent['artifacts']['expert.pt']})
        verify_hashes(a.render_root,rendered['artifacts'])
    ctx.finish()
    rec.update(status='passed_final_camera_stage',fresh_replay_exact=bool(repeat),rgb_exports_verified=293,frozen_verified=18,
               peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['artifact-root','data-root','batch','output']:
        p.add_argument('--'+name,type=Path,required=True)
    for name in ['render-root','repeat-from']:
        p.add_argument('--'+name,type=Path)
    p.add_argument('--stage',choices=['render','infer'],required=True)
    a=p.parse_args();a.output=a.output.resolve();a.batch=a.batch.resolve()
    if not a.batch.is_relative_to(ROOT/'outputs/quality/external_camera') or not a.output.is_relative_to(a.batch) or a.output==a.batch:
        p.error('Use a new stage directory in the new external-camera batch')
    if a.stage=='infer' and not a.render_root:p.error('Need matching render stage')
    c=read(ROOT/CONFIG)
    with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        a.output.mkdir(parents=True,exist_ok=False)
        rec={'experiment_id':c['experiment_id'],'stage':a.stage,'status':'started',
             'config_sha256':sha256(ROOT/CONFIG),'code_sha256':{n:sha256(ROOT/n) for n in CODE},
             'git_commit':c['git_commit'],'dataset_manifest':c['dataset_manifest'],'split_id':c['split_id'],
             'seed':c['seed'],'target_labels_decoded':False,'real_photos_decoded':False,'trained_steps':0,
             'quality_metrics':None,'views':[],'artifacts':{}}
        started=time.monotonic()
        try:
            for base,hashes in [(ROOT,c['input_hashes']),(ROOT,c['code_sha256']),
                (a.artifact_root,c['artifact_metadata_hashes']),(a.data_root,c['data_metadata_hashes'])]:
                verify_hashes(base,hashes)
            write_new_json(a.output/'started.json',rec)
            for n in [CONFIG,*CODE]:
                dest=a.output/'code'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
            execute(a,c,rec)
            for base,hashes in [(ROOT,c['input_hashes']),(ROOT,c['code_sha256']),
                (a.artifact_root,c['artifact_metadata_hashes']),(a.data_root,c['data_metadata_hashes'])]:
                verify_hashes(base,hashes)
            if sha256(ROOT/CONFIG)!=rec['config_sha256']:raise ValueError('Config changed')
        except BaseException as e:
            rec.update(status='failed_final_camera_stage',error=f'{type(e).__name__}: {e}')
            raise
        finally:
            rec['artifacts']={str(q.relative_to(a.output)):sha256(q) for q in a.output.glob('*.npz')}
            rec['seconds']=time.monotonic()-started
            if a.repeat_from:rec['repeat_from_result_sha256']=sha256(a.repeat_from/'result.json')
            write_new_json(a.output/'result.json',rec)
            print(json.dumps({'status':rec['status'],'output':str(a.output),'seconds':rec['seconds']}),flush=True)


if __name__=='__main__':main()
