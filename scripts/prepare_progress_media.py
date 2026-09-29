#!/usr/bin/env python3
"""REPORT002: camera-only frozen RGB110/B2/M2F media, and saved B2 recount.

Each stage owns a fresh output directory. No training, optimizer, or model edits.
Original failures remain on disk; a new --work directory is needed after failure.
"""
import argparse
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, verify_hashes

CONFIG = ROOT / 'experiments/runs/REPORT-002-PROGRESS-WEB/config.yaml'


def read(p):
    return json.loads(Path(p).read_text())


def save(p, data):
    Path(p).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def execute(a, c, out, rec):
    import numpy as np
    from PIL import Image
    frozen = read(ROOT / 'experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json')
    protected = {e['path']: e['sha256'] for e in frozen['artifacts']}
    verify_hashes(a.artifact_root, protected)
    b2run = ROOT / c['b2']
    b2result = read(b2run / 'result.json')
    if a.stage == 'score':
        from src.quality_semantic_audit import independent_confusion as confusion, metrics
        names = b2result['access']['prediction']
        assert len(names) == 74
        cm = np.zeros((5, 5), np.int64)
        rows = []
        label_manifest = ROOT / 'experiments/runs/SEM-001/label_manifest.json'
        label_records = {r['image']: r for r in read(label_manifest)['records']}
        verify_hashes(ROOT, b2result['binding']['input_hashes'])
        oof = read(ROOT / 'experiments/splits/sem_005_oof_spatial_v1.json')
        assert sha256(label_manifest) == oof['label_manifest_sha256']
        for name in names:
            pred = b2run / 'predictions' / name
            target = a.artifact_root / 'r2/sem_001/labels' / name
            key = 'predictions/' + name
            verify_hashes(b2run, {key: b2result['artifacts'][key]})
            assert sha256(target) == label_records[name]['label_png_sha256']
            mat = confusion(np.array(Image.open(target)), np.array(Image.open(pred)))
            cm += mat
            rows.append(dict(name=name, prediction_sha256=sha256(pred), target_sha256=sha256(target), metrics=metrics(mat)))
        rec.update(metrics=metrics(cm), views=rows, source_result_sha256=sha256(b2run/'result.json'),
                   scope='74 historically reused developer views; original raw B2, no fusion; saved prediction recount')
    else:
        import torch
        from src.quality_cameras import read_camera_requests
        from src.quality_external_camera import pinhole, official_labels
        from src.camera_adapter import Sim3
        cameras = read_camera_requests(a.artifact_root / c['trajectory'])
        names = [f'frame{i:04d}.png' for i in range(144)]
        assert sorted(cameras) == names
        rec['trajectory_hashes'] = {p.name: sha256(p) for p in sorted((a.artifact_root/c['trajectory']).iterdir()) if p.is_file()}
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.set_num_threads(4)
        if a.stage == 'render':
            from src.quality_scaffold import ScaffoldQualityRenderer
            from src.quality_backbone import render_official
            from scripts.run_quality_rgb import source_manifest
            source = a.artifact_root / 'r1/third_party/scaffold-gs'
            run = ROOT / c['rgb110']
            parent = read(run/'result.json')
            verify_hashes(run, parent['artifacts'])
            assert source_manifest(source) == read(run/'source_manifest.json')
            sys.path[:0] = [str(source), str(source/'submodules/diff-gaussian-rasterization'), str(source/'submodules/simple-knn')]
            renderer = ScaffoldQualityRenderer(source, run/'model', 30000, train_camera_count=len(read(run/'access.json')['train']))
            assert asdict(renderer.identity) == b2result['binding']['identity']
            signature = renderer.rgb_signature()
            for folder in ['native', 'official']:
                (out/folder).mkdir()
            rec['cached_rgb_exact'] = []
            for i, name in enumerate(names):
                cam = cameras[name]
                rgb = np.rint(renderer.render_pinhole(pinhole(cam, renderer.official_to_model)).rgb*255).astype(np.uint8)
                official = render_official(renderer, cam)['rgb']
                cached = ROOT/'outputs/quality/external_camera/SEM-351-r1/render'/f'{Path(name).stem}.npz'
                if cached.exists():
                    with np.load(cached, allow_pickle=False) as z:
                        assert np.array_equal(rgb,z['rgb']) and np.array_equal(official,z['official_rgb']), name
                    rec['cached_rgb_exact'].append(name)
                Image.fromarray(rgb).save(out/'native'/name)
                Image.fromarray(official).save(out/'official'/name)
                if i in (0, 71, 143):
                    assert np.array_equal(rgb, np.rint(renderer.render_pinhole(pinhole(cam,renderer.official_to_model)).rgb*255).astype(np.uint8))
            assert renderer.rgb_signature() == signature
            verify_hashes(run, parent['artifacts'])
            rec.update(rgb_signature=signature, model_artifacts_exact=True, repeated_rgb_exact=True)
        else:
            render = read(a.work/'render/result.json')
            assert render['status'] == 'completed'
            verify_hashes(a.work/'render', render['artifacts'])
            if a.stage == 'b2':
                from src.quality_experts import load_expert, predict
                cfg = read(ROOT/'experiments/runs/SEM-331-QUALITY-EXPERTS/config.yaml')
                run, result = b2run, b2result
                verify_hashes(run, {'expert.pt':result['artifacts']['expert.pt']})
                payload = torch.load(run/'expert.pt', map_location='cpu', weights_only=True)
                assert payload['binding'] == result['binding'] and payload['access'] == result['access'] and payload['steps'] == 3000
                weights = a.artifact_root/cfg['model']['relative_weights']
                verify_hashes(weights, result['binding']['model_hashes'])
                model = load_expert(weights, cfg['seed']).cuda().eval()
                model.load_state_dict(payload['state_dict'], strict=True)
                del payload
                predict_one = lambda rgb: predict(model,rgb,crop=768,stride=512,amp_dtype=torch.bfloat16)
                checkpoint = 'expert.pt'
            else:
                from src.quality_mask2former_final import FinalContext
                from src.quality_mask2former import load_official_model, native_predict
                ctx = FinalContext(ROOT,a.artifact_root,ROOT/'outputs/quality/mask2former/SEM-349-r1')
                ctx.setup_runtime()
                run = ROOT/c['m2f'];result=read(run/'result.json')
                verify_hashes(run, {'model.pt':result['artifacts']['model.pt']})
                payload=torch.load(run/'model.pt',map_location='cpu',weights_only=True)
                assert payload['binding']==ctx.binding and payload['steps']==6000
                model,_=load_official_model(ctx.model_config(),ctx.runtime/'weights/model_final_6b4a3a.pkl')
                model.load_state_dict(payload['model'],strict=True)
                model.cuda().eval();del payload
                predict_one=lambda rgb:native_predict(model,rgb)
                checkpoint='model.pt'
            # Closure to saved native prediction on a real existing render, no labels.
            closure_name='054.png'
            export=ROOT/'outputs/quality/expert_inputs/Scaffold-development-dualline-20260909-r1/images'
            if a.stage=='m2f':export=ctx.prediction_root/'images'
            closure=predict_one(np.array(Image.open(export/closure_name)))
            saved=run/'predictions'/closure_name
            assert np.array_equal(closure.argmax(0).astype(np.uint8),np.array(Image.open(saved)))
            rec['saved_prediction_closure_exact']=closure_name
            (out/'native').mkdir();(out/'official').mkdir()
            for i,name in enumerate(names):
                rgb=np.array(Image.open(a.work/'render/native'/name))
                probability=predict_one(rgb)
                labels=probability.argmax(0).astype(np.uint8)
                if a.stage=='m2f':
                    old=ROOT/'outputs/quality/external_camera/SEM-351-r1/infer'/f'{Path(name).stem}.npz'
                    if old.exists():
                        with np.load(old,allow_pickle=False) as z:assert np.array_equal(probability,z['five']),name
                Image.fromarray(labels).save(out/'native'/name)
                Image.fromarray(official_labels(labels,cameras[name])).save(out/'official'/name)
                if i%24==0:print(json.dumps({'stage':a.stage,'frame':i}),flush=True)
            verify_hashes(run,{checkpoint:result['artifacts'][checkpoint]})
            verify_hashes(a.work/'render',render['artifacts'])
            rec.update(checkpoint_sha256=result['artifacts'][checkpoint], source_result_sha256=sha256(run/'result.json'), rgb_inputs_exact=True)
            if a.stage=='m2f':ctx.finish()
    verify_hashes(a.artifact_root, protected)
    rec['frozen18_exact']=True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--work',type=Path,default=ROOT/'outputs/progress_web_work/r1')
    p.add_argument('--stage',choices=['render','b2','m2f','score'],required=True)
    p.add_argument('--attempt',default='',help='Optional suffix; preserve earlier failed attempts')
    a=p.parse_args();a.work=a.work.resolve();a.artifact_root=a.artifact_root.resolve()
    c=read(CONFIG);out=a.work/(a.stage + ('-' + a.attempt if a.attempt else ''))
    out.mkdir(parents=True,exist_ok=False)
    rec=dict(experiment_id=c['experiment_id'],stage=a.stage,status='running',trained_steps=0,
             config_sha256=sha256(CONFIG),script_sha256=sha256(Path(__file__)), started=time.time())
    save(out/'started.json',rec)
    try:
        with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
            if a.stage!='score':fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            execute(a,c,out,rec)
        rec.update(status='completed',seconds=time.time()-rec['started'],artifacts={str(x.relative_to(out)):sha256(x) for x in out.rglob('*.png')})
        save(out/'result.json',rec)
        print(json.dumps({'stage':a.stage,'status':'completed','seconds':rec['seconds']}),flush=True)
    except Exception:
        rec.update(status='failed',error=traceback.format_exc());save(out/'failure.json',rec);raise


if __name__=='__main__':main()
