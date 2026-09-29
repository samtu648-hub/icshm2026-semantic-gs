#!/usr/bin/env python3
"""One fixed old training-side model/three-view identity check; no training."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_expert_scale import Context, native_predict
from src.quality_protocol import sha256, verify_hashes, write_new_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--artifact-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output = a.output.resolve()
    if not a.output.is_relative_to(ROOT/'outputs/quality/expert_scale'):
        p.error('Isolated scale output required')
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    r = dict(status='started', new_training=0, target_labels_decoded=False)
    try:
        with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            ctx = Context(ROOT, a.artifact_root, 'oof-0')
            r['runtime'] = ctx.setup_runtime()
            import torch
            import numpy as np
            from PIL import Image
            from src.quality_mask2former import load_official_model
            from src.quality_semantic_audit import check_replay
            from src.semantic_cable_binary import read_rgb
            folder = ROOT/'outputs/quality/mask2former/SEM-341-r1/stability'
            saved = json.loads((folder/'result.json').read_text())
            verify_hashes(folder, saved['artifacts'])
            names = [ctx.access['train'][i] for i in [0, len(ctx.access['train'])//2, -1]]
            if saved['prediction_views'] != names or saved['access'] != ctx.access:
                raise ValueError('Original training-side identity views mismatch')
            model, _ = load_official_model(ctx.model_config(), ctx.runtime/'weights/model_final_6b4a3a.pkl')
            payload = torch.load(folder/'model.pt', map_location='cpu', weights_only=True)
            if payload['binding'] != saved['binding'] or payload['access'] != ctx.access:
                raise ValueError('Original model identity mismatch')
            model.load_state_dict(payload['model'], strict=True)
            del payload
            model.cuda()
            for name in names:
                prob = native_predict(model, read_rgb(ctx.export_root/'images'/name), size=768)
                with np.load(folder/'probabilities'/(Path(name).stem+'.npz'), allow_pickle=False) as z:
                    stored = z['probabilities']
                with Image.open(folder/'predictions'/name) as im:
                    check_replay(prob, stored, np.array(im))
            ctx.finish()
            r.update(status='passed_scale_identity', names=names, probabilities_exact=True, binding=ctx.binding,
                     source_sha256=sha256(folder/'result.json'))
    except BaseException as exc:
        r.update(status='failed_scale_identity', error=repr(exc))
        raise
    finally:
        r['seconds'] = time.monotonic()-start
        write_new_json(a.output/'result.json', r)


if __name__ == '__main__':
    main()
