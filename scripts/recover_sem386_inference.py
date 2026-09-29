#!/usr/bin/env python3
"""Recover zero-mass FP16 score pixels using the same logits in FP32.

No training, epsilon/background imputation, label lookup or score tuning.
Successful native pixels are preserved exactly. Only invalid zero-mass pixels
use the identical softmax/sigmoid/einsum formula with autocast disabled.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import run_sem386_context_memory as runner
from src.quality_protocol import sha256, write_new_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--diagnose-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(runner.CONFIG.read_text())
    batch = ROOT / config['output_path']
    _, torch, _ = runner.base.shared.setup_runtime(args.artifact_root)
    import src.quality_mask2former as api
    original_loader = api.load_official_model
    events = []

    def loader(*a, **kw):
        model, loaded = original_loader(*a, **kw)
        original_semantic = model.semantic_inference
        def semantic(mask_cls, mask_pred):
            score = original_semantic(mask_cls, mask_pred)
            if not torch.isfinite(score).all() or (score < 0).any():
                raise ValueError('Nonfinite/negative scores are not zero-mass underflow')
            invalid = score.float().sum(0) <= 0
            if invalid.any():
                with torch.autocast('cuda', enabled=False):
                    precise = torch.einsum('qc,qhw->chw', mask_cls.float().softmax(-1)[..., :-1], mask_pred.float().sigmoid())
                if not torch.isfinite(precise).all() or not (precise.sum(0) > 0).all():
                    raise ValueError('FP32 score aggregation did not resolve zero mass')
                events.append(dict(zero_mass_pixels=int(invalid.sum()), original_dtype=str(score.dtype),
                                   original_min_mass=float(score.float().sum(0).min()),
                                   recovered_min_mass=float(precise.sum(0)[invalid].min()),
                                   class_logits_range=[float(mask_cls.min()), float(mask_cls.max())],
                                   mask_logits_range=[float(mask_pred.min()), float(mask_pred.max())]))
                score = torch.where(invalid[None], precise, score.float())
            return score
        model.semantic_inference = semantic
        return model, loaded

    api.load_official_model = loader
    audit_path = batch / ('zero-mass-diagnosis.json' if args.diagnose_only else 'inference-numeric-recovery.json')
    audit = dict(experiment_id=runner.EID, code_sha256=sha256(Path(__file__)),
                 policy='FP32 same-logit aggregation on zero-mass pixels only; no labels, epsilon or imputation', events=events)
    try:
        if args.diagnose_only:
            import fcntl
            from src.semantic_cable_binary import read_rgb
            with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                runtime = ROOT/'outputs/quality/mask2former/SEM-340'
                c = api.config_for_model(runtime/'vendor/Mask2Former', 'configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml', False)
                model, _ = api.load_official_model(c, runtime/'weights/model_final_6b4a3a.pkl')
                model.load_state_dict(torch.load(batch/'mask2former/model-6000.pt', map_location='cpu', weights_only=True)['model'])
                model.cuda()
                result = api.native_predict(model, read_rgb(ROOT/config['rgb_evaluation']/'images/377.png'))
                audit.update(image='377.png', finite=bool(__import__('numpy').isfinite(result).all()), status='diagnosis_complete')
        else:
            runner.base.EID, runner.base.CONFIG = runner.EID, runner.CONFIG
            runner.base.CODE_PATHS = (*runner.base.CODE_PATHS, 'scripts/run_sem386_context_memory.py', 'scripts/recover_sem386_inference.py')
            runner.base.inference(config, args.artifact_root)
            previous = batch/'inference-failed-r1/predictions'
            count = 0
            for p in previous.glob('*.png'):
                if sha256(p) != sha256(batch/'inference/predictions'/p.name):
                    raise ValueError(f'Previously valid prediction changed: {p.name}')
                count += 1
            audit.update(status='completed', previously_valid_images_exact=count)
    except BaseException as e:
        audit.update(status='failed_preserved', error=f'{type(e).__name__}: {e}')
        raise
    finally:
        api.load_official_model = original_loader
        write_new_json(audit_path, audit)
        print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
