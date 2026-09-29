"""SEM368 isolated native-FOV raster scaling; old SEM341 code stays immutable."""
import json
from pathlib import Path
import shutil
import numpy as np
from src.quality_mask2former_training import Context as BaseContext, CODE as BASE_CODE
from src.quality_protocol import sha256, verify_hashes

CONFIG = 'experiments/runs/SEM-368-EXPERT-INPUT-SCALE/config.yaml'
CODE = tuple(BASE_CODE) + ('src/quality_expert_scale.py', 'scripts/run_quality_expert_scale.py',
        'scripts/audit_quality_expert_scale.py', 'scripts/queue_quality_expert_scale.py',
        'scripts/check_quality_expert_scale_identity.py', 'tests/test_quality_expert_scale.py')


def resize_pair(rgb, label, size=864):
    import torch
    from torch.nn import functional as F
    if rgb.shape != (768, 768, 3) or label.shape != (768, 768) or rgb.dtype != np.uint8 or label.dtype != np.uint8:
        raise ValueError('Unchanged native768 crop required')
    image = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()
    if size == 768:
        return image, label.copy()
    if size != 864:
        raise ValueError('Only registered864 scale or identity768 allowed')
    image = F.interpolate(image[None], size=(size, size), mode='bilinear', align_corners=False, antialias=False)[0]
    target = F.interpolate(torch.from_numpy(label.copy()).float()[None, None], size=(size, size), mode='nearest-exact')[0, 0]
    return image, target.numpy().astype(np.uint8)


def native_predict(model, rgb, size=864):
    import torch
    from torch.nn import functional as F
    from src.semantic2d import tile_starts
    from src.quality_mask2former import semantic_probability
    if rgb.shape != (989, 1320, 3) or rgb.dtype != np.uint8 or size not in (768, 864):
        raise ValueError('Registered native grid and scale required')
    if model.sem_seg_postprocess_before_inference or model.instance_on or model.panoptic_on:
        raise ValueError('Semantic scores must be combined before native size restoration')
    score = np.zeros((5, 989, 1320), np.float32)
    count = np.zeros((989, 1320), np.float32)
    model.eval()
    with torch.inference_mode():
        for y in tile_starts(989, 768, 512):
            for x in tile_starts(1320, 768, 512):
                crop = torch.from_numpy(rgb[y:y+768, x:x+768].transpose(2, 0, 1).copy()).float()
                if size != 768:
                    crop = F.interpolate(crop[None], size=(size, size), mode='bilinear', align_corners=False, antialias=False)[0]
                with torch.autocast('cuda', dtype=torch.float16):
                    scores = model([{'image': crop, 'height': 768, 'width': 768}])[0]['sem_seg']
                if scores.device.type != 'cuda' or scores.shape != (5, 768, 768):
                    raise RuntimeError('Author CPU fallback or native score shape mismatch')
                score[:, y:y+768, x:x+768] += semantic_probability(scores.float().cpu().numpy())
                count[y:y+768, x:x+768] += 1
    if (count == 0).any():
        raise ValueError('Native tile coverage lost')
    return semantic_probability(score / count[None])


class Context(BaseContext):
    def __init__(self, root, artifact_root, partition):
        self.scale_cfg = json.loads((Path(root)/CONFIG).read_text())
        super().__init__(root, artifact_root, partition)
        base_training = self.cfg['training']
        c = self.scale_cfg
        if c['training'] != base_training or c['network_input'] != 864 or c['seed'] != self.cfg['seed']:
            raise ValueError('More than registered raster scale changed')
        approval = json.loads((self.root/c['approval']).read_text())
        verify_hashes(self.root, {CONFIG: approval['config_sha256'], c['scale_protocol']: approval['protocol_sha256'],
                                  c['scale_draft']: approval['draft_sha256']})
        self.hashes.update({CONFIG: sha256(self.root/CONFIG), c['approval']: sha256(self.root/c['approval']),
                           c['scale_protocol']: approval['protocol_sha256'], c['scale_draft']: approval['draft_sha256'],
                           **{n: sha256(self.root/n) for n in CODE}})
        for fold, source in c['baseline_roles'].items():
            folder = self.root/source['run']
            verify_hashes(folder, {'result.json': source['result_sha256'], 'training.jsonl': source['training_trace_sha256'],
                                   'model.pt': source['model_sha256']})
        source = c['baseline_roles'][partition]
        if self.access != source['access']:
            raise ValueError('Original base fold role mismatch')
        self.baseline_trace = [json.loads(line)['sample'] for line in (self.root/source['run']/'training.jsonl').read_text().splitlines()]
        if len(self.baseline_trace) != 6000:
            raise ValueError('Complete baseline data trace missing')
        self.cfg = c
        self.binding = {**self.binding, 'experiment_id': c['experiment_id'], 'config_sha256': sha256(self.root/CONFIG),
                        'code_sha256': {n: sha256(self.root/n) for n in CODE}, 'network_input':864,
                        'baseline_sources': {f: v['result_sha256'] for f, v in c['baseline_roles'].items()}}
        self.finish()

    def guard(self):
        # Science and identity are strict; elapsed/resource estimates are advisory.
        verify_hashes(self.root, self.hashes)
        verify_hashes(self.runtime, self.runtime_hashes)
        if shutil.disk_usage(self.root).free <= 0:
            raise OSError('Actual disk exhausted')

    def finish(self):
        super().finish()
        # Do not import CPU-only fusion runners: they intentionally mask CUDA.
        from src.quality_experts import verify_export
        final = json.loads((self.root/self.scale_cfg['freeze_config']).read_text())
        paired = json.loads((self.root/final['b2_config']).read_text())
        target_root = self.root/paired['study']['target_root']
        verify_export(target_root, final['prediction_names'], sha256(self.rgb_run/'result.json'),
                      self.binding['rgb_identity'], 'development')

    def model_config(self):
        c = super().model_config()
        if (c.MODEL.MASK_FORMER.OVERSAMPLE_RATIO != 3.0 or c.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO != .75
                or c.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE):
            raise ValueError('Original point sampling or score restoration changed')
        return c
