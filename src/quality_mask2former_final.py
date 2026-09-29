"""SEM349 final-fit roles and aggregate disk limits; frozen SEM341 is unchanged."""
import hashlib
import json
import random
import shutil
from pathlib import Path
from src.quality_mask2former_training import Context
from src.quality_static_training import semantic_access
from src.quality_experts import verify_export
from src.quality_protocol import sha256, verify_hashes

CONFIG = 'experiments/runs/SEM-349-MASK2FORMER-FINAL/config.yaml'


def sampling_plan(names, seed, steps=6000):
    if len(names) != 219 or len(set(names)) != 219 or names != sorted(names):
        raise ValueError('Final sampler requires exactly sorted219 training names')
    rng = random.Random(seed)
    rows = []
    for _ in range(steps):
        rows.append(dict(name=names[rng.randrange(len(names))], top=rng.randint(0, 989-768),
                         left=rng.randint(0, 1320-768), flip=rng.random() < .5))
    return rows


def sampling_digest(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, separators=(',', ':')).encode()+b'\n')
    return digest.hexdigest()


def enforce_budget(batch, free_bytes, *, pending_bytes=0, maximum_bytes=4*1024**3, reserve_bytes=100*1024**3):
    used = sum(p.stat().st_size for p in Path(batch).rglob('*') if p.is_file())
    if pending_bytes < 0 or used+pending_bytes > maximum_bytes or free_bytes-pending_bytes < reserve_bytes:
        raise OSError('Aggregate4GiB or free100GiB limit: stop and preserve all artifacts')
    return used


class FinalContext:
    def __init__(self, root, artifact_root, batch):
        self.root, self.artifact_root, self.batch = Path(root), Path(artifact_root), Path(batch)
        self.cfg = c = json.loads((self.root/CONFIG).read_text())
        verify_hashes(self.root, c['input_hashes']); verify_hashes(self.root, c['code_sha256'])
        if c['experiment_id'] != 'SEM-349-MASK2FORMER-FINAL' or c['training']['steps'] != 6000:
            raise ValueError('Wrong final-fit protocol')
        self.parent = Context(root, artifact_root, 'oof-0')
        if c['training'] != self.parent.cfg['training'] or c['seed'] != self.parent.cfg['seed']:
            raise ValueError('Final fit must retain the complete SEM341 training recipe')
        self.access = semantic_access(self.parent.split, self.parent.oof,
                                      json.loads((self.parent.rgb_run/'access.json').read_text()), 'development')
        if (len(self.access['train']) != 219 or len(self.access['prediction']) != 74
                or self.access['train'] != c['train_names'] or self.access['prediction'] != c['prediction_names']):
            raise ValueError('Final train/development roles differ')
        self.export_root = self.parent.export_root
        self.prediction_root = self.root/c['prediction_export']
        self.prediction_export = verify_export(self.prediction_root, self.access['prediction'],
            self.parent.pc['rgb_result_sha256'], self.parent.binding['rgb_identity'], 'development')
        self.runtime = self.parent.runtime
        self.binding = dict(self.parent.binding, partition='development', config_sha256=sha256(self.root/CONFIG),
                            code_sha256=c['code_sha256'], final_input_hashes=c['input_hashes'],
                            prediction_export_sha256=sha256(self.prediction_root/'result.json'),
                            training_role_sha256=hashlib.sha256(json.dumps(self.access,sort_keys=True).encode()).hexdigest())
        self.finish()

    def labels(self):
        from src.quality_static_training import TrainingLabels
        return TrainingLabels(self.artifact_root/'r2/sem_001/labels', self.access['train'],
                              self.parent.label_manifest, self.cfg['resolution'])

    def guard(self, pending_bytes=0):
        verify_hashes(self.root, {CONFIG:self.binding['config_sha256'], **self.cfg['code_sha256'], **self.cfg['input_hashes']})
        self.parent.guard()
        return enforce_budget(self.batch, shutil.disk_usage(self.root).free, pending_bytes=pending_bytes)

    def finish(self):
        self.guard(); self.parent.finish()
        verify_hashes(self.prediction_root/'images', {e['name']:e['sha256'] for e in self.prediction_export['rendered_images']})

    def setup_runtime(self):
        return self.parent.setup_runtime()

    def model_config(self):
        return self.parent.model_config()


def require_approval(path, ctx, preflight):
    receipt = json.loads(Path(path).read_text())
    if (receipt.get('decision') != 'user_approved_one_final219_6000_fit_and_two_replays'
            or receipt.get('config_sha256') != ctx.binding['config_sha256']
            or receipt.get('preflight_sha256') != sha256(preflight)
            or receipt.get('batch') != str(ctx.batch.relative_to(ctx.root))
            or not receipt.get('user_confirmation')):
        raise ValueError('Missing matching explicit final-fit approval receipt')
    return sha256(path)
