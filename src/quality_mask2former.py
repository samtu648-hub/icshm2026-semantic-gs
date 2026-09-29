"""SEM340 native semantic adapter; lazy imports keep CPU contract tests light."""
from pathlib import Path
from collections import OrderedDict
import pickle
import numpy as np

from src.quality_experts import validate_probability
from src.quality_protocol import sha256

WEIGHT_SHA256 = "f0c081e4fb76d3387141341cd799f1a615aa8178d6873157e1d11e0c8773ef38"
CLASS_KEYS = {"sem_seg_head.predictor.class_embed.weight", "sem_seg_head.predictor.class_embed.bias"}


def upgrade_official_keys(saved):
    """Exact author's v1->v2 query rename; never substitute a random query tensor."""
    old = "sem_seg_head.predictor.static_query.weight"
    new = "sem_seg_head.predictor.query_feat.weight"
    out = dict(saved)
    if old in out:
        if new in out:
            raise ValueError("Ambiguous old/new pretrained query keys")
        out[new] = out.pop(old)
        return out, {old: new}
    return out, {}


def versioned_state(state, current):
    """Keys already validated in current schema: prevent a second legacy rename.

    Official .pkl strips PyTorch metadata even though pixel_decoder keys already
    have the v2 prefix. Author head hooks would add that prefix a second time.
    """
    out = OrderedDict(state)
    out._metadata = current._metadata
    return out


def semantic_probability(scores):
    """Author scores sum query-class-softmax times mask-sigmoid, NOT class logits.

    Normalize across project classes; no-object is excluded upstream, not mapped
    to background. This explicit adaptation requires its own OOF calibration.
    """
    x = np.asarray(scores)
    if x.ndim != 3 or x.shape[0] != 5 or not x.size or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Expected finite nonnegative five-class semantic scores")
    x = x.astype(np.float32)
    total = x.sum(0, keepdims=True)
    if not np.isfinite(total).all() or (total <= 0).any():
        raise ValueError("Invalid/empty semantic mass; do not fabricate background")
    p = x / total
    validate_probability(p, 5, x.shape[1:])
    return p


def class_masks(label):
    x = np.asarray(label)
    if x.ndim != 2 or x.dtype != np.uint8 or not x.size or x.max() > 4:
        raise ValueError("Official native class IDs 0..4 required")
    classes = np.unique(x).astype(np.int64)
    return classes, x[None] == classes[:, None, None]


def config_for_model(upstream, relative_config, activation_checkpoint):
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from mask2former import add_maskformer2_config
    c = get_cfg()
    add_deeplab_config(c)
    add_maskformer2_config(c)
    # The pinned and reviewed official chain contains two Python-eval YAML tags
    # for resize lists. No user-provided config is accepted at this boundary.
    c.merge_from_file(str(Path(upstream) / relative_config))
    c.MODEL.DEVICE = "cpu"
    c.MODEL.WEIGHTS = ""  # no implicit network loading/backbone-only substitution
    c.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 5
    c.MODEL.SWIN.USE_CHECKPOINT = activation_checkpoint
    c.DATASETS.TRAIN = ("sem340_training_probe",)
    c.DATASETS.TEST = ()
    c.INPUT.CROP.SIZE = (768, 768)
    c.INPUT.SIZE_DIVISIBILITY = 32
    c.SOLVER.IMS_PER_BATCH = 1
    c.freeze()
    return c


def load_official_model(cfg, checkpoint):
    import torch
    from detectron2.modeling import build_model
    if sha256(checkpoint) != WEIGHT_SHA256 or Path(checkpoint).stat().st_size != 866036678:
        raise ValueError("Official ADE semantic weight digest/size mismatch")
    # Pickle is executable: only the exact hash-verified official download above.
    with Path(checkpoint).open("rb") as stream:
        saved = pickle.load(stream)["model"]
    saved, renamed = upgrade_official_keys(saved)
    model = build_model(cfg)
    current = model.state_dict()
    if not CLASS_KEYS <= saved.keys() or not CLASS_KEYS <= current.keys():
        raise ValueError("Unexpected official classifier topology")
    if (saved["sem_seg_head.predictor.class_embed.weight"].shape != (151, 256)
            or saved["sem_seg_head.predictor.class_embed.bias"].shape != (151,)):
        raise ValueError("Not the complete ADE150 semantic checkpoint")
    # Criterion weights are class-count-dependent, non-learned loss metadata.
    allowed_missing = CLASS_KEYS | {"criterion.empty_weight"}
    unknown = set(saved) - set(current)
    missing = set(current) - set(saved)
    mismatch = {k for k in saved.keys() & current.keys() if saved[k].shape != tuple(current[k].shape)}
    if unknown or missing - allowed_missing or mismatch - allowed_missing:
        raise ValueError(f"Non-classifier mismatch: unknown={unknown}, missing={missing}, shapes={mismatch}")
    loaded = versioned_state({k: torch.as_tensor(v) for k, v in saved.items() if k not in allowed_missing}, current)
    result = model.load_state_dict(loaded, strict=False)
    if set(result.missing_keys) - allowed_missing or result.unexpected_keys:
        raise ValueError(f"Unexpected pretrained model load failure: {result}")
    for k, v in loaded.items():
        if not torch.equal(model.state_dict()[k], v):
            raise ValueError("Pretrained tensor not preserved: " + k)
    return model, {"official_tensors": len(saved), "loaded_tensors": len(loaded),
                   "initialized_project_keys": sorted(result.missing_keys), "all_loaded_tensors_exact": True,
                   "author_version_rename": renamed}


def author_optimizer(model, cfg):
    """Author's parameter grouping; clipping is done explicitly after AMP unscale."""
    import torch
    norm = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm,
            torch.nn.GroupNorm, torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d, torch.nn.LayerNorm, torch.nn.LocalResponseNorm)
    groups, seen = [], set()
    for name, module in model.named_modules():
        for key, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad or parameter in seen:
                continue
            seen.add(parameter)
            lr, decay = cfg.SOLVER.BASE_LR, cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in name:
                lr *= cfg.SOLVER.BACKBONE_MULTIPLIER
            if "relative_position_bias_table" in key or "absolute_pos_embed" in key:
                decay = 0.
            if isinstance(module, norm):
                decay = cfg.SOLVER.WEIGHT_DECAY_NORM
            if isinstance(module, torch.nn.Embedding):
                decay = cfg.SOLVER.WEIGHT_DECAY_EMBED
            groups.append({"params": [parameter], "lr": lr, "weight_decay": decay})
    return torch.optim.AdamW(groups, cfg.SOLVER.BASE_LR)


def native_predict(model, rgb):
    import torch
    from src.semantic2d import tile_starts
    if rgb.shape != (989, 1320, 3) or rgb.dtype != np.uint8:
        raise ValueError("Native1320x989 RGB required")
    score = np.zeros((5, 989, 1320), np.float32)
    count = np.zeros((989, 1320), np.float32)
    model.eval()
    with torch.inference_mode():
        for y in tile_starts(989, 768, 512):
            for x in tile_starts(1320, 768, 512):
                crop = torch.from_numpy(rgb[y:y+768, x:x+768].transpose(2, 0, 1).copy()).float()
                model_input = [{"image": crop, "height": 768, "width": 768}]
                with torch.autocast("cuda", dtype=torch.float16):
                    scores = model(model_input)[0]["sem_seg"]
                if scores.device.type != "cuda":
                    raise RuntimeError("Author OOM fallback reached CPU; not an all-GPU preflight")
                try:
                    probability = semantic_probability(scores.float().cpu().numpy())
                except ValueError as error:
                    if str(error) != "Invalid/empty semantic mass; do not fabricate background":
                        raise
                    # A project-class score can underflow to exact zero in the
                    # FP16 inference path. Re-evaluate only that unchanged tile
                    # in FP32; never invent a background distribution.
                    with torch.autocast("cuda", enabled=False):
                        scores = model(model_input)[0]["sem_seg"]
                    probability = semantic_probability(scores.float().cpu().numpy())
                    print(
                        {"event": "fp32_empty_mass_retry", "top": y, "left": x},
                        flush=True,
                    )
                score[:, y:y+768, x:x+768] += probability
                count[y:y+768, x:x+768] += 1
    if (count == 0).any():
        raise ValueError("Uncovered sliding inference pixels")
    return semantic_probability(score / count[None])
