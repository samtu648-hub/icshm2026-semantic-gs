"""Matched SegFormer experts, native pixels, hash-bound rendered-only inputs."""
from pathlib import Path
import json
import numpy as np
from src.quality_protocol import CLASSES, sha256, verify_hashes, unique_names


def verify_export(root, names, parent_hash, identity, partition):
    root = Path(root)
    result = json.loads((root / "result.json").read_text())
    if (result.get("status") != "completed_native_expert_rgb_export"
            or result.get("parent_result_sha256") != parent_hash or result.get("identity") != identity
            or result.get("partition") != partition or result.get("pixel_space") != "PINHOLE"
            or result.get("real_rgb_decoded") is not False or result.get("labels_decoded") is not False
            or result.get("repeated_first_camera_exact") is not True):
        raise ValueError("Expert RGB export parent, domain, grid or partition differs")
    entries = result["rendered_images"]
    if unique_names([e["name"] for e in entries], "export") != unique_names(names, "expected export"):
        raise ValueError("Export must cover exactly its registered semantic partition")
    if any((e["width"], e["height"]) != (1320, 989) for e in entries):
        raise ValueError("Expert image resolution changed")
    verify_hashes(root / "images", {e["name"]: e["sha256"] for e in entries})
    verify_hashes(root / "code", result["code_sha256"])
    return result


def expert_counts_weights(counts, positive_class=None):
    counts = np.asarray(counts, np.int64)
    if counts.shape != (5,) or np.any(counts <= 0):
        raise ValueError("Five training-only class counts required")
    if positive_class is not None:
        if positive_class not in (1, 2):
            raise ValueError("Only deck or stay_cable binary experts are registered")
        counts = np.array([counts.sum() - counts[positive_class], counts[positive_class]])
    return counts, np.clip(np.median(counts.astype(float)) / counts, .05, 20).astype(np.float32)


def target_for_expert(label, positive_class=None):
    label = np.asarray(label)
    if label.dtype != np.uint8 or label.ndim != 2 or not label.size or int(label.max()) > 4:
        raise ValueError("Expected official five-class uint8 target")
    if positive_class is None:
        return label
    if positive_class not in (1, 2):
        raise ValueError("Unregistered expert target")
    return (label == positive_class).astype(np.uint8)


def validate_probability(value, classes, shape):
    if (value.shape != (classes, *shape) or value.dtype != np.float32 or not np.isfinite(value).all()
            or value.min() < 0 or value.max() > 1 or not np.allclose(value.sum(0), 1., atol=2e-6, rtol=0)):
        raise ValueError("Invalid native expert probability field")


def load_expert(model_dir, seed, positive_class=None):
    import torch
    from transformers import SegformerForSemanticSegmentation
    names = CLASSES if positive_class is None else ("non_" + CLASSES[positive_class], CLASSES[positive_class])
    torch.manual_seed(seed)
    return SegformerForSemanticSegmentation.from_pretrained(model_dir, local_files_only=True,
        num_labels=len(names), id2label=dict(enumerate(names)), label2id={n: i for i, n in enumerate(names)},
        ignore_mismatched_sizes=True)


def predict(model, rgb, *, crop, stride, amp_dtype):
    import torch
    import torch.nn.functional as F
    from src.semantic2d import tile_starts
    from src.semantic_cable_binary import normalize_rgb
    if rgb.shape != (989, 1320, 3) or rgb.dtype != np.uint8 or stride > crop or stride <= 0 or crop > 989:
        raise ValueError("Expected registered native crop/grid")
    classes = model.config.num_labels
    score, weight = np.zeros((classes, *rgb.shape[:2]), np.float32), np.zeros(rgb.shape[:2], np.float32)
    model.eval()
    with torch.inference_mode():
        for top in tile_starts(rgb.shape[0], crop, stride):
            for left in tile_starts(rgb.shape[1], crop, stride):
                x = normalize_rgb(rgb[top:top+crop, left:left+crop])[None].cuda()
                with torch.autocast("cuda", dtype=amp_dtype):
                    logits = F.interpolate(model(pixel_values=x).logits, size=(crop, crop), mode="bilinear", align_corners=False)
                    prob = logits.float().softmax(1)[0].cpu().numpy()
                score[:, top:top+crop, left:left+crop] += prob
                weight[top:top+crop, left:left+crop] += 1
    if (weight == 0).any():
        raise ValueError("Sliding inference left uncovered pixels")
    score /= weight[None]
    score /= score.sum(0, keepdims=True)
    validate_probability(score, classes, rgb.shape[:2])
    return score
