"""SEM-311: frozen image encoders and a matched five-class decoder.

Only the input projection has a different width between arms. Native encoder
stride, pretraining and channel width are parts of the feature-source treatment;
this experiment cannot isolate pretraining alone or claim equal parameter count.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

ARMS = ("mit_b2_control", "dinov2_vitb14_candidate")


def state_hash(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class FrozenFeatureSegmenter(nn.Module):
    def __init__(self, backbone, arm, *, width=128, seed=20260805):
        super().__init__()
        if arm not in ARMS or width % 8:
            raise ValueError("Invalid feature arm or decoder width")
        self.arm = arm
        self.backbone = backbone.eval().requires_grad_(False)
        channels = (backbone.config.hidden_sizes[-1] if arm == ARMS[0]
                    else backbone.config.hidden_size)
        torch.manual_seed(seed)
        self.projection = nn.Conv2d(channels, width, 1)
        # Reset after the differently sized projection so common tensors match.
        torch.manual_seed(seed + 1)
        self.decoder = nn.Sequential(
            nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv2d(width, 5, 1),
        )

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, *, pixel_values):
        height, width = pixel_values.shape[-2:]
        if height % 224 or width % 224:
            raise ValueError("SEM-311 input must be divisible by both patch14 and MiT stride32 (224)")
        grid = (height // 16, width // 16)
        with torch.no_grad():
            features = self.backbone(pixel_values=pixel_values).last_hidden_state
            if self.arm == ARMS[1]:
                patch_grid = (height // 14, width // 14)
                count = patch_grid[0] * patch_grid[1]
                if features.ndim != 3 or features.shape[1] != count + 1:
                    raise ValueError("DINOv2 CLS/patch token layout mismatch")
                # Official DINOv2-base has one CLS token followed by H/14*W/14 patches.
                features = features[:, -count:].transpose(1, 2).reshape(
                    features.shape[0], features.shape[-1], *patch_grid)
            elif features.ndim != 4:
                raise ValueError("MiT final feature map must be BCHW")
            features = F.interpolate(features, grid, mode="bilinear", align_corners=False)
        return SimpleNamespace(logits=self.decoder(self.projection(features)))

    def head_state(self):
        return {"projection": self.projection.state_dict(), "decoder": self.decoder.state_dict()}

    def load_head(self, state):
        self.projection.load_state_dict(state["projection"], strict=True)
        self.decoder.load_state_dict(state["decoder"], strict=True)


def load_segmenter(model_dir, arm, *, width=128, seed=20260805):
    from transformers import AutoModel, SegformerModel
    loader = SegformerModel if arm == ARMS[0] else AutoModel
    backbone, loading = loader.from_pretrained(model_dir, local_files_only=True, output_loading_info=True)
    allowed_unused = {"classifier.weight", "classifier.bias"} if arm == ARMS[0] else set()
    if (loading["missing_keys"] or loading["mismatched_keys"] or loading["error_msgs"]
            or set(loading["unexpected_keys"]) - allowed_unused):
        raise ValueError("Incomplete or incompatible official encoder checkpoint")
    if arm == ARMS[1] and (backbone.config.model_type != "dinov2"
                          or backbone.config.patch_size != 14
                          or backbone.config.hidden_size != 768
                          or backbone.config.num_hidden_layers != 12):
        raise ValueError("SEM-311 requires the registered DINOv2 ViT-B/14")
    return FrozenFeatureSegmenter(backbone, arm, width=width, seed=seed)


def quality_gates(metrics, fold_metrics, config):
    """New prospective composite gates; never reinterpret historical runs."""
    base, control, candidate = (metrics[k] for k in ("baseline", *ARMS))
    b, c, m = (r["per_class"]["foundation"] for r in (base, candidate, control))
    gates = {
        "miou_gain": candidate["mIoU_with_background"] - base["mIoU_with_background"] >= config["miou_gain"],
        "foundation_iou_gain": c["iou"] - b["iou"] >= config["foundation_iou_gain"],
        "recall_retention": c["recall"] >= b["recall"] - config["maximum_recall_drop"],
        "recall_floor": c["recall"] >= config["recall_floor"],
        "beats_control_miou": candidate["mIoU_with_background"] - control["mIoU_with_background"] >= config["control_miou_gain"],
        "beats_control_foundation": c["iou"] - m["iou"] >= config["control_foundation_iou_gain"],
    }
    improved = 0
    for fold, row in fold_metrics.items():
        first, second = row["baseline"], row[ARMS[1]]
        delta = second["per_class"]["foundation"]["iou"] - first["per_class"]["foundation"]["iou"]
        improved += int(delta > 0)
        gates[f"fold_{fold}_foundation_safety"] = delta >= -config["maximum_fold_foundation_iou_drop"]
        gates[f"fold_{fold}_miou_safety"] = second["mIoU_with_background"] >= first["mIoU_with_background"] - config["maximum_fold_miou_drop"]
    gates["improves_two_spatial_folds"] = improved >= 2
    return gates
