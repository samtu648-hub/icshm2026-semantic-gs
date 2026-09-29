"""SEM381 frozen DINOv3 feature pyramid for the existing Mask2Former head."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.quality_protocol import sha256


EXPECTED_FILES = {
    "config.json": "3c9cc418f4622fd6d5587fd142b6f3cba0ba6a69f67ced907d8b7f26118451ec",
    "preprocessor_config.json": "960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7",
    "LICENSE.md": "25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e",
    "model.safetensors": "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b",
}
EXPECTED_WEIGHT_BYTES = 342662192
FEATURE_NAMES = ("res2", "res3", "res4", "res5")
FEATURE_LAYERS = (3, 6, 9, 12)
FEATURE_STRIDES = (4, 8, 16, 32)
FEATURE_CHANNELS = (192, 384, 768, 1536)


def validate_model_dir(model_dir: Path) -> dict[str, str]:
    model_dir = Path(model_dir)
    actual = {name: sha256(model_dir / name) for name in EXPECTED_FILES}
    if actual != EXPECTED_FILES:
        raise ValueError("DINOv3 local source digest mismatch")
    if (model_dir / "model.safetensors").stat().st_size != EXPECTED_WEIGHT_BYTES:
        raise ValueError("DINOv3 local weight size mismatch")
    config = json.loads((model_dir / "config.json").read_text())
    expected = {
        "architectures": ["DINOv3ViTModel"],
        "model_type": "dinov3_vit",
        "patch_size": 16,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_register_tokens": 4,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Unexpected DINOv3 ViT-B/16 architecture")
    processor = json.loads((model_dir / "preprocessor_config.json").read_text())
    if (
        processor.get("image_mean") != [0.485, 0.456, 0.406]
        or processor.get("image_std") != [0.229, 0.224, 0.225]
        or not processor.get("do_normalize")
        or not processor.get("do_rescale")
    ):
        raise ValueError("Unexpected DINOv3 RGB normalization")
    return actual


def module_state_sha256(module) -> str:
    """Hash tensor names, dtypes, shapes and bytes without serializing a checkpoint."""
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_dinov3_sfp(model_dir: Path, *, adapter_seed: int = 20260805):
    import torch
    from torch import nn
    from detectron2.layers import ShapeSpec
    from detectron2.modeling import Backbone
    from transformers import AutoModel

    source_hashes = validate_model_dir(model_dir)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(adapter_seed)
        dinov3, loading = AutoModel.from_pretrained(
            model_dir, local_files_only=True, output_loading_info=True
        )
        if any(
            loading[key]
            for key in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
        ):
            raise ValueError(f"Incomplete DINOv3 load: {loading}")

        class FrozenDinoSimpleFeaturePyramid(Backbone):
            def __init__(self, encoder) -> None:
                super().__init__()
                self.dinov3 = encoder.eval().requires_grad_(False)
                self.norms = nn.ModuleList([nn.LayerNorm(768) for _ in FEATURE_LAYERS])
                self.adapters = nn.ModuleDict(
                    {
                        "res2": nn.Sequential(
                            nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2),
                            nn.GELU(),
                            nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2),
                        ),
                        "res3": nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2),
                        "res4": nn.Identity(),
                        "res5": nn.Sequential(
                            nn.MaxPool2d(kernel_size=2, stride=2),
                            nn.Conv2d(768, 1536, kernel_size=1),
                        ),
                    }
                )

            def train(self, mode: bool = True):
                super().train(mode)
                self.dinov3.eval()
                return self

            @property
            def size_divisibility(self) -> int:
                return 32

            def output_shape(self):
                return {
                    name: ShapeSpec(channels=channels, stride=stride)
                    for name, channels, stride in zip(
                        FEATURE_NAMES, FEATURE_CHANNELS, FEATURE_STRIDES
                    )
                }

            def forward(self, pixel_values):
                if pixel_values.ndim != 4:
                    raise ValueError("DINOv3 SFP requires BCHW input")
                height, width = pixel_values.shape[-2:]
                if height % 32 or width % 32:
                    raise ValueError("DINOv3 SFP input must be divisible by 32")
                grid = (height // 16, width // 16)
                count = grid[0] * grid[1]
                with torch.no_grad():
                    hidden = self.dinov3(
                        pixel_values=pixel_values,
                        output_hidden_states=True,
                        return_dict=True,
                    ).hidden_states
                features = {}
                for name, layer, norm in zip(FEATURE_NAMES, FEATURE_LAYERS, self.norms):
                    tokens = norm(hidden[layer].detach())
                    if tokens.ndim != 3 or tokens.shape[1] != count + 5:
                        raise ValueError(
                            "DINOv3 CLS/register/patch token layout mismatch"
                        )
                    patches = (
                        tokens[:, -count:]
                        .transpose(1, 2)
                        .reshape(tokens.shape[0], 768, *grid)
                    )
                    features[name] = self.adapters[name](patches).contiguous()
                expected = {
                    name: (
                        pixel_values.shape[0],
                        channels,
                        height // stride,
                        width // stride,
                    )
                    for name, channels, stride in zip(
                        FEATURE_NAMES, FEATURE_CHANNELS, FEATURE_STRIDES
                    )
                }
                if any(
                    tuple(features[name].shape) != shape
                    for name, shape in expected.items()
                ):
                    raise ValueError("DINOv3 SFP produced an invalid feature pyramid")
                return features

        backbone = FrozenDinoSimpleFeaturePyramid(dinov3)

    if any(parameter.requires_grad for parameter in backbone.dinov3.parameters()):
        raise ValueError("DINOv3 encoder must remain frozen")
    return backbone, {
        "source_sha256": source_hashes,
        "weight_bytes": EXPECTED_WEIGHT_BYTES,
        "loading_info": {key: list(value) for key, value in loading.items()},
        "encoder_parameters": sum(p.numel() for p in backbone.dinov3.parameters()),
        "encoder_tensors": len(backbone.dinov3.state_dict()),
        "encoder_frozen": True,
        "selected_hidden_states": list(FEATURE_LAYERS),
        "feature_names": list(FEATURE_NAMES),
        "feature_strides": list(FEATURE_STRIDES),
        "feature_channels": list(FEATURE_CHANNELS),
        "adapter_seed": adapter_seed,
    }


def load_dinov3_mask2former(
    cfg, official_checkpoint: Path, model_dir: Path, *, adapter_seed: int = 20260805
):
    """Retain the ADE Mask2Former head while replacing and discarding Swin-L."""
    from src.quality_mask2former import load_official_model

    model, official = load_official_model(cfg, official_checkpoint)
    decoder_before = module_state_sha256(model.sem_seg_head)
    backbone, dino = build_dinov3_sfp(model_dir, adapter_seed=adapter_seed)
    model.backbone = backbone
    decoder_after = module_state_sha256(model.sem_seg_head)
    if decoder_before != decoder_after:
        raise ValueError("Replacing Swin changed the retained Mask2Former head")
    import torch

    expected_mean = torch.tensor([123.675, 116.28, 103.53])
    if not torch.allclose(
        model.pixel_mean.flatten().cpu(), expected_mean, atol=1e-5, rtol=0
    ):
        raise ValueError("Mask2Former/DINOv3 ImageNet normalization mismatch")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return model, {
        "official_mask2former": official,
        "retained_sem_seg_head_sha256": decoder_after,
        "retained_sem_seg_head_tensors": len(model.sem_seg_head.state_dict()),
        "discarded_official_swin_backbone": True,
        "dinov3": dino,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "total_parameters": trainable + frozen,
    }
