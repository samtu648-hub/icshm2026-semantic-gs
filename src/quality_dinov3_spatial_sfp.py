"""SEM383 frozen DINOv3 SFP with a lightweight RGB spatial-prior branch."""

from __future__ import annotations

from pathlib import Path

from src.quality_dinov3_mask2former import (
    EXPECTED_WEIGHT_BYTES,
    FEATURE_CHANNELS,
    FEATURE_LAYERS,
    FEATURE_NAMES,
    FEATURE_STRIDES,
    module_state_sha256,
    validate_model_dir,
)


OFFICIAL_ADAPTER_REPOSITORY = "https://github.com/facebookresearch/dinov3"
OFFICIAL_ADAPTER_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
OFFICIAL_ADAPTER_PATH = "dinov3/eval/segmentation/models/backbone/dinov3_adapter.py"
SPATIAL_PRIOR_PARAMETERS = 514048
SFP_PARAMETERS = 3842496


def build_dinov3_spatial_sfp(model_dir: Path, *, adapter_seed: int = 20260805):
    """Build the SEM382 SFP plus a deformable-free RGB spatial-prior residual.

    This is conceptually inspired by the official DINOv3 adapter's convolutional
    spatial prior, but it is not an implementation of that adapter or ViT-Adapter.
    The official deformable Extractors are intentionally omitted so the existing
    Mask2Former feature contract and the 12 GB native-768 protocol remain fixed.
    """
    import torch
    from detectron2.layers import ShapeSpec
    from detectron2.modeling import Backbone
    from torch import nn
    from transformers import AutoModel

    source_hashes = validate_model_dir(model_dir)

    class SpatialStage(nn.Sequential):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=in_channels,
                    bias=False,
                ),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(8, out_channels),
                nn.GELU(),
            )

    class FrozenDinoSpatialFeaturePyramid(Backbone):
        def __init__(self, encoder) -> None:
            super().__init__()
            self.dinov3 = encoder.eval().requires_grad_(False)
            # Construct these modules in the exact SEM382 order before adding the
            # treatment branch, preserving the shared initialization byte-for-byte.
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
            self.spatial_prior = nn.ModuleDict(
                {
                    "res2": nn.Sequential(
                        nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                        nn.GroupNorm(8, 32),
                        nn.GELU(),
                        nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
                        nn.GroupNorm(8, 64),
                        nn.GELU(),
                    ),
                    "res3": SpatialStage(64, 96),
                    "res4": SpatialStage(96, 128),
                    "res5": SpatialStage(128, 192),
                }
            )
            spatial_channels = (64, 96, 128, 192)
            self.spatial_projections = nn.ModuleDict(
                {
                    name: nn.Conv2d(in_channels, out_channels, kernel_size=1)
                    for name, in_channels, out_channels in zip(
                        FEATURE_NAMES, spatial_channels, FEATURE_CHANNELS
                    )
                }
            )
            self.spatial_gates = nn.ParameterDict(
                {
                    name: nn.Parameter(torch.full((channels,), 1e-3))
                    for name, channels in zip(FEATURE_NAMES, FEATURE_CHANNELS)
                }
            )
            self._initialize_spatial_branch()

        def _initialize_spatial_branch(self) -> None:
            for module in (self.spatial_prior, self.spatial_projections):
                for layer in module.modules():
                    if isinstance(layer, nn.Conv2d):
                        nn.init.kaiming_normal_(
                            layer.weight, mode="fan_out", nonlinearity="relu"
                        )
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
                    elif isinstance(layer, nn.GroupNorm):
                        nn.init.ones_(layer.weight)
                        nn.init.zeros_(layer.bias)

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
                raise ValueError("DINOv3 spatial SFP requires BCHW input")
            height, width = pixel_values.shape[-2:]
            if height % 32 or width % 32:
                raise ValueError("DINOv3 spatial SFP input must be divisible by 32")
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
                    raise ValueError("DINOv3 CLS/register/patch token layout mismatch")
                patches = (
                    tokens[:, -count:]
                    .transpose(1, 2)
                    .reshape(tokens.shape[0], 768, *grid)
                )
                features[name] = self.adapters[name](patches).contiguous()

            spatial = pixel_values
            for name in FEATURE_NAMES:
                spatial = self.spatial_prior[name](spatial)
                residual = self.spatial_projections[name](spatial)
                gate = self.spatial_gates[name].view(1, -1, 1, 1)
                features[name] = (features[name] + gate * residual).contiguous()

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
                tuple(features[name].shape) != shape for name, shape in expected.items()
            ):
                raise ValueError(
                    "DINOv3 spatial SFP produced an invalid feature pyramid"
                )
            return features

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
        backbone = FrozenDinoSpatialFeaturePyramid(dinov3)

    if any(parameter.requires_grad for parameter in backbone.dinov3.parameters()):
        raise ValueError("DINOv3 encoder must remain frozen")
    spatial_parameters = sum(
        parameter.numel()
        for name, parameter in backbone.named_parameters()
        if name.startswith(("spatial_prior.", "spatial_projections.", "spatial_gates."))
    )
    if spatial_parameters != SPATIAL_PRIOR_PARAMETERS:
        raise ValueError(
            f"Spatial-prior parameter count mismatch: {spatial_parameters}"
        )
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
        "existing_sfp_parameters": SFP_PARAMETERS,
        "spatial_prior_parameters": spatial_parameters,
        "total_adapter_parameters": SFP_PARAMETERS + spatial_parameters,
        "official_adapter_reference": {
            "repository": OFFICIAL_ADAPTER_REPOSITORY,
            "commit": OFFICIAL_ADAPTER_COMMIT,
            "path": OFFICIAL_ADAPTER_PATH,
            "license": "DINOv3 License",
        },
        "implementation_scope": (
            "custom deformable-free RGB spatial-prior residual on the existing SFP; "
            "not Meta DINOv3 Adapter and not ViT-Adapter"
        ),
    }


def load_dinov3_spatial_mask2former(
    cfg, official_checkpoint: Path, model_dir: Path, *, adapter_seed: int = 20260805
):
    """Retain the ADE Mask2Former head and replace only the DINO SFP backbone."""
    import torch

    from src.quality_mask2former import load_official_model

    model, official = load_official_model(cfg, official_checkpoint)
    decoder_before = module_state_sha256(model.sem_seg_head)
    backbone, dino = build_dinov3_spatial_sfp(model_dir, adapter_seed=adapter_seed)
    model.backbone = backbone
    decoder_after = module_state_sha256(model.sem_seg_head)
    if decoder_before != decoder_after:
        raise ValueError("Replacing Swin changed the retained Mask2Former head")
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
