"""Frozen-RGB Scaffold-GS semantic sidecar used by SEM-001.

The sidecar owns only one trainable tensor: five logits per saved Scaffold-GS
anchor.  Generated Gaussians inherit their anchor logits.  Scaffold geometry,
opacity, RGB color MLP, covariance MLP and appearance embedding are loaded from
the frozen RGB-030-FULL artifact and used only under ``torch.no_grad``.
"""

from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

CLASS_COUNT = 5


@dataclass(frozen=True)
class ScaffoldSettings:
    feat_dim: int = 32
    n_offsets: int = 10
    voxel_size: float = 0.001
    update_depth: int = 3
    update_init_factor: int = 16
    update_hierachy_factor: int = 4
    use_feat_bank: bool = False
    appearance_dim: int = 32
    ratio: int = 1
    add_opacity_dist: bool = False
    add_cov_dist: bool = False
    add_color_dist: bool = False
    train_camera_count: int = 303


def configure_scaffold_import(source_root: Path) -> None:
    root = source_root.resolve()
    # Match the pinned R1 launch environment.  Both CUDA extensions are built
    # in these upstream submodule directories; importing only the repository
    # root works on an editable development shell but not in a clean runner.
    for path in (root / "submodules" / "diff-gaussian-rasterization", root / "submodules" / "simple-knn", root):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def pipeline() -> SimpleNamespace:
    return SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False,
    )


def load_frozen_scaffold(source_root: Path, point_cloud: Path, settings: ScaffoldSettings):
    """Load the RGB artifact and disable gradients on every RGB parameter."""
    import torch

    configure_scaffold_import(source_root)
    from scene.gaussian_model import GaussianModel

    model = GaussianModel(
        feat_dim=settings.feat_dim,
        n_offsets=settings.n_offsets,
        voxel_size=settings.voxel_size,
        update_depth=settings.update_depth,
        update_init_factor=settings.update_init_factor,
        update_hierachy_factor=settings.update_hierachy_factor,
        use_feat_bank=settings.use_feat_bank,
        appearance_dim=settings.appearance_dim,
        ratio=settings.ratio,
        add_opacity_dist=settings.add_opacity_dist,
        add_cov_dist=settings.add_cov_dist,
        add_color_dist=settings.add_color_dist,
    )
    model.set_appearance(settings.train_camera_count)
    model.load_ply_sparse_gaussian(str(point_cloud))
    model.load_mlp_checkpoints(str(point_cloud.parent))
    model.eval()
    for _name, parameter in frozen_rgb_named_parameters(model):
        parameter.requires_grad_(False)
    # GaussianModel is deliberately not an nn.Module.  Enumerate both its
    # anchor tensors and every nested (including TorchScript) RGB module.
    trainable = frozen_rgb_trainable_parameters(model)
    if trainable:
        raise RuntimeError(f"Frozen Scaffold-GS still has trainable RGB parameters: {trainable}")
    if not torch.cuda.is_available():
        raise RuntimeError("SEM-001 requires CUDA because Scaffold-GS rasterization is a CUDA extension")
    return model


def frozen_rgb_named_parameters(model: Any) -> list[tuple[str, Any]]:
    """Enumerate every persisted RGB tensor despite upstream not using nn.Module."""
    entries: list[tuple[str, Any]] = [
        ("anchor", model._anchor),
        ("offset", model._offset),
        ("anchor_feat", model._anchor_feat),
        ("opacity", model._opacity),
        ("scaling", model._scaling),
        ("rotation", model._rotation),
    ]
    module_names = ["mlp_opacity", "mlp_cov", "mlp_color"]
    if model.appearance_dim > 0:
        module_names.append("embedding_appearance")
    if model.use_feat_bank:
        module_names.append("mlp_feature_bank")
    for prefix in module_names:
        module = getattr(model, prefix)
        entries.extend((f"{prefix}.{name}", parameter) for name, parameter in module.named_parameters())
    return entries


def frozen_rgb_trainable_parameters(model: Any) -> list[str]:
    return [name for name, parameter in frozen_rgb_named_parameters(model) if parameter.requires_grad]


class AnchorSemanticSidecar:  # thin composition rather than a GaussianModel mutation
    def __init__(self, anchor_count: int, *, initial_background_logit: float = 0.0):
        import torch

        self.logits = torch.nn.Parameter(torch.zeros((anchor_count, CLASS_COUNT), dtype=torch.float32, device="cuda"))
        if initial_background_logit:
            with torch.no_grad():
                self.logits[:, 0] = initial_background_logit

    def state_dict(self) -> dict[str, Any]:
        return {"semantic_anchor_logits": self.logits.detach().cpu()}

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        import torch

        stored = payload.get("semantic_anchor_logits")
        if not isinstance(stored, torch.Tensor) or tuple(stored.shape) != tuple(self.logits.shape):
            raise ValueError("Semantic checkpoint anchor-logit shape does not match frozen RGB anchor count")
        with torch.no_grad():
            self.logits.copy_(stored.to(device="cuda", dtype=torch.float32))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_rgb_hashes(rgb_run_dir: Path, iteration: int) -> dict[str, str]:
    point_dir = rgb_run_dir / "point_cloud" / f"iteration_{iteration}"
    required = [
        rgb_run_dir / f"chkpnt{iteration}.pth",
        point_dir / "point_cloud.ply",
        point_dir / "opacity_mlp.pt",
        point_dir / "cov_mlp.pt",
        point_dir / "color_mlp.pt",
        point_dir / "embedding_appearance.pt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Frozen RGB artifact incomplete: " + ", ".join(missing))
    return {path.name: sha256(path) for path in required}


def _raster_settings(view: Any, background):
    from diff_gaussian_rasterization import GaussianRasterizationSettings

    return GaussianRasterizationSettings(
        image_height=int(view.image_height),
        image_width=int(view.image_width),
        tanfovx=math.tan(view.FoVx * 0.5),
        tanfovy=math.tan(view.FoVy * 0.5),
        bg=background,
        scale_modifier=1.0,
        viewmatrix=view.world_view_transform,
        projmatrix=view.full_proj_transform,
        sh_degree=1,
        campos=view.camera_center,
        prefiltered=False,
        debug=False,
    )


def _render_color_channels(geometry: tuple[Any, Any, Any, Any, Any], colors, view: Any, background):
    import torch
    from diff_gaussian_rasterization import GaussianRasterizer

    xyz, opacity, scaling, rotation, _ = geometry
    # No geometry gradients are required, but the renderer expects a tensor in
    # this argument.  It only carries gradients for dynamic Gaussian geometry.
    means2d = torch.zeros_like(xyz, requires_grad=False)
    rasterizer = GaussianRasterizer(raster_settings=_raster_settings(view, background))
    rendered, _ = rasterizer(
        means3D=xyz,
        means2D=means2d,
        shs=None,
        colors_precomp=colors,
        opacities=opacity,
        scales=scaling,
        rotations=rotation,
        cov3D_precomp=None,
    )
    return rendered


def frozen_generated_geometry(view: Any, model: Any, pipe: Any):
    """Generate the exact frozen Scaffold Gaussians and their anchor mask."""
    import torch
    from gaussian_renderer import generate_neural_gaussians, prefilter_voxel

    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        visible = prefilter_voxel(view, model, pipe, background)
        xyz, _rgb, opacity, scaling, rotation, _neural_opacity, active_offsets = generate_neural_gaussians(
            view, model, visible, is_training=True
        )
    return (xyz.detach(), opacity.detach(), scaling.detach(), rotation.detach(), active_offsets.detach()), visible.detach()


def render_semantic_probabilities(
    view: Any,
    model: Any,
    sidecar: AnchorSemanticSidecar,
    pipe: Any,
    *,
    anchor_postprocess_mask: Any | None = None,
):
    """Differentiably render five semantic probabilities with frozen RGB geometry.

    ``anchor_postprocess_mask`` is an inference-only SEM-070 hook.  The base
    SEM-001 path leaves it as ``None`` and is therefore unchanged.  When
    supplied, only the generated Gaussian probabilities inherited from a
    selected frozen anchor receive the registered
    ``stay_cable -> background`` mass transfer; RGB geometry, opacity and
    appearance tensors are never modified.
    """
    import torch

    geometry, visible = frozen_generated_geometry(view, model, pipe)
    active_offsets = geometry[4]
    anchor_logits = sidecar.logits[visible]
    offset_logits = anchor_logits.repeat_interleave(model.n_offsets, dim=0)[active_offsets]
    probabilities = torch.softmax(offset_logits, dim=1)
    if anchor_postprocess_mask is not None:
        from src.semantic_postprocess import apply_offset_probability_transfer

        visible_anchor_indices = torch.nonzero(visible, as_tuple=False).squeeze(1)
        offset_anchor_indices = visible_anchor_indices.repeat_interleave(model.n_offsets)[active_offsets]
        probabilities = apply_offset_probability_transfer(
            probabilities,
            offset_anchor_indices,
            anchor_postprocess_mask,
        )
    rgb_like = _render_color_channels(
        geometry, probabilities[:, :3], view, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    )
    last_two_as_rgb = torch.cat((probabilities[:, 3:5], torch.zeros_like(probabilities[:, :1])), dim=1)
    tail = _render_color_channels(
        geometry, last_two_as_rgb, view, torch.zeros(3, dtype=torch.float32, device="cuda"))
    output = torch.cat((rgb_like, tail[:2]), dim=0).clamp_min(0.0)
    return output / output.sum(dim=0, keepdim=True).clamp_min(torch.finfo(output.dtype).tiny)


def render_frozen_rgb(view: Any, model: Any, pipe: Any):
    """Use the unmodified upstream RGB renderer solely for isolation evidence."""
    import torch
    from gaussian_renderer import prefilter_voxel, render

    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        visible = prefilter_voxel(view, model, pipe, background)
        return render(view, model, pipe, background, visible_mask=visible)["render"].detach()
