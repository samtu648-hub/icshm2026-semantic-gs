"""Feature Splatting components for a frozen Standard 3DGS model.

The upstream Standard renderer clamps override-color output to ``[0, 1]``.
Signed latent channels are therefore mapped into the open unit interval and
restored with a separately rendered alpha mass. Every pass still uses the
same native projection, sorting, opacity, and front-to-back transmittance.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib


def crop_camera(camera, *, top: int, left: int, size: int):
    """Return the exact pinhole camera for an axis-aligned image crop."""
    import numpy as np

    if camera.model != "PINHOLE":
        raise ValueError("Feature Splatting requires a PINHOLE camera")
    if (
        not all(isinstance(value, int) for value in (top, left, size))
        or min(top, left) < 0
        or size <= 0
        or top + size > camera.height
        or left + size > camera.width
    ):
        raise ValueError("Invalid camera crop")
    matrix = np.array(camera.matrix, dtype=np.float64, copy=True)
    matrix[0, 2] -= left
    matrix[1, 2] -= top
    return replace(camera, width=size, height=size, matrix=matrix)


def crop_and_flip(array, sample):
    """Apply a frozen sampling-plan row to a NumPy image or label."""
    import numpy as np

    top, left = int(sample["top"]), int(sample["left"])
    size = int(sample["crop"])
    if (
        array.ndim not in (2, 3)
        or array.shape[0] < top + size
        or array.shape[1] < left + size
    ):
        raise ValueError("Sampling crop lies outside the source array")
    output = array[top : top + size, left : left + size]
    if bool(sample["flip"]):
        output = output[:, ::-1]
    return np.ascontiguousarray(output)


def signed_feature_affine(values, *, margin: float = 0.5):
    """Map N x C signed values into (0, 1) without blocking their gradients."""
    torch = importlib.import_module("torch")

    if values.ndim != 2 or values.shape[1] < 1 or not torch.isfinite(values).all():
        raise ValueError("Expected finite N x C feature values")
    if not isinstance(margin, (int, float)) or margin <= 0:
        raise ValueError("A positive affine margin is required")
    with torch.no_grad():
        lower = values.amin(dim=0) - float(margin)
        scale = values.amax(dim=0) - values.amin(dim=0) + 2.0 * float(margin)
    normalized = (values - lower) / scale
    return normalized, lower, scale


def restore_signed_composite(normalized_image, alpha, lower, scale):
    """Undo the affine map after alpha compositing over a black background."""
    if normalized_image.ndim != 3 or alpha.shape != (1, *normalized_image.shape[1:]):
        raise ValueError("Invalid normalized feature image or alpha mass")
    if lower.shape != scale.shape or lower.numel() != normalized_image.shape[0]:
        raise ValueError("Affine metadata differs from rendered channels")
    return normalized_image * scale[:, None, None] + alpha * lower[:, None, None]


def render_signed_features(native_render, view, model, pipe, values, background):
    """Render arbitrary signed channels through repeated native RGB passes."""
    torch = importlib.import_module("torch")
    functional = importlib.import_module("torch.nn.functional")

    if values.ndim != 2 or values.shape[0] != model.get_xyz.shape[0]:
        raise ValueError("Feature field does not match frozen Gaussian topology")
    if values.device != model.get_xyz.device or values.dtype != model.get_xyz.dtype:
        raise ValueError("Feature field device/dtype differs from frozen geometry")
    if background.shape != (3,) or torch.count_nonzero(background).item() != 0:
        raise ValueError("Signed feature rendering requires a black background")
    normalized, lower, scale = signed_feature_affine(values)
    with torch.no_grad():
        unit = torch.ones(
            (values.shape[0], 3), device=values.device, dtype=values.dtype
        )
        alpha = native_render(view, model, pipe, background, override_color=unit)[
            "render"
        ][:1].detach()
    images = []
    for start in range(0, values.shape[1], 3):
        channels = normalized[:, start : start + 3]
        count = channels.shape[1]
        colors = functional.pad(channels, (0, 3 - count)).contiguous()
        image = native_render(view, model, pipe, background, override_color=colors)[
            "render"
        ][:count]
        images.append(
            restore_signed_composite(
                image,
                alpha,
                lower[start : start + count],
                scale[start : start + count],
            )
        )
    output = torch.cat(images, dim=0)
    if not torch.isfinite(output).all():
        raise FloatingPointError("Non-finite rendered latent feature")
    return output, alpha


def build_feature_splatting_field(
    point_count: int,
    *,
    latent_dim: int = 32,
    hidden_dim: int = 256,
    class_count: int = 5,
    teacher_dim: int = 768,
    device="cuda",
    dtype=None,
):
    """Build the official full Feature Splatting decoder plus a class head."""
    torch = importlib.import_module("torch")
    nn = torch.nn

    if min(point_count, latent_dim, hidden_dim, class_count, teacher_dim) <= 0:
        raise ValueError("Feature Splatting dimensions must be positive")
    dtype = dtype or torch.float32

    class FeatureSplattingField(nn.Module):
        def __init__(self):
            super().__init__()
            self.latent = nn.Parameter(
                torch.zeros((point_count, latent_dim), device=device, dtype=dtype)
            )
            self.trunk = nn.utils.weight_norm(
                nn.Conv2d(latent_dim, hidden_dim, kernel_size=1)
            )
            self.class_head = nn.utils.weight_norm(
                nn.Conv2d(hidden_dim, class_count, kernel_size=1)
            )
            self.teacher_head = nn.utils.weight_norm(
                nn.Conv2d(hidden_dim, teacher_dim, kernel_size=1)
            )
            self.to(device=device, dtype=dtype)

        def encode(self, rendered_latent):
            if rendered_latent.ndim == 3:
                rendered_latent = rendered_latent[None]
            if rendered_latent.ndim != 4 or rendered_latent.shape[1] != latent_dim:
                raise ValueError("Rendered latent must be CHW or BCHW")
            return torch.relu(self.trunk(rendered_latent))

        def classify(self, rendered_latent):
            return self.class_head(self.encode(rendered_latent))

        def project_teacher(self, rendered_latent):
            return self.teacher_head(self.encode(rendered_latent))

        def decode(self, rendered_latent):
            hidden = self.encode(rendered_latent)
            return self.class_head(hidden), self.teacher_head(hidden)

    return FeatureSplattingField()


def field_state_sha256(field) -> str:
    """Hash names, dtypes, shapes, and bytes of a feature-field state."""
    torch = importlib.import_module("torch")

    digest = hashlib.sha256()
    for name, value in sorted(field.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def final_patch_tokens(teacher, rgb):
    """Return normalized final DINOv3 patch tokens as B x 768 x H/16 x W/16."""
    torch = importlib.import_module("torch")

    if rgb.ndim != 4 or rgb.shape[1] != 3 or rgb.shape[-2] % 16 or rgb.shape[-1] % 16:
        raise ValueError("DINOv3 input must be BCHW RGB divisible by patch size 16")
    mean = torch.tensor((0.485, 0.456, 0.406), device=rgb.device, dtype=rgb.dtype)[
        None, :, None, None
    ]
    std = torch.tensor((0.229, 0.224, 0.225), device=rgb.device, dtype=rgb.dtype)[
        None, :, None, None
    ]
    grid = (rgb.shape[-2] // 16, rgb.shape[-1] // 16)
    with torch.no_grad():
        output = teacher(pixel_values=(rgb - mean) / std, return_dict=True)
        tokens = output.last_hidden_state.detach()
    count = grid[0] * grid[1]
    if tokens.ndim != 3 or tokens.shape[1] != count + 5 or tokens.shape[2] != 768:
        raise ValueError("Unexpected DINOv3 CLS/register/patch token layout")
    return tokens[:, -count:].transpose(1, 2).reshape(tokens.shape[0], 768, *grid)


def patch_cosine_loss(student, teacher):
    """Feature Splatting cosine distance averaged over all patch locations."""
    torch = importlib.import_module("torch")
    functional = importlib.import_module("torch.nn.functional")

    if student.shape != teacher.shape or student.ndim != 4:
        raise ValueError(
            "Student and teacher patch features must have the same BCHW shape"
        )
    loss = (1.0 - functional.cosine_similarity(student, teacher, dim=1)).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite DINO feature loss")
    return loss


__all__ = [
    "build_feature_splatting_field",
    "crop_and_flip",
    "crop_camera",
    "field_state_sha256",
    "final_patch_tokens",
    "patch_cosine_loss",
    "render_signed_features",
    "restore_signed_composite",
    "signed_feature_affine",
]
