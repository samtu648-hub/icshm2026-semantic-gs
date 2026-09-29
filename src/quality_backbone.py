"""New-backbone contracts; legacy renderers and frozen sidecars remain unchanged.

Concrete upstream implementations must satisfy this interface. Defining a
Protocol is NOT a claim that PGSR/SOF feature rendering has been implemented.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import re
from typing import Protocol

import numpy as np

from src.camera_adapter import CameraFromWorld, Sim3, SimpleRadialCamera, official_to_model_camera, redistort_rgb, redistort_semantic_probabilities
from src.quality_protocol import CLASSES


@dataclass(frozen=True)
class CameraRequest:
    width: int
    height: int
    matrix: np.ndarray
    pose: CameraFromWorld
    model: str = "PINHOLE"
    radial_k: float = 0.0

    def __post_init__(self):
        k = np.asarray(self.matrix)
        if not isinstance(self.width, int) or not isinstance(self.height, int) or min(self.width, self.height) <= 0:
            raise ValueError("Positive integer image dimensions required")
        if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
            raise ValueError("Invalid camera intrinsics")
        if not np.allclose(k[2], [0, 0, 1]) or abs(k[0, 1]) > 1e-8 or abs(k[1, 0]) > 1e-8:
            raise ValueError("COLMAP camera must have zero skew")
        if self.model not in ("PINHOLE", "SIMPLE_RADIAL") or not np.isfinite(self.radial_k):
            raise ValueError("Unsupported camera model")
        if self.model == "SIMPLE_RADIAL" and not np.isclose(k[0, 0], k[1, 1]):
            raise ValueError("SIMPLE_RADIAL requires one focal length")
        if self.model == "PINHOLE" and self.radial_k != 0:
            raise ValueError("PINHOLE cannot have radial distortion")
        r, t = self.pose.rotation, self.pose.translation
        if not np.isfinite(r).all() or not np.isfinite(t).all() or not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1, atol=1e-6):
            raise ValueError("Invalid official CAMERA_FROM_WORLD pose")


@dataclass(frozen=True)
class ModelIdentity:
    backbone: str
    checkpoint_sha256: str
    topology_sha256: str

    def __post_init__(self):
        if self.backbone not in ("scaffold-gs", "pgsr", "sof"):
            raise ValueError("Unsupported reconstruction backend")
        for value in (self.checkpoint_sha256, self.topology_sha256):
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("Full content hashes required")


def validate_attachment(metadata, identity: ModelIdentity):
    expected = {"schema_version": 1, "backbone": identity.backbone,
                "checkpoint_sha256": identity.checkpoint_sha256,
                "topology_sha256": identity.topology_sha256, "classes": list(CLASSES),
                "geometry_frozen": True, "pixel_space": "PINHOLE"}
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError("Semantic attachment does not match frozen backend contract")


def array_signature(named_arrays):
    """Hash names, dtypes, shapes AND bytes; callers must include all RGB tensors."""
    digest = hashlib.sha256()
    if not named_arrays:
        raise ValueError("Empty tensor fingerprint")
    for name, value in sorted(named_arrays.items()):
        a = np.asarray(value)
        if a.dtype.hasobject:
            raise ValueError(f"Object model tensor: {name}")
        if not np.isfinite(a).all():
            raise ValueError(f"Non-finite model tensor {name}: {int((~np.isfinite(a)).sum())}/{a.size} elements")
        digest.update(f"{name}\0{a.dtype.str}\0{a.shape}\0".encode())
        digest.update(np.ascontiguousarray(a).tobytes())
    return digest.hexdigest()


def validate_probabilities(probabilities, height, width):
    p = np.asarray(probabilities)
    if p.shape != (5, height, width) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("Invalid native five-class probabilities")
    if not np.allclose(p.sum(axis=0), 1, atol=1e-5, rtol=0):
        raise ValueError("Semantic probabilities must sum to one")
    return p


@dataclass
class NativeRender:
    rgb: np.ndarray
    probabilities: np.ndarray | None = None
    features: np.ndarray | None = None
    depth: np.ndarray | None = None
    normals: np.ndarray | None = None
    visibility: np.ndarray | None = None

    def validate(self, camera):
        h, w = camera.height, camera.width
        if self.rgb.shape != (h, w, 3) or not np.isfinite(self.rgb).all() or self.rgb.min() < 0 or self.rgb.max() > 1:
            raise ValueError("RGB must be native HxWx3 in [0,1]")
        if self.probabilities is not None:
            validate_probabilities(self.probabilities, h, w)
        for key, prefix in (("features", None), ("depth", ()), ("normals", (3,)), ("visibility", ())):
            x = getattr(self, key)
            if x is None:
                continue
            shape_ok = x.ndim == 3 and x.shape[0] > 0 and x.shape[1:] == (h, w) if prefix is None else x.shape == prefix + (h, w)
            if not shape_ok or not np.isfinite(x).all():
                raise ValueError(f"Invalid native {key}")


class BackboneRenderer(Protocol):
    identity: ModelIdentity
    official_to_model: Sim3

    def render_pinhole(self, camera: CameraRequest) -> NativeRender:
        """Render metadata-only input using the backbone's native compositor."""
        ...


def render_official(renderer: BackboneRenderer, camera: CameraRequest):
    model_pose = official_to_model_camera(camera.pose, renderer.official_to_model)
    native_camera = replace(camera, pose=model_pose, model="PINHOLE", radial_k=0.0)
    native = renderer.render_pinhole(native_camera)
    native.validate(native_camera)
    rgb = native.rgb.copy()
    probabilities = None if native.probabilities is None else native.probabilities.copy()
    if camera.model == "SIMPLE_RADIAL":
        k = camera.matrix
        radial = SimpleRadialCamera(camera.width, camera.height, float(k[0, 0]), float(k[0, 2]), float(k[1, 2]), camera.radial_k)
        rgb = redistort_rgb(rgb, radial)
        if probabilities is not None:
            probabilities = redistort_semantic_probabilities(probabilities, radial)
            # Out-of-frame remapping has zero support; explicitly abstain to background.
            unsupported = probabilities.sum(axis=0) == 0
            probabilities[0, unsupported] = 1
            validate_probabilities(probabilities, camera.height, camera.width)
    return {"rgb": np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8),
            "semantic": None if probabilities is None else probabilities.argmax(axis=0).astype(np.uint8),
            "probabilities": probabilities, "native_diagnostics": native,
            "diagnostic_pixel_space": "PINHOLE", "output_pixel_space": camera.model}
