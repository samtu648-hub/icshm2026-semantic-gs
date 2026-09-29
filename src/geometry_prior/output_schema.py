"""Versioned geometry-output contract shared by GeometryPrior adapters.

The contract describes files that a real optional model may emit.  It contains
no model import, checkpoint access, CUDA call, or filesystem side effect.
"""

from __future__ import annotations

from copy import deepcopy


GEOMETRY_OUTPUT_SCHEMA_V2 = {
    "schema_id": "geometry_prior_npz_v2",
    "coordinate_frame": "official_colmap_reference_or_explicit_sim3_diagnostic",
    "required_arrays": {
        "depth": "float32[height,width], camera_plus_z",
        "confidence": "float32[height,width]",
        "valid_mask": "uint8[height,width]",
    },
    "optional_arrays": {
        "world_points": "float32[height,width,3]",
        "pred_cam2world": "float64[4,4]",
        "normal": "float32[height,width,3]",
        "sky_or_reflection_mask": "uint8[height,width]",
    },
    "required_metadata": (
        "source_model",
        "source_commit",
        "checkpoint_sha256",
        "image_name",
        "official_camera_id",
        "input_resolution",
        "coordinate_frame",
        "depth_scale",
        "window_manifest_id",
    ),
}


def geometry_output_schema_v2() -> dict:
    """Return a caller-owned copy of the pre-registered output contract."""
    return deepcopy(GEOMETRY_OUTPUT_SCHEMA_V2)


def validate_geometry_manifest_v2(manifest: dict) -> None:
    """Reject a manifest that cannot describe v2 fusion-compatible outputs."""
    if manifest.get("schema_version") != 2:
        raise ValueError("GeometryPrior v2 manifest must use schema_version=2")
    schema = manifest.get("geometry_output_schema")
    if not isinstance(schema, dict) or schema.get("schema_id") != "geometry_prior_npz_v2":
        raise ValueError("GeometryPrior v2 manifest is missing geometry_prior_npz_v2")
    windows = manifest.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("GeometryPrior v2 manifest must contain at least one window")
    for window in windows:
        images = window.get("images")
        if not isinstance(images, list) or len(images) < 2 or len(images) != len(set(images)):
            raise ValueError("Each geometry window must have at least two unique images")
        if not window.get("window_manifest_id"):
            raise ValueError("Each geometry window must have a stable window_manifest_id")
