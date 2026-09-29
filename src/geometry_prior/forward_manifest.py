"""Manifest and dependency checks shared by optional geometry-prior runners."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from .windowing import sliding_windows


def make_window_manifest(
    image_names: list[str],
    output: Path,
    *,
    model: str,
    window_size: int,
    overlap: int,
    segments: dict | None = None,
    max_windows: int | None = None,
) -> dict:
    windows = []
    if segments:
        for segment in segments["segments"]:
            segment_windows = sliding_windows(segment["images"], window_size, overlap)
            for index, names in enumerate(segment_windows):
                windows.append({"window_id": f'{segment["segment_id"]}-w{index:03d}', "segment_id": segment["segment_id"], "images": names})
    else:
        windows = [{"window_id": f"w{index:03d}", "images": names} for index, names in enumerate(sliding_windows(image_names, window_size, overlap))]
    if max_windows is not None:
        windows = windows[:max_windows]
    payload = {
        "schema_version": 1,
        "model": model,
        "image_count": len(image_names),
        "window_size": window_size,
        "overlap": overlap,
        "window_count": len(windows),
        "windows": windows,
        "output_contract": "per-window NPZ depth/confidence/world_points plus manifest",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def require_optional_package(package: str, source_root: Path | None, checkpoint: Path | None) -> None:
    if source_root is not None and not source_root.exists():
        raise RuntimeError(f"Configured source_root does not exist: {source_root}")
    if checkpoint is not None and not checkpoint.exists():
        raise RuntimeError(f"Configured checkpoint does not exist: {checkpoint}")
    if importlib.util.find_spec(package) is None:
        raise RuntimeError(
            f"Optional geometry model package '{package}' is not installed. "
            "Install only in the isolated geometry environment after source/commit/license approval."
        )
