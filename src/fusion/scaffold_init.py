"""Build a new external Scaffold scene that differs only in initialization PLY."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _link_read_only(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing destination {destination}")
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def build_scaffold_init_scene(source_scene: Path, init_ply: Path, destination: Path) -> dict:
    """Create an external, non-mutating scene with only points3D.ply replaced."""
    source_sparse = source_scene / "sparse" / "0"
    required = [source_scene / "images", source_sparse / "cameras.txt", source_sparse / "images.txt", init_ply]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source scene inputs: " + ", ".join(missing))
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing scene destination {destination}")
    sparse = destination / "sparse" / "0"
    sparse.mkdir(parents=True)
    _link_read_only(source_scene / "images", destination / "images")
    for name in ("cameras.txt", "images.txt", "test.txt", "points3D.txt"):
        source = source_sparse / name
        if source.exists():
            _link_read_only(source, sparse / name)
    shutil.copy2(init_ply, sparse / "points3D.ply")
    manifest = {
        "schema_version": 1,
        "source_scene_layout": "read_only_symlinks",
        "initialization_change": "sparse/0/points3D.ply only",
        "source_images_sha256": "directory_symlink_not_hashed",
        "candidate_init_ply_sha256": sha256_file(init_ply),
        "scene_init_ply_sha256": sha256_file(sparse / "points3D.ply"),
        "coordinate_frame": "official_colmap_reference",
    }
    (destination / "geometry_init_scene_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
