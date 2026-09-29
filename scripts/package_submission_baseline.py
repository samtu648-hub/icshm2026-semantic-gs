#!/usr/bin/env python3
"""Build a Git-ignored, relocatable RGB-030-FULL + SEM-001 baseline bundle.

The bundle contains the frozen model artifacts, the exact lightweight project
renderer, a source snapshot of the pinned Scaffold-GS checkout, its license and
patch provenance, and a content-addressed manifest.  It deliberately does not
contain released images, labels, test cameras, logs, predictions or secrets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = (
    "scripts/render_submission.py",
    "scripts/render_external_colmap_camera.py",
    "src/camera_adapter.py",
    "src/scaffold_semantic_sidecar.py",
    "configs/submission/sem001_baseline_v1.json",
)
SOURCE_IGNORE_NAMES = {
    ".git",
    "backup",
    "build",
    "__pycache__",
    ".cache",
    "wandb",
    "outputs",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git(source_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source_root), *args], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).stdout.strip()


def _copy_source_snapshot(source_root: Path, destination: Path) -> None:
    def ignore(_directory: str, names: list[str]) -> set[str]:
        ignored = {name for name in names if name in SOURCE_IGNORE_NAMES or name.endswith(".egg-info")}
        ignored.update(name for name in names if name.endswith((".pyc", ".log")))
        return ignored

    shutil.copytree(source_root, destination, ignore=ignore, symlinks=False)


def _source_file(entry: dict[str, str], rgb_run_dir: Path, semantic_checkpoint: Path) -> Path:
    kind, relative = entry["source"].split(":", 1)
    if kind == "rgb_run_dir":
        return rgb_run_dir / relative
    if kind == "semantic_checkpoint":
        if Path(relative).name != semantic_checkpoint.name:
            raise ValueError("Semantic artifact name in lock does not match supplied checkpoint")
        return semantic_checkpoint
    raise ValueError(f"Unsupported artifact source kind: {kind}")


def _content_manifest(bundle: Path) -> dict[str, Any]:
    entries = []
    manifest_path = bundle / "metadata" / "artifact_manifest.json"
    for path in sorted(item for item in bundle.rglob("*") if item.is_file() and item != manifest_path):
        entries.append({"path": path.relative_to(bundle).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return {"schema_version": 1, "file_count": len(entries), "files": entries}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=REPO_ROOT / "configs/submission/sem001_baseline_v1.json")
    parser.add_argument("--semantic-config", type=Path, default=REPO_ROOT / "experiments/runs/SEM-001/config.yaml")
    parser.add_argument("--rgb-run-dir", type=Path, required=True)
    parser.add_argument("--semantic-checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    semantic_config = json.loads(args.semantic_config.read_text(encoding="utf-8"))
    if semantic_config.get("experiment_id") != lock["semantic_experiment"]:
        raise ValueError("Semantic config does not match the frozen lock")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty bundle directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    actual_commit = _git(args.source_root, "rev-parse", "HEAD")
    if actual_commit != lock["scaffold"]["upstream_commit"]:
        raise ValueError(f"Scaffold commit mismatch: {actual_commit}")
    patch = REPO_ROOT / lock["scaffold"]["patch"]
    if sha256(patch) != lock["scaffold"]["patch_sha256"]:
        raise ValueError("Scaffold patch hash mismatch")
    subprocess.run(
        ["git", "-C", str(args.source_root), "apply", "--reverse", "--check", str(patch)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    copied_artifacts = []
    for entry in lock["artifacts"]:
        source = _source_file(entry, args.rgb_run_dir, args.semantic_checkpoint)
        if not source.is_file():
            raise FileNotFoundError(source)
        actual_hash = sha256(source)
        if actual_hash != entry["sha256"]:
            raise ValueError(f"Frozen artifact hash mismatch for {source.name}: {actual_hash}")
        destination = args.output_dir / entry["bundle_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied_artifacts.append({**entry, "bytes": destination.stat().st_size})

    for relative in CODE_FILES:
        source = REPO_ROOT / relative
        destination = args.output_dir / "code" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (args.output_dir / "metadata").mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.semantic_config, args.output_dir / "metadata" / "sem001_config.json")
    shutil.copy2(patch, args.output_dir / "metadata" / "scaffold-gs-r1.patch")
    _copy_source_snapshot(args.source_root, args.output_dir / "third_party" / "scaffold-gs")

    provenance = {
        "schema_version": 1,
        "submission_id": lock["submission_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rgb_experiment": lock["rgb_experiment"],
        "semantic_experiment": lock["semantic_experiment"],
        "scaffold_upstream_url": lock["scaffold"]["upstream_url"],
        "scaffold_upstream_commit": actual_commit,
        "scaffold_patch_sha256": lock["scaffold"]["patch_sha256"],
        "source_snapshot_excludes": sorted(SOURCE_IGNORE_NAMES | {"*.egg-info", "*.pyc", "*.log"}),
        "artifacts": copied_artifacts,
        "contains_dataset_or_predictions": False,
        "official_protocol_status": "final command, filenames and metric details not yet published",
    }
    write_json(args.output_dir / "metadata" / "provenance.json", provenance)
    readme = """# Frozen Scaffold-GS + SEM-001 fallback bundle

This is the current reproducible fallback, not a claim about the organizers' still-unpublished final filename protocol.

Render one or more externally supplied COLMAP poses with the frozen environment:

```bash
python code/scripts/render_submission.py --bundle . \\
  --cameras /path/to/cameras.txt --images /path/to/images.txt \\
  --output-dir /path/to/output --write-logits
```

The input pose convention is `CAMERA_FROM_WORLD`. The renderer never reads reference RGB or labels. RGB is written under `rgb/`, semantic class-ID PNGs under `semantic/`, and optional five-channel logits under `logits/`. Class ID 2 is `stay_cable`.

Run `python code/scripts/audit_submission_bundle.py --bundle . --renders /path/to/output` after rendering. The vendored Scaffold-GS snapshot retains its research-only license; redistribution and use must follow `third_party/scaffold-gs/LICENSE.md`.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")

    # The auditor is copied last because it is also part of the immutable manifest.
    auditor_source = REPO_ROOT / "scripts/audit_submission_bundle.py"
    auditor_destination = args.output_dir / "code" / "scripts" / auditor_source.name
    shutil.copy2(auditor_source, auditor_destination)
    manifest = _content_manifest(args.output_dir)
    write_json(args.output_dir / "metadata" / "artifact_manifest.json", manifest)
    print(json.dumps({"bundle": str(args.output_dir), "files": manifest["file_count"], "status": "created"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
