#!/usr/bin/env python3
"""Materialize a lightweight accepted-run view after a passed GEO-026 audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError(f"Refusing to overwrite {args.destination}")
    result = json.loads((args.audit_dir / "result.json").read_text(encoding="utf-8"))
    accepted = json.loads(
        (args.audit_dir / "accepted_records.json").read_text(encoding="utf-8")
    )
    if (
        result.get("experiment_id") != "GEO-026-P0-F64"
        or result.get("status") != "completed_passed_for_fusion"
        or accepted.get("status") != "accepted_for_fusion"
        or len(accepted.get("records", [])) != 7575
    ):
        raise ValueError("GEO-026 audit is not a complete accepted result")
    source_progress_hash = sha256(args.source_run / "progress.json")
    if source_progress_hash != accepted.get("source_progress_sha256"):
        raise ValueError("Preserved source progress changed after GEO-026 audit")
    source_manifest = json.loads(
        (args.source_run / "window_manifest.json").read_text(encoding="utf-8")
    )
    args.destination.mkdir(parents=True)
    (args.destination / "windows").symlink_to(
        (args.source_run / "windows").resolve(), target_is_directory=True
    )
    manifest = {
        **source_manifest,
        "model_execution": "completed",
        "status": "accepted_for_operational_v2_fusion",
        "preserved_source_acceptance": "GEO-026-P0-F64",
        "records": accepted["records"],
        "numeric_audit": {
            "experiment_id": "GEO-026-P0-F64",
            "result_sha256": sha256(args.audit_dir / "result.json"),
            "accepted_records_sha256": sha256(
                args.audit_dir / "accepted_records.json"
            ),
            "source_progress_sha256": source_progress_hash,
        },
    }
    output = args.destination / "window_manifest.json"
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": manifest["status"], "manifest": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
