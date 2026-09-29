#!/usr/bin/env python3
"""Verify the exact shared Standard source binding before either GEO-027 arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(*args: str, cwd: Path) -> str:
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected = config["source_binding"]
    observed = {
        "working_tree_diff_sha256": hashlib.sha256(
            subprocess.run(
                ["git", "diff", "--binary"],
                cwd=args.source,
                check=True,
                stdout=subprocess.PIPE,
            ).stdout
        ).hexdigest(),
        "dataset_readers_sha256": sha256(args.source / "scene/dataset_readers.py"),
        "train_py_sha256": sha256(args.source / "train.py"),
        "render_py_sha256": sha256(args.source / "render.py"),
        "diff_gaussian_rasterization_commit": command(
            "git", "rev-parse", "HEAD", cwd=args.source / "submodules/diff-gaussian-rasterization"
        ),
        "simple_knn_commit": command(
            "git", "rev-parse", "HEAD", cwd=args.source / "submodules/simple-knn"
        ),
    }
    commit = command("git", "rev-parse", "HEAD", cwd=args.source)
    if commit != config["upstream_commit"] or observed != expected:
        raise ValueError(
            f"Standard source binding mismatch: commit={commit} observed={observed}"
        )
    repo_root = (args.repo_root or args.config.resolve().parents[3]).resolve()
    runner_observed = {
        relative: sha256(repo_root / relative)
        for relative in config.get("runner_binding", {})
    }
    if runner_observed != config.get("runner_binding", {}):
        raise ValueError(f"GEO-027 runner binding mismatch: {runner_observed}")
    print(
        json.dumps(
            {
                "status": "passed",
                "commit": commit,
                "source_binding": observed,
                "runner_binding": runner_observed,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
