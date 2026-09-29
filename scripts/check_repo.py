#!/usr/bin/env python3
"""Check repository upload safety and required research files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


MAX_BYTES = 50 * 1024 * 1024
REQUIRED = {
    "README.md",
    "AGENTS.md",
    "PROJECT_STATUS.md",
    "ROADMAP.md",
    "TASKS.md",
    "DECISIONS.md",
    "RISKS.md",
    "METRICS.md",
    "experiments/registry.yaml",
    "data/README.md",
}
FORBIDDEN_SUFFIXES = {".ckpt", ".pth", ".pt", ".ply", ".safetensors"}
FORBIDDEN_PREFIXES = (
    "data/Contest Dataset",
    "data/raw/",
    "data/processed/",
    "checkpoints/",
    "wandb/",
)


def git_candidates(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [root / raw.decode("utf-8") for raw in result.stdout.split(b"\0") if raw]


def ignored(root: Path, relative: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", relative],
        cwd=root,
    )
    return result.returncode == 0


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []
    for relative in sorted(REQUIRED):
        if not (root / relative).is_file():
            failures.append(f"missing required file: {relative}")

    for path in git_candidates(root):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.stat().st_size > MAX_BYTES:
            failures.append(f"candidate file exceeds 50MB: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden model/geometry artifact: {relative}")
        if any(relative.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            failures.append(f"forbidden runtime/data path is commit candidate: {relative}")
        if relative.startswith("outputs/") and relative != "outputs/README.md":
            failures.append(f"forbidden runtime output is commit candidate: {relative}")

    ignored_samples = {
        "data/raw/example.png",
        "data/Contest Dataset/images/001.png",
        "outputs/example.png",
        "checkpoints/model.ckpt",
        "wandb/run/file",
        ".env",
    }
    for sample in sorted(ignored_samples):
        if not ignored(root, sample):
            failures.append(f"ignore rule missing for sample path: {sample}")

    if failures:
        print("Repository safety check failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print("Repository upload-safety check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
