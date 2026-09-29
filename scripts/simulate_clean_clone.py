#!/usr/bin/env python3
"""Create a temporary commit and clone to verify the repository without local data."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def git_candidates(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [Path(raw.decode("utf-8")) for raw in result.stdout.split(b"\0") if raw]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    candidates = git_candidates(root)
    with tempfile.TemporaryDirectory(prefix="icshm-clone-check-") as temp:
        temp_root = Path(temp)
        seed = temp_root / "seed"
        clone = temp_root / "clone"
        seed.mkdir()

        for relative in candidates:
            source = root / relative
            if not source.is_file():
                continue
            destination = seed / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        run(["git", "init", "-b", "main"], seed)
        run(["git", "config", "user.name", "Clone Check"], seed)
        run(["git", "config", "user.email", "clone-check@example.invalid"], seed)
        run(["git", "add", "."], seed)
        run(["git", "commit", "-m", "test: temporary clone validation"], seed)
        run(["git", "clone", "--quiet", str(seed), str(clone)], temp_root)
        run(["bash", "scripts/bootstrap_ubuntu.sh", "--check-only"], clone)
        run(["bash", "scripts/repo_health.sh"], clone)

    print("Clean clone simulation passed without competition data.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"clone simulation failed: {exc}", file=sys.stderr)
        raise SystemExit(exc.returncode)

