#!/usr/bin/env python3
"""Create a version-controlled experiment record without external services."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import date
from pathlib import Path


ID_PATTERN = re.compile(r"^[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True, dest="experiment_id")
    parser.add_argument("--title", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--owner", default="unassigned")
    parser.add_argument("--parent", default=None)
    parser.add_argument("--backbone", default="not_set")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (primarily for tests).",
    )
    return parser.parse_args()


def replace_tokens(text: str, values: dict[str, str]) -> str:
    for token, value in values.items():
        text = text.replace(token, value)
    return text


def create_experiment(
    root: Path,
    experiment_id: str,
    title: str,
    phase: str,
    owner: str = "unassigned",
    parent: str | None = None,
    backbone: str = "not_set",
) -> Path:
    root = root.resolve()
    if not ID_PATTERN.fullmatch(experiment_id):
        raise ValueError(
            "Experiment ID must use uppercase letters/numbers and hyphens, "
            "for example RGB-001."
        )

    registry_path = root / "experiments" / "registry.yaml"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = registry.get("experiments", [])
    if any(entry.get("experiment_id") == experiment_id for entry in entries):
        raise ValueError(f"Experiment ID already exists: {experiment_id}")
    if parent and not any(entry.get("experiment_id") == parent for entry in entries):
        raise ValueError(f"Parent experiment is not registered: {parent}")

    run_dir = root / "experiments" / "runs" / experiment_id
    if run_dir.exists():
        raise ValueError(f"Experiment directory already exists: {run_dir}")

    template_dir = root / "experiments" / "templates"
    values = {
        "__EXPERIMENT_ID__": experiment_id,
        "__TITLE__": title,
        "__PHASE__": phase,
        "__OWNER__": owner,
        "__PARENT__": parent or "none",
        "__BACKBONE__": backbone,
    }

    run_dir.mkdir(parents=True)
    try:
        for name in ("config.yaml", "report.md"):
            source = template_dir / name
            rendered = replace_tokens(source.read_text(encoding="utf-8"), values)
            (run_dir / name).write_text(rendered, encoding="utf-8")

        entries.append(
            {
                "experiment_id": experiment_id,
                "title": title,
                "phase": phase,
                "status": "planned",
                "owner": owner,
                "git_commit": "not_available",
                "parent_experiment": parent,
                "dataset_manifest": "data/manifests/contest_dataset_v1.json",
                "split_id": "not_set",
                "config": f"experiments/runs/{experiment_id}/config.yaml",
                "seed": "not_set",
                "backbone": backbone,
                "GPU": "not_set",
                "start_time": None,
                "end_time": None,
                "output_path": f"experiments/runs/{experiment_id}/output",
                "metrics_summary": {},
                "report_path": f"experiments/runs/{experiment_id}/report.md",
                "conclusion": "not_run",
                "keep_or_drop": "undecided",
                "reason": "Experiment created; no result yet.",
                "created_date": date.today().isoformat(),
            }
        )
        registry_path.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise

    return run_dir


def main() -> int:
    args = parse_args()
    try:
        run_dir = create_experiment(
            root=args.root,
            experiment_id=args.experiment_id,
            title=args.title,
            phase=args.phase,
            owner=args.owner,
            parent=args.parent,
            backbone=args.backbone,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Created {run_dir}")
    print("Next: complete config.yaml before starting the experiment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
