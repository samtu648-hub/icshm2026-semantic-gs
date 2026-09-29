#!/usr/bin/env python3
"""Repeat original AbsGS, without importing any Gaussian dropout module."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import run_rgb152_spatial as base
from src.quality_protocol import sha256, write_new_json, verify_hashes

CONFIG = ROOT / "experiments/runs/RGB-158-NATIVE-ABSGS-REPLAY/config.yaml"


def read(path):
    return json.loads(Path(path).read_text())


def merge(a, b):
    r = dict(a)
    for k, v in b.items():
        r[k] = (
            merge(r[k], v) if isinstance(v, dict) and isinstance(r.get(k), dict) else v
        )
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-root", type=Path, required=True)
    a = p.parse_args()
    while subprocess.check_output(
        [
            "systemctl",
            "--user",
            "show",
            "icshm-dropout-frozen-semantic-probe.service",
            "-p",
            "ActiveState",
            "--value",
        ],
        text=True,
    ).strip() in ("active", "activating"):
        time.sleep(15)
    c = read(CONFIG)
    c = merge(read(ROOT / c["base_config"]), c)
    root = ROOT / c["output"]["root"]
    root.mkdir(parents=True, exist_ok=False)
    base.EID, base.CONFIG = c["experiment_id"], CONFIG
    base.CODE_PATHS = tuple(base.CODE_PATHS) + (str(Path(__file__).relative_to(ROOT)),)
    paths = list(base.CODE_PATHS) + [
        str(CONFIG.relative_to(ROOT)),
        c["base_config"],
        c["split_path"],
        c["protocol"],
    ]
    binding = {f: sha256(ROOT / f) for f in paths}
    write_new_json(
        root / "binding.json",
        {"files": binding, "resolved_config": c, "dropout_module_installed": False},
    )
    result = {"status": "running", "stages": []}
    begin = time.monotonic()
    try:
        for stage, fn in [
            ("preflight", base.preflight),
            ("train", base.train),
            ("evaluate", base.render_and_evaluate),
        ]:
            verify_hashes(ROOT, binding)
            start = time.monotonic()
            fn(c, a.artifact_root.resolve())
            result["stages"].append(
                {"stage": stage, "seconds": time.monotonic() - start, "exit_status": 0}
            )
        verify_hashes(ROOT, binding)
        result["status"] = "completed"
    except BaseException as e:
        result.update(status="failed_preserved", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        result["seconds"] = time.monotonic() - begin
        write_new_json(root / "execution-result.json", result)


if __name__ == "__main__":
    main()
