#!/usr/bin/env python3
"""Reproduce isolated candidate sources/extensions without touching the base env.

Downloads official public code only. No datasets, credentials, system packages,
GPU driver changes, deletion or uploads. Every attempt retains its build logs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, write_new_json
from scripts.run_quality_rgb import source_manifest

SOURCES = {
    "pgsr": ("https://github.com/zju3dv/PGSR.git", "de24f1a38b350387e8d8fe381b2cd70c1ae946e7", "quality_pgsr_compat.patch"),
    "sof": ("https://github.com/r4dl/SOF.git", "b9eb4170c843014f5f96d54924976161bd675469", "quality_sof_compat.patch"),
    "pytorch3d": ("https://github.com/facebookresearch/pytorch3d.git", "0a7d4c1a171e8b768c63f15b17564f9ad495f49b", None),
    "fused-ssim": ("https://github.com/rahul-goel/fused-ssim.git", "a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38", None),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quality-root", type=Path, default=ROOT / "outputs/quality")
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--cuda", type=Path, required=True)
    p.add_argument("--attempt", required=True)
    p.add_argument("--build", action="store_true")
    args = p.parse_args()
    if not args.attempt.replace("-", "").isalnum():
        raise ValueError("Invalid attempt identifier")
    root = args.quality_root.resolve()
    logdir = root / "preparation" / args.attempt
    logdir.mkdir(parents=True, exist_ok=False)
    record = {"status": "preparing", "sources": {}, "build_requested": args.build}
    python = args.python.absolute()
    env = dict(os.environ, CUDA_HOME=str(args.cuda), TORCH_CUDA_ARCH_LIST="12.0", QUALITY_CUDA_ARCHS="120", CUDA_ARCHITECTURES="120", MAX_JOBS="4")
    try:
        for name, (url, commit, patch_name) in SOURCES.items():
            source = root / "source" / name
            with (logdir / f"{name}-source.log").open("x") as log:
                if not source.exists():
                    source.parent.mkdir(parents=True, exist_ok=True)
                    subprocess.run(["git", "clone", "--no-checkout", url, str(source)], stdout=log, stderr=subprocess.STDOUT, check=True)
                    subprocess.run(["git", "-C", str(source), "checkout", "--detach", commit], stdout=log, stderr=subprocess.STDOUT, check=True)
                actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
                if actual != commit:
                    raise ValueError(f"Existing {name} checkout has different commit; left untouched")
                if patch_name:
                    patch = ROOT / "third_party/patches" / patch_name
                    applied = subprocess.run(["git", "-C", str(source), "apply", "--reverse", "--check", str(patch)], stdout=log, stderr=subprocess.STDOUT).returncode == 0
                    if not applied:
                        subprocess.run(["git", "-C", str(source), "apply", "--check", str(patch)], stdout=log, stderr=subprocess.STDOUT, check=True)
                        subprocess.run(["git", "-C", str(source), "apply", str(patch)], stdout=log, stderr=subprocess.STDOUT, check=True)
                record["sources"][name] = {"url": url, "commit": commit, "patch_sha256": sha256(ROOT / "third_party/patches" / patch_name) if patch_name else None}
        if args.build:
            steps = [("pgsr", "submodules/diff-plane-rasterization"), ("pgsr", "submodules/simple-knn"),
                     ("sof", "submodules/diff-gaussian-rasterization"), ("sof", "submodules/simple-knn"), ("fused-ssim", ".")]
            for index, (name, relative) in enumerate(steps):
                with (logdir / f"build-{index}-{name}.log").open("x") as log:
                    subprocess.run([str(python), "setup.py", "build_ext", "--inplace", "--force"], cwd=root / "source" / name / relative,
                                   env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            overlay = root / "python-overlay"
            # Never overwrite an existing overlay: validate it instead.
            if overlay.exists():
                import importlib.metadata
                versions = {d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions(path=[str(overlay)])}
                if versions.get("dacite") != "1.9.2" or versions.get("trimesh") != "4.7.4":
                    raise ValueError("Existing overlay incomplete/different; preserve it and use an independently registered runtime directory")
            else:
                with (logdir / "overlay.log").open("x") as log:
                    subprocess.run([str(python), "-m", "pip", "install", "--no-deps", "--target", str(overlay), "--report", str(logdir / "pip-report.json"),
                                    "dacite==1.9.2", "trimesh==4.7.4"], stdout=log, stderr=subprocess.STDOUT, check=True)
        for name in SOURCES:
            write_new_json(logdir / f"{name}-files.json", source_manifest(root / "source" / name))
        record["status"] = "sources_prepared_gpu_runtime_not_yet_validated"
    except BaseException as exc:
        record.update(status="preparation_failed_not_quality_failure", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_new_json(logdir / "result.json", record)
        print(json.dumps({"status": record["status"], "output": str(logdir)}), flush=True)


if __name__ == "__main__":
    main()
