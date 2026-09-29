#!/usr/bin/env python3
"""Build only SEM340's isolated native sources using an existing read-only runtime."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, write_new_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--cuda", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {"status": "building", "steps": [], "script_sha256": sha256(Path(__file__))}
    try:
        import torch
        cfg = json.loads((ROOT / "experiments/runs/SEM-340-MASK2FORMER-PREFLIGHT/config.yaml").read_text())
        record["config_sha256"] = sha256(ROOT / "experiments/runs/SEM-340-MASK2FORMER-PREFLIGHT/config.yaml")
        record["torch"] = torch.__version__
        for directory, key in [("Mask2Former", "mask2former"), ("detectron2", "detectron2")]:
            path = a.root.resolve() / "vendor" / directory
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
            if commit != cfg["upstream"][key + "_commit"]:
                raise ValueError("Unregistered upstream checkout")
        if shutil.disk_usage(a.root).free < 100 * 1024**3:
            raise ValueError("100GiB disk reserve")
        site = Path(torch.__file__).resolve().parents[1]
        includes = [a.cuda.resolve() / "targets/x86_64-linux/include", *sorted((site / "nvidia").glob("*/include"))]
        libraries = [a.cuda.resolve() / "targets/x86_64-linux/lib", *sorted((site / "nvidia").glob("*/lib"))]
        env = os.environ.copy()
        env.update(CUDA_HOME=str(a.cuda.resolve()), FORCE_CUDA="1", TORCH_CUDA_ARCH_LIST="12.0",
                   MAX_JOBS="3", PYTHONDONTWRITEBYTECODE="1",
                   PATH=str(Path(sys.executable).parent) + os.pathsep + env["PATH"],
                   CPATH=os.pathsep.join(map(str, includes)), LIBRARY_PATH=os.pathsep.join(map(str, libraries)))
        record["build_environment"] = {k: env[k] for k in ["CUDA_HOME", "CPATH", "LIBRARY_PATH", "MAX_JOBS", "TORCH_CUDA_ARCH_LIST"]}
        for name, path in [("msda", a.root / "vendor/Mask2Former/mask2former/modeling/pixel_decoder/ops"),
                           ("detectron2", a.root / "vendor/detectron2")]:
            start = time.monotonic()
            with (a.output / (name + ".log")).open("x") as log:
                result = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"],
                                        cwd=path, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            record["steps"].append({"name": name, "exit_code": result.returncode, "seconds": time.monotonic()-start})
            if result.returncode:
                raise RuntimeError(name + " compilation failed; original log retained")
        record["status"] = "completed_native_build_not_model_validation"
    except Exception as exc:
        record.update(status="failed_build", error=repr(exc))
        raise
    finally:
        record["seconds"] = time.monotonic() - started
        write_new_json(a.output / "result.json", record)


if __name__ == "__main__":
    main()
