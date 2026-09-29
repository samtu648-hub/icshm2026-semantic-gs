#!/usr/bin/env python3
"""QUAL003 fixed PGSR checkpoints, same RGB116 views; no training/selection."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "audit", "source", "artifact-root", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--extra-pythonpath", type=Path, action="append", default=[])
    a = p.parse_args()
    with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a.output.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        cp = ROOT / "experiments/runs/QUAL-003-COVERAGE-INIT-DIAG/config.yaml"
        cfg = json.loads(cp.read_text())
        record = {"experiment_id": cfg["experiment_id"], "stage": "pgsr_checkpoints", "status": "initializing",
            "config_sha256": sha256(cp), "seed": 0, "training": False, "labels_decoded": False, "checkpoint_promotion": False,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
        code = ("scripts/audit_quality_pgsr_checkpoints.py", "scripts/run_quality_rgb.py", "scripts/run_quality_stage_a.py",
            "src/quality_native_gaussian.py", "src/quality_cameras.py", "src/quality_backbone.py", "src/quality_protocol.py", "src/camera_adapter.py")
        record["code_sha256"] = {n: sha256(ROOT / n) for n in code}
        try:
            import numpy as np
            from PIL import Image
            from scripts.run_quality_rgb import source_manifest
            from scripts.run_quality_stage_a import verify_run, verify_audit
            from src.quality_cameras import read_camera_requests
            from src.quality_native_gaussian import NativeGaussianRenderer
            protocol = json.loads((ROOT / cfg["protocol"]).read_text())
            parent = verify_run(a.run, mode="full", experiment_id="RGB-115-PGSR-SAFE-CYCLE",
                common_ply_sha256=cfg["initialization"]["point_ply_sha256"])
            verify_audit(a.audit, a.run, "completed_rgb_reload_audit")
            if source_manifest(a.source) != json.loads((a.run / "source_manifest.json").read_text()):
                raise ValueError("Native RGB source changed")
            if len(a.extra_pythonpath) != len(parent.get("extra_sources", [])):
                raise ValueError("Exact auxiliary sources required")
            for extra, entry in zip(a.extra_pythonpath, parent.get("extra_sources", [])):
                if source_manifest(extra) != json.loads((a.run / entry["manifest"]).read_text()):
                    raise ValueError("Auxiliary source changed")
            sys.path[:0] = list(map(str, [a.source, a.source / "submodules/diff-plane-rasterization", a.source / "submodules/simple-knn", *a.extra_pythonpath]))
            access = json.loads((a.run / "access.json").read_text())
            cameras = read_camera_requests(a.run / "scene/sparse/0")
            reference_path = ROOT / "outputs/quality/diagnostics/RGB-116-RGB-112-SOF-NVS-r1/result.json"
            reference = json.loads(reference_path.read_text())
            train = [sorted(access["train"])[i] for i in np.linspace(0, len(access["train"])-1, 12).astype(int)]
            if train != reference["train_sample"] or sorted(access["evaluation"]) != sorted(reference["evaluation"]):
                raise ValueError("Checkpoint views differ from fixed RGB116 protocol")
            groups = {"train_sample": train, "development": sorted(access["evaluation"])}
            frozen = json.loads((ROOT / protocol["frozen_manifest"]).read_text())
            hashes = {i["path"]: i["sha256"] for i in frozen["artifacts"]}
            verify_hashes(a.artifact_root, hashes)
            rows, aggregate, photo_hashes = {}, {}, {}
            max_author_error = 0
            for step in cfg["pgsr_checkpoints"]:
                renderer = NativeGaussianRenderer("pgsr", a.source, a.run / "model", step,
                    train_camera_metadata=[cameras[n] for n in access["train"]])
                signature = renderer.rgb_signature()
                aggregate[str(step)] = {}
                for group, names in groups.items():
                    metrics = []
                    for name in names:
                        rgb = renderer.render_pinhole(cameras[name]).rgb
                        path = a.artifact_root / protocol["rgb"]["scene"] / "images" / name
                        with Image.open(path) as im:
                            target = np.asarray(im).astype(np.float32) / 255
                        if rgb.shape != target.shape:
                            raise ValueError("Native paired RGB grid mismatch")
                        error = rgb-target
                        mse = float(np.mean(error**2))
                        value = {"PSNR": float(-10*np.log10(mse)) if mse > 0 else None, "MAE": float(np.mean(np.abs(error)))}
                        rows[f"{step}/{group}/{name}"] = value
                        metrics.append(value)
                        photo_hashes[str(path.relative_to(a.artifact_root))] = sha256(path)
                        if step == 30000 and group == "development":
                            with Image.open(a.audit / "rgb" / name) as im:
                                author = np.asarray(im).astype(np.int16)
                            actual = np.rint(rgb*255).astype(np.int16)
                            max_author_error = max(max_author_error, int(np.abs(actual-author).max()))
                    aggregate[str(step)][group] = {k: float(np.mean([v[k] for v in metrics if v[k] is not None])) for k in ("PSNR", "MAE")}
                if renderer.rgb_signature() != signature:
                    raise ValueError("Checkpoint diagnostic changed RGB tensors")
                del renderer
                print(json.dumps({"checkpoint": step, "metrics": aggregate[str(step)]}), flush=True)
            if max_author_error > 1:
                raise ValueError("PGSR final adapter disagrees with author beyond quantization")
            for name in (*code, str(cp.relative_to(ROOT))):
                dest = a.output / "code" / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, dest)
            write_new_json(a.output / "per_view.json", rows)
            verify_hashes(ROOT, record["code_sha256"])
            verify_hashes(a.artifact_root, hashes)
            verify_hashes(a.run, parent["artifacts"])
            record.update(status="completed_pgsr_checkpoint_diagnosis", parent_result_sha256=sha256(a.run / "result.json"),
                rgb_audit_sha256=sha256(a.audit / "result.json"), reference_sample_sha256=sha256(reference_path),
                train_sample=train, evaluation=groups["development"], checkpoints=aggregate,
                final_author_max_uint8_difference=max_author_error, frozen_verified=len(hashes),
                input_hashes=protocol["input_hashes"], paired_photo_hashes=photo_hashes,
                historical_loss_components="not recorded individually; cannot reconstruct or attribute a loss cause",
                artifacts={str(p.relative_to(a.output)): sha256(p) for p in a.output.rglob("*") if p.is_file()})
        except BaseException as exc:
            record.update(status="pgsr_checkpoint_diagnosis_failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["seconds"] = time.monotonic()-started
            write_new_json(a.output / "result.json", record)
            print(json.dumps({"status": record["status"], "output": str(a.output)}), flush=True)


if __name__ == "__main__":
    main()
