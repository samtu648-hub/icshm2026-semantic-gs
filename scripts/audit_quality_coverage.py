#!/usr/bin/env python3
"""QUAL003 native all-one opacity vs sealed static semantic probabilities."""
import argparse
from dataclasses import asdict
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
from src.quality_coverage import coverage_counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("rgb-run", "semantic-run", "semantic-audit", "source", "artifact-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--extra-pythonpath", type=Path, action="append", default=[])
    a = p.parse_args()
    with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a.output.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        cp = ROOT / "experiments/runs/QUAL-003-COVERAGE-INIT-DIAG/config.yaml"
        cfg = json.loads(cp.read_text())
        record = {"experiment_id": cfg["experiment_id"], "stage": "native_coverage", "status": "initializing",
            "config_sha256": sha256(cp), "seed": cfg["seed"], "rgb_modified": False, "geometry_modified": False,
            "threshold_selection": False, "geometric_truth_claim": None,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
        try:
            import numpy as np
            import torch
            from PIL import Image
            from scripts.run_quality_rgb import source_manifest
            from src.quality_cameras import read_camera_requests
            from src.quality_native_gaussian import NativeGaussianRenderer
            from src.quality_scaffold import ScaffoldQualityRenderer
            from src.quality_semantic_fields import PrimitiveSemanticField
            from src.quality_static_training import TrainingLabels
            sem = json.loads((a.semantic_run / "result.json").read_text())
            audit = json.loads((a.semantic_audit / "result.json").read_text())
            rgb = json.loads((a.rgb_run / "result.json").read_text())
            if (sem["status"] != "completed_static_fit_pending_independent_evaluation" or sem["trained_steps"] != 10000
                    or sem["access"]["partition"] != "development"
                    or audit["status"] != "completed_independent_static_semantic_audit"
                    or audit["semantic_run_result_sha256"] != sha256(a.semantic_run / "result.json")
                    or sem["binding"]["parent_result_sha256"] != sha256(a.rgb_run / "result.json")):
                raise ValueError("Requires accepted full native static semantics from the matching RGB")
            verify_hashes(a.semantic_run, sem["artifacts"])
            verify_hashes(a.rgb_run, rgb["artifacts"])
            native = ("src/quality_semantic_fields.py", "src/quality_scaffold.py", "src/quality_native_gaussian.py",
                      "src/quality_cameras.py", "src/quality_backbone.py", "src/camera_adapter.py", "src/scaffold_semantic_sidecar.py")
            verify_hashes(ROOT, {n: sem["code_sha256"][n] for n in native})
            sources = source_manifest(a.source)
            if sources != json.loads((a.rgb_run / "source_manifest.json").read_text()):
                raise ValueError("RGB source mismatch")
            extras = rgb.get("extra_sources", [])
            if len(extras) != len(a.extra_pythonpath):
                raise ValueError("Auxiliary source count differs")
            for path, entry in zip(a.extra_pythonpath, extras):
                if source_manifest(path) != json.loads((a.rgb_run / entry["manifest"]).read_text()):
                    raise ValueError("Auxiliary source changed")
            code = ("scripts/audit_quality_coverage.py", "src/quality_coverage.py", "src/quality_protocol.py",
                    "src/quality_static_training.py", "scripts/run_quality_rgb.py", *native)
            record["code_sha256"] = {n: sha256(ROOT / n) for n in code}
            for name in (*code, str(cp.relative_to(ROOT))):
                dest = a.output / "code" / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, dest)
            protocol = json.loads((ROOT / cfg["protocol"]).read_text())
            verify_hashes(ROOT, protocol["input_hashes"])
            split = json.loads((ROOT / "experiments/splits/sem_001_scaffold_static_v1.json").read_text())
            names = sorted(split["semantic_validation_labeled"])
            if names != sorted(sem["prediction_views"]) or len(names) != 74:
                raise ValueError("Coverage target set differs from fixed development views")
            access = json.loads((a.rgb_run / "access.json").read_text())
            cameras = read_camera_requests(a.rgb_run / "scene/sparse/0")
            backbone = sem["binding"]["identity"]["backbone"]
            extension = "diff-plane-rasterization" if backbone == "pgsr" else "diff-gaussian-rasterization"
            sys.path[:0] = list(map(str, (a.source, a.source / "submodules" / extension, a.source / "submodules/simple-knn", *a.extra_pythonpath)))
            if backbone == "scaffold-gs":
                renderer = ScaffoldQualityRenderer(a.source, a.rgb_run / "model", 30000, train_camera_count=len(access["train"]))
            else:
                renderer = NativeGaussianRenderer(backbone, a.source, a.rgb_run / "model", 30000, train_camera_metadata=[cameras[n] for n in access["train"]])
            if asdict(renderer.identity) != sem["binding"]["identity"] or renderer.rgb_signature() != sem["frozen_rgb_signature"]:
                raise ValueError("Coverage renderer not the sealed semantic RGB parent")
            field = PrimitiveSemanticField(renderer)
            field.load_state_dict(torch.load(a.semantic_run / "semantic_field.pt", map_location="cpu", weights_only=True)["field"], renderer)
            sc = json.loads((ROOT / "experiments/runs/SEM-330-QUALITY-STATIC/config.yaml").read_text())
            manifest = json.loads((ROOT / sc["label_manifest"]).read_text())
            labels = TrainingLabels(a.artifact_root / sc["labels"], names, manifest, cfg["coverage"]["resolution"])
            frozen = json.loads((ROOT / protocol["frozen_manifest"]).read_text())
            frozen_hashes = {i["path"]: i["sha256"] for i in frozen["artifacts"]}
            verify_hashes(a.artifact_root, frozen_hashes)
            (a.output / "coverage").mkdir()
            before = renderer.render_pinhole(cameras[names[0]]).rgb.copy()
            tolerance = cfg["coverage"]["maximum_compositing_error"]
            per_view, max_error = {}, 0.
            with torch.no_grad():
                ones = torch.ones((len(renderer.field_positions), 1), dtype=renderer.field_positions.dtype, device="cuda")
                primitive_prob = field.logits.softmax(1)
                for i, name in enumerate(names):
                    alpha = renderer.render_channels(cameras[name], ones)[0].cpu().numpy()
                    summed = renderer.render_channels(cameras[name], primitive_prob).sum(0).cpu().numpy()
                    error = float(np.abs(alpha-summed).max())
                    max_error = max(max_error, error)
                    if error > tolerance:
                        raise ValueError("All-one coverage disagrees with native class-channel mass")
                    with np.load(a.semantic_run / "probabilities" / (Path(name).stem + ".npz"), allow_pickle=False) as z:
                        prob = z["probabilities"]
                    if not np.array_equal(prob, field.render_probabilities(renderer, cameras[name]).cpu().numpy()):
                        raise ValueError("Semantic replay changed while diagnosing opacity")
                    gt = labels.read(name)
                    stats = coverage_counts(alpha, prob, gt, cfg["coverage"]["alpha_bins"], tolerance)
                    rgb_image = renderer.render_pinhole(cameras[name]).rgb
                    photo_path = a.artifact_root / protocol["rgb"]["scene"] / "images" / name
                    with Image.open(photo_path) as im:
                        photo = np.asarray(im).astype(np.float32) / 255
                    if photo.shape != rgb_image.shape:
                        raise ValueError("Paired native RGB size mismatch")
                    mae = np.abs(photo-rgb_image).mean(2)
                    stats["rgb_mae_by_gt_class"] = {str(c): float(mae[gt == c].mean()) if (gt == c).any() else None for c in range(5)}
                    stats["paired_prepared_photo_sha256"] = sha256(photo_path)
                    stats["native_mass_max_abs_error"] = error
                    np.savez_compressed(a.output / "coverage" / (Path(name).stem + ".npz"), alpha=alpha,
                        residual_background=np.maximum(1-alpha, 0), low_coverage=(alpha < .5))
                    per_view[name] = stats
                    if i == 0 or (i+1) % 25 == 0:
                        print(json.dumps({"coverage_views": i+1, "total": len(names), "backbone": backbone}), flush=True)
            if renderer.rgb_signature() != sem["frozen_rgb_signature"] or not np.array_equal(before, renderer.render_pinhole(cameras[names[0]]).rgb):
                raise ValueError("Coverage diagnosis modified RGB")
            verify_hashes(ROOT, record["code_sha256"])
            verify_hashes(a.artifact_root, frozen_hashes)
            verify_hashes(a.semantic_run, sem["artifacts"])
            pooled = {str(c): {key: sum(v["classes"][str(c)][key] for v in per_view.values()) for key in per_view[names[0]]["classes"][str(c)]} for c in range(5)}
            write_new_json(a.output / "per_view.json", per_view)
            record.update(status="completed_native_coverage_diagnosis", backbone=backbone,
                binding=sem["binding"], semantic_run_sha256=sha256(a.semantic_run / "result.json"),
                semantic_audit_sha256=sha256(a.semantic_audit / "result.json"), views=names,
                classes=pooled, max_native_compositing_error=max_error, frozen_verified=len(frozen_hashes),
                low_coverage_pixels=sum(v["low_coverage_pixels"] for v in per_view.values()),
                target_label_hashes={n: labels.records[n]["label_png_sha256"] for n in names},
                input_hashes=protocol["input_hashes"], historical_development_diagnosis_not_causality=True,
                artifacts={str(p.relative_to(a.output)): sha256(p) for p in a.output.rglob("*") if p.is_file()})
        except BaseException as exc:
            record.update(status="native_coverage_diagnosis_failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["seconds"] = time.monotonic() - started
            write_new_json(a.output / "result.json", record)
            print(json.dumps({"status": record["status"], "output": str(a.output)}), flush=True)


if __name__ == "__main__":
    main()
