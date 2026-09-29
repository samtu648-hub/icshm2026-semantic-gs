#!/usr/bin/env python3
"""Two fresh native semantic replays followed by fixed full-image label scoring."""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.quality_semantic_audit import check_replay, independent_confusion, metrics
from src.quality_static_training import TrainingLabels, semantic_access
from scripts.run_quality_rgb import source_manifest

RENDER_CODE = ("src/quality_semantic_fields.py", "src/quality_scaffold.py", "src/quality_native_gaussian.py",
               "src/quality_cameras.py", "src/quality_backbone.py", "src/camera_adapter.py", "src/scaffold_semantic_sidecar.py")


def replay(args, result, parent):
    import numpy as np
    import torch
    from PIL import Image
    from src.quality_cameras import read_camera_requests
    from src.quality_native_gaussian import NativeGaussianRenderer
    from src.quality_scaffold import ScaffoldQualityRenderer
    from src.quality_semantic_fields import PrimitiveSemanticField
    cameras = read_camera_requests(args.rgb_run / "scene/sparse/0")
    access = json.loads((args.rgb_run / "access.json").read_text())
    backbone = result["binding"]["identity"]["backbone"]
    if backbone == "scaffold-gs":
        renderer = ScaffoldQualityRenderer(args.source, args.rgb_run / "model", 30000, train_camera_count=len(access["train"]))
    else:
        renderer = NativeGaussianRenderer(backbone, args.source, args.rgb_run / "model", 30000,
            train_camera_metadata=[cameras[n] for n in access["train"]])
    if asdict(renderer.identity) != result["binding"]["identity"] or renderer.rgb_signature() != result["frozen_rgb_signature"]:
        raise ValueError("Reloaded RGB deployment identity/signature changed")
    field = PrimitiveSemanticField(renderer)
    payload = torch.load(args.run / "semantic_field.pt", map_location="cpu", weights_only=True)
    if payload["binding"] != result["binding"] or payload["access"] != result["access"] or payload["trained_steps"] != 10000:
        raise ValueError("Checkpoint training provenance differs")
    field.load_state_dict(payload["field"], renderer)
    before = {n: renderer.render_pinhole(cameras[n]).rgb.copy() for n in result["fixed_rgb_views_exact"]}
    with torch.no_grad():
        for name in result["prediction_views"]:
            actual = field.render_probabilities(renderer, cameras[name]).cpu().numpy()
            with np.load(args.run / "probabilities" / (Path(name).stem + ".npz"), allow_pickle=False) as archive:
                stored = archive["probabilities"]
            with Image.open(args.run / "predictions" / name) as image:
                png = np.array(image)
            check_replay(actual, stored, png)
    if renderer.rgb_signature() != result["frozen_rgb_signature"] or any(not np.array_equal(rgb, renderer.render_pinhole(cameras[n]).rgb) for n, rgb in before.items()):
        raise ValueError("Independent semantic replay modified RGB")
    write_new_json(args.output, {"status": "passed_fresh_native_semantic_replay", "views": result["prediction_views"],
        "probabilities_exact": True, "rgb_exact": True, "labels_decoded": False, "pid": os.getpid(),
        "parent_result_sha256": sha256(args.run / "result.json")})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "rgb-run", "source", "artifact-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--extra-pythonpath", type=Path, action="append", default=[])
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    result = json.loads((args.run / "result.json").read_text())
    parent = json.loads((args.rgb_run / "result.json").read_text())
    if args.worker:
        replay(args, result, parent)
        return
    with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        record = {"experiment_id": "SEM-330-QUALITY-STATIC", "status": "running", "quality_metrics": None,
                  "semantic_run_result_sha256": sha256(args.run / "result.json"), "target_labels_decoded": False,
                  "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
        audit_config = ROOT / "experiments/runs/SEM-330-QUALITY-STATIC/audit_config.yaml"
        record["audit_config_sha256"] = sha256(audit_config)
        code = ("scripts/audit_quality_static.py", "src/quality_semantic_audit.py", "src/quality_static_training.py",
                "scripts/run_quality_rgb.py", "src/quality_protocol.py", *RENDER_CODE)
        record["code_sha256"] = {n: sha256(ROOT / n) for n in code}
        try:
            if result["status"] != "completed_static_fit_pending_independent_evaluation" or result["trained_steps"] != 10000:
                raise ValueError("Requires completed registered full semantic fit")
            binding = result["binding"]
            if binding["parent_result_sha256"] != sha256(args.rgb_run / "result.json"):
                raise ValueError("Wrong RGB parent")
            verify_hashes(args.run, result["artifacts"])
            verify_hashes(args.run / "code", result["code_sha256"])
            # Training launchers may evolve; deployment functions must still match.
            verify_hashes(ROOT, {n: result["code_sha256"][n] for n in RENDER_CODE})
            verify_hashes(ROOT, binding["input_hashes"])
            verify_hashes(args.rgb_run, parent["artifacts"])
            sources = source_manifest(args.source)
            if sources != json.loads((args.rgb_run / "source_manifest.json").read_text()):
                raise ValueError("Native source changed")
            extras = parent.get("extra_sources", [])
            if len(extras) != len(args.extra_pythonpath):
                raise ValueError("Auxiliary source count differs")
            for entry, extra in zip(extras, args.extra_pythonpath):
                if source_manifest(extra) != json.loads((args.rgb_run / entry["manifest"]).read_text()):
                    raise ValueError("Auxiliary source bytes changed")
            config = json.loads((ROOT / "experiments/runs/SEM-330-QUALITY-STATIC/config.yaml").read_text())
            protocol = json.loads((ROOT / config["protocol"]).read_text())
            manifest = json.loads((ROOT / config["label_manifest"]).read_text())
            frozen = json.loads((ROOT / protocol["frozen_manifest"]).read_text())
            hashes = {a["path"]: a["sha256"] for a in frozen["artifacts"]}
            verify_hashes(args.artifact_root, hashes)
            split = json.loads((ROOT / "experiments/splits/sem_001_scaffold_static_v1.json").read_text())
            oof = json.loads((ROOT / "experiments/splits/sem_005_oof_spatial_v1.json").read_text())
            access = semantic_access(split, oof, json.loads((args.rgb_run / "access.json").read_text()), result["access"]["partition"])
            if access != result["access"] or result["prediction_views"] != access["prediction"]:
                raise ValueError("Audit target access differs from registered training partition")
            source = args.source.resolve()
            extension = "diff-plane-rasterization" if binding["identity"]["backbone"] == "pgsr" else "diff-gaussian-rasterization"
            env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, [source, source / "submodules" / extension,
                source / "submodules/simple-knn", ROOT, *[p.resolve() for p in args.extra_pythonpath]])), OMP_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1")
            workers = []
            for i in range(2):
                destination = args.output / f"replay-{i}.json"
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--run", str(args.run.resolve()),
                    "--rgb-run", str(args.rgb_run.resolve()), "--source", str(source), "--artifact-root", str(args.artifact_root.resolve()), "--output", str(destination.resolve())]
                with (args.output / f"replay-{i}.log").open("x") as log:
                    subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                workers.append(json.loads(destination.read_text()))
            if len({w["pid"] for w in workers}) != 2 or any(w["status"] != "passed_fresh_native_semantic_replay" for w in workers):
                raise ValueError("Two independent native replays did not pass")
            import numpy as np
            from PIL import Image
            labels = TrainingLabels(args.artifact_root / config["labels"], access["prediction"], manifest, config["resolution"])
            pooled = np.zeros((5, 5), np.int64)
            per_view = {}
            for name in access["prediction"]:
                target = labels.read(name)
                record["target_labels_decoded"] = True
                with Image.open(args.run / "predictions" / name) as image:
                    prediction = np.array(image)
                counts = independent_confusion(target, prediction)
                pooled += counts
                per_view[name] = metrics(counts)
            write_new_json(args.output / "per_view.json", per_view)
            verify_hashes(args.artifact_root, hashes)
            verify_hashes(args.run, result["artifacts"])
            verify_hashes(args.rgb_run, parent["artifacts"])
            verify_hashes(ROOT, record["code_sha256"])
            if source_manifest(source) != sources or sha256(audit_config) != record["audit_config_sha256"]:
                raise ValueError("Audit source/config changed")
            record.update(status="completed_independent_static_semantic_audit", quality_metrics=metrics(pooled),
                          fresh_reloads=workers, seed=binding["seed"], binding=binding, access=access,
                          frozen_verified=len(hashes), historical_development_or_shared_rgb_oof=True,
                          target_label_hashes={n: labels.records[n]["label_png_sha256"] for n in access["prediction"]})
        except BaseException as exc:
            record.update(status="semantic_audit_failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["seconds"] = time.monotonic() - started
            write_new_json(args.output / "result.json", record)
            print(json.dumps({"status": record["status"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
