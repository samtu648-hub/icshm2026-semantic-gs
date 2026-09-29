#!/usr/bin/env python3
"""Read-only RGB history audit, using exact native SOF deployment rendering."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json
from scripts.run_quality_rgb import source_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "audit", "source", "artifact-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--extra-pythonpath", type=Path, action="append", default=[])
    args = parser.parse_args()
    config_path = ROOT / "experiments/runs/RGB-116-SOF-CHECKPOINT-DIAG/config.yaml"
    config = json.loads(config_path.read_text())
    parent = json.loads((args.run / "result.json").read_text())
    proof = json.loads((args.audit / "result.json").read_text())
    if parent["experiment_id"] not in config["parent_experiment"].split(",") or proof["run_result_sha256"] != sha256(args.run / "result.json") or proof["status"] != "completed_rgb_reload_audit":
        raise ValueError("Wrong SOF parent or unaccepted final audit")
    with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        record = {"experiment_id": config["experiment_id"], "status": "running", "parent": parent["experiment_id"],
                  "parent_result_sha256": sha256(args.run / "result.json"), "config_sha256": sha256(config_path),
                  "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "training": False, "labels_decoded": False, "checkpoint_promotion": False, "seed": parent["seed"], "checkpoints": {}}
        code = ("scripts/diagnose_sof_checkpoints.py", "scripts/run_quality_rgb.py", "src/quality_protocol.py",
                "src/quality_native_gaussian.py", "src/quality_scaffold.py", "src/quality_backbone.py", "src/quality_cameras.py", "src/camera_adapter.py", "src/scaffold_semantic_sidecar.py")
        record["code_sha256"] = {n: sha256(ROOT / n) for n in code}
        try:
            verify_hashes(args.run, parent["artifacts"])
            verify_hashes(ROOT, parent["inputs"])
            sources = source_manifest(args.source)
            if sources != json.loads((args.run / "source_manifest.json").read_text()):
                raise ValueError("SOF source changed since full training")
            extras = parent.get("extra_sources", [])
            if len(extras) != len(args.extra_pythonpath):
                raise ValueError("Provide exact native auxiliary source list")
            for entry, extra in zip(extras, args.extra_pythonpath):
                if source_manifest(extra) != json.loads((args.run / entry["manifest"]).read_text()):
                    raise ValueError("Auxiliary source changed")
            sys.path[:0] = list(map(str, [args.source.resolve(), args.source.resolve() / "submodules/diff-gaussian-rasterization",
                args.source.resolve() / "submodules/simple-knn", *[p.resolve() for p in args.extra_pythonpath]]))
            import numpy as np
            import torch
            from PIL import Image
            from src.quality_cameras import read_camera_requests
            from src.quality_native_gaussian import NativeGaussianRenderer
            from src.quality_scaffold import saved_namespace
            protocol = json.loads((ROOT / "experiments/runs/QUAL-001-FOUNDATION/config.yaml").read_text())
            frozen = json.loads((ROOT / protocol["frozen_manifest"]).read_text())
            hashes = {a["path"]: a["sha256"] for a in frozen["artifacts"]}
            verify_hashes(args.artifact_root, hashes)
            access = json.loads((args.run / "access.json").read_text())
            train = sorted(access["train"])
            sample = [train[i] for i in np.linspace(0, len(train) - 1, 12, dtype=int)]
            names = sample + sorted(access["evaluation"])
            if len(names) != len(set(names)) or len(train) != 303 or len(access["evaluation"]) != 74:
                raise ValueError("Unexpected diagnostic camera access")
            cameras = read_camera_requests(args.run / "scene/sparse/0")
            record.update(train_sample=sample, evaluation=sorted(access["evaluation"]), mesh_args=saved_namespace(args.run / "model/mesh_args"))
            write_new_json(args.output / "started.json", record)
            photo_hashes = json.loads((args.run / "rgb_input_hashes.json").read_text())
            for iteration in config["checkpoints"]:
                renderer = NativeGaussianRenderer("sof", args.source, args.run / "model", iteration,
                    train_camera_metadata=[cameras[n] for n in access["train"]])
                signature = renderer.rgb_signature()
                rows = []
                for name in names:
                    path = args.run / "scene/images" / name
                    if sha256(path) != photo_hashes[name]:
                        raise ValueError("Reference photo changed")
                    with Image.open(path) as image:
                        gt = np.array(image.convert("RGB"), dtype=np.float32) / 255
                    rgb = renderer.render_pinhole(cameras[name]).rgb
                    if rgb.shape != gt.shape or rgb.shape != (989, 1320, 3) or not np.isfinite(rgb).all():
                        raise ValueError("Invalid native diagnostic RGB")
                    error = rgb.astype(np.float64) - gt
                    mse = float(np.mean(error * error))
                    row = {"view": name, "partition": "train_sample" if name in sample else "development",
                           "PSNR": float(-10 * np.log10(max(mse, np.finfo(float).tiny))), "MAE": float(np.abs(error).mean())}
                    if iteration == 30000 and name in access["evaluation"]:
                        with Image.open(args.audit / "rgb" / name) as image:
                            author = np.array(image.convert("RGB"))
                        maximum = int(np.abs(np.rint(rgb * 255).astype(np.int16) - author.astype(np.int16)).max())
                        row["author_vs_adapter_max_uint8"] = maximum
                        if maximum > 1:
                            raise ValueError("Final deployment adapter differs from audited author RGB beyond quantization")
                    rows.append(row)
                    if name in (sample[0], "054.png", "159.png"):
                        Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(args.output / f"{iteration}-{name}")
                if renderer.rgb_signature() != signature:
                    raise ValueError("Read-only diagnostic changed model")
                summary = {scope: {metric: float(np.mean([row[metric] for row in rows if row["partition"] == scope])) for metric in ("PSNR", "MAE")}
                           for scope in ("train_sample", "development")}
                write_new_json(args.output / f"iteration-{iteration}.json", {"summary": summary, "views": rows})
                record["checkpoints"][str(iteration)] = summary
                print(json.dumps({"iteration": iteration, "summary": summary}), flush=True)
                del renderer
                torch.cuda.empty_cache()
            verify_hashes(args.artifact_root, hashes)
            verify_hashes(args.run, parent["artifacts"])
            verify_hashes(ROOT, record["code_sha256"])
            if source_manifest(args.source) != sources or sha256(config_path) != record["config_sha256"]:
                raise ValueError("Diagnostic source/config changed")
            record.update(status="completed_checkpoint_association_not_causal_intervention", frozen_verified=len(hashes),
                          final_author_adapter_within_one_level=True, mesh_args_sha256=sha256(args.run / "model/mesh_args"))
        except BaseException as exc:
            record.update(status="checkpoint_diagnostic_failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["seconds"] = time.monotonic() - started
            write_new_json(args.output / "result.json", record)
            print(json.dumps({"status": record["status"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
