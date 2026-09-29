#!/usr/bin/env python3
"""One isolated registered reconstruction run; no implicit retry or promotion."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json


def source_manifest(source):
    suffixes = {".py", ".cu", ".cpp", ".cuh", ".h", ".hpp", ".json", ".so"}
    return {str(p.relative_to(source)): sha256(p) for p in sorted(source.rglob("*"))
            if p.is_file() and not {".git", "build", "__pycache__"} & set(p.relative_to(source).parts)
            and (p.suffix in suffixes or "LICENSE" in p.name or "NOTICE" in p.name or p.name in ("METADATA", "RECORD", "WHEEL"))}


def materialize_scene(original, target, backbone):
    """Copy mutable metadata/PLY; share image bytes read-only, never original sparse."""
    target.mkdir(parents=True, exist_ok=False)
    (target / "images").symlink_to((original / "images").resolve(), target_is_directory=True)
    shutil.copytree(original / "sparse/0", target / "sparse/0")
    shutil.copy2(original / "preparation_manifest.json", target / "preparation_manifest.json")
    if backbone == "pgsr":
        for item in (target / "sparse/0").iterdir():
            if item.is_file():
                shutil.copy2(item, target / "sparse" / item.name)


def audit_pgsr_neighbors(path, train_names):
    """Audit the author's actual neighbor list, not estimated geometry quality."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    allowed = set(train_names)
    def name(value):
        return value if value.endswith(".png") else value + ".png"
    seen, counts = set(), {}
    for row in rows:
        target = name(row["ref_name"])
        sources = [name(s) for s in row["nearest_name"]]
        if target not in allowed or target in seen or target in sources or len(sources) != len(set(sources)) or not set(sources) <= allowed:
            raise ValueError("PGSR native neighbor set violates train-only access")
        seen.add(target)
        counts[target] = len(sources)
    if seen != allowed:
        raise ValueError("PGSR neighbor record omits training cameras")
    return {"eligible_camera_fraction": sum(v > 0 for v in counts.values()) / len(counts),
            "empty_neighbor_cameras": sorted(n for n, c in counts.items() if not c),
            "neighbor_counts": counts, "scope": "actual camera eligibility; not per-step or pixel coverage",
            "source_sha256": sha256(path), "empty_behavior": "native skip; virtual cameras disabled by registered defaults"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True, choices=("RGB-110-SCAFFOLD-CTRL", "RGB-111-PGSR", "RGB-112-SOF-NVS", "RGB-113-SOF-GEO", "RGB-115-PGSR-SAFE-CYCLE"))
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--quality-root", type=Path, default=ROOT / "outputs/quality")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--cuda", type=Path, required=True)
    p.add_argument("--mode", choices=("smoke", "full"), required=True)
    p.add_argument("--attempt", required=True)
    p.add_argument("--extra-pythonpath", type=Path, action="append", default=[])
    p.add_argument("--image-cache-device", choices=("cpu", "cuda"), help="Explicit lossless image-cache placement override; no resolution change")
    p.add_argument("--smoke-run", type=Path, help="Required for full RGB115 only: exact matching successful smoke")
    args = p.parse_args()
    if not args.attempt.replace("-", "").isalnum():
        raise ValueError("Attempt must be a simple alphanumeric identifier")
    configpath = ROOT / "experiments/runs" / args.experiment / "config.yaml"
    config = json.loads(configpath.read_text())
    repair = args.experiment == "RGB-115-PGSR-SAFE-CYCLE"
    if (repair and args.mode == "full") != (args.smoke_run is not None):
        p.error("--smoke-run is required for full RGB115 only")
    if repair:
        if args.image_cache_device not in (None, config["image_cache_device"]):
            p.error("Repair trial must keep its registered CPU image cache")
        args.image_cache_device = config["image_cache_device"]
    protocolpath = ROOT / config["protocol"]
    protocol = json.loads(protocolpath.read_text())
    source = args.source.resolve()
    # Do not resolve the Python symlink: doing so would escape its virtualenv.
    python = args.python.absolute()
    if not python.is_file() or not (args.cuda / "bin/nvcc").is_file():
        raise FileNotFoundError("Configured isolated Python/CUDA toolchain unavailable")
    args.quality_root = args.quality_root.resolve()
    args.quality_root.mkdir(parents=True, exist_ok=True)
    with (args.quality_root / "gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = args.quality_root / "runs" / f"{args.experiment}-{args.mode}-{args.attempt}"
        run.mkdir(parents=True, exist_ok=False)
        record = {"experiment_id": args.experiment, "mode": args.mode, "attempt": args.attempt,
                  "seed": config["seed"], "config_sha256": sha256(configpath), "protocol_sha256": sha256(protocolpath),
                  "status": "preflight", "started_unix": time.time(), "quality_metrics": None}
        record["image_cache_device_override"] = args.image_cache_device
        start = time.monotonic()
        try:
            record["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            record["inputs"] = verify_hashes(ROOT, protocol["input_hashes"])
            frozen = json.loads((ROOT / protocol["frozen_manifest"]).read_text())
            frozen_hashes = {a["path"]: a["sha256"] for a in frozen["artifacts"]}
            verify_hashes(args.artifact_root, frozen_hashes)
            if shutil.disk_usage(run).free < protocol["execution"]["minimum_free_gib"] * 2**30:
                raise RuntimeError("Insufficient free disk reserve")
            commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
            if commit != config["upstream_commit"]:
                raise ValueError("Unpinned upstream source")
            record["source_commit"] = commit
            source_files = source_manifest(source)
            if repair:
                from src.quality_pgsr_prerequisite import check_repair
                record["repair_prerequisite"] = check_repair(ROOT, config, source_files)
                subprocess.run(["git", "-C", str(source), "apply", "--reverse", "--check", str(ROOT / config["source_delta_patch"])], check=True)
                extra_inputs = ("experiments/runs/RGB-115-PGSR-SAFE-CYCLE/execution_config.yaml", config["source_delta_patch"])
                record["inputs"].update({n: sha256(ROOT / n) for n in extra_inputs})
            write_new_json(run / "source_manifest.json", source_files)
            diff = subprocess.check_output(["git", "-C", str(source), "diff", "--binary"])
            (run / "upstream.patch").write_bytes(diff)
            record["source_manifest_sha256"] = sha256(run / "source_manifest.json")
            record["source_diff_sha256"] = hashlib.sha256(diff).hexdigest()
            record["extra_sources"] = []
            for extra in args.extra_pythonpath:
                extra = extra.resolve()
                extra_top = subprocess.run(["git", "-C", str(extra), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
                is_checkout = extra_top.returncode == 0 and Path(extra_top.stdout.strip()).resolve() == extra
                extra_commit = subprocess.check_output(["git", "-C", str(extra), "rev-parse", "HEAD"], text=True).strip() if is_checkout else None
                snapshot = source_manifest(extra)
                filename = f"extra_source_{len(record['extra_sources'])}.json"
                write_new_json(run / filename, snapshot)
                record["extra_sources"].append({"name": extra.name, "commit": extra_commit, "manifest": filename, "manifest_sha256": sha256(run / filename)})
            code_names = ["scripts/run_quality_rgb.py", "scripts/quality_upstream_worker.py", "src/quality_protocol.py", "src/quality_numeric.py"]
            if repair:
                code_names.extend(["src/quality_pgsr_geometry.py", "src/quality_pgsr_prerequisite.py"])
            record["code_sha256"] = {name: sha256(ROOT / name) for name in code_names}
            for name in code_names:
                destination = run / "code" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, destination)
            if args.smoke_run is not None:
                from src.quality_pgsr_prerequisite import validate_smoke
                smoke = json.loads((args.smoke_run / "result.json").read_text())
                validate_smoke(smoke, experiment=args.experiment, config_hash=record["config_sha256"],
                    protocol_hash=record["protocol_sha256"], code_hashes=record["code_sha256"],
                    source_hash=record["source_manifest_sha256"], extra_sources=record["extra_sources"])
                verify_hashes(args.smoke_run, smoke["artifacts"])
                verify_hashes(ROOT, smoke["inputs"])
                common_ply = args.artifact_root / protocol["rgb"]["scene"] / "sparse/0/points3D.ply"
                if sha256(args.smoke_run / "model/input.ply") != sha256(common_ply):
                    raise ValueError("Repair smoke does not use common initialization")
                record["smoke_result_sha256"] = sha256(args.smoke_run / "result.json")
            rgb = json.loads((ROOT / "experiments/splits/rgb_r1_spatial_v1.json").read_text())["primary_fold"]
            scene = run / "scene"
            original = args.artifact_root / protocol["rgb"]["scene"]
            actual_names = {f.name for f in (original / "images").glob("*.png")}
            if actual_names != set(rgb["train_rgb_400"]) | set(rgb["validation_labeled"]):
                raise ValueError("Prepared scene includes incorrect images")
            if sorted((original / "sparse/0/test.txt").read_text().splitlines()) != sorted(rgb["validation_labeled"]):
                raise ValueError("Prepared scene validation mismatch")
            materialize_scene(original, scene, config["backbone"])
            write_new_json(run / "access.json", {"train": rgb["train_rgb_400"], "evaluation": rgb["validation_labeled"]})
            record["scene_input_sha256"] = {str(f.relative_to(scene)): sha256(f) for f in sorted(scene.rglob("*")) if f.is_file() and f.suffix in (".txt", ".json", ".ply")}
            # Hash actual RGB bytes once per run; no GT labels are decoded.
            write_new_json(run / "rgb_input_hashes.json", {name: sha256(original / "images" / name) for name in sorted(actual_names)})
            extensions = ([source / "submodules/diff-plane-rasterization"] if config["backbone"] == "pgsr" else [source / "submodules/diff-gaussian-rasterization"])
            paths = extensions + [source / "submodules/simple-knn", source, ROOT] + [v.resolve() for v in args.extra_pythonpath]
            env = dict(os.environ, CUDA_HOME=str(args.cuda), PYTHONPATH=os.pathsep.join(map(str, paths)),
                       PATH=f"{python.parent}:{args.cuda}/bin:" + os.environ.get("PATH", ""),
                       LD_LIBRARY_PATH=f"{args.cuda}/lib64:" + os.environ.get("LD_LIBRARY_PATH", ""),
                       TORCH_CUDA_ARCH_LIST="12.0", OMP_NUM_THREADS="4", HF_HUB_OFFLINE="1", WANDB_MODE="disabled",
                       PYTHONDONTWRITEBYTECODE="1", TORCH_HOME=str(args.artifact_root / "r1/weights/torch"))
            iterations = protocol["rgb"]["smoke_iterations" if args.mode == "smoke" else "iterations"]
            saves = [iterations] if args.mode == "smoke" else protocol["rgb"]["save_iterations"]
            common = ["--source_path", str(scene), "--model_path", str(run / "model"), "--eval", "--resolution", "1", "--iterations", str(iterations),
                      "--test_iterations", str(iterations), "--save_iterations", *map(str, saves), "--checkpoint_iterations", *map(str, saves), "--quiet"]
            if args.image_cache_device:
                common.extend(["--data_device", args.image_cache_device])
            worker = run / "code/scripts/quality_upstream_worker.py"
            command = [str(python), str(worker), "--source", str(source), "--access", str(run / "access.json"), "--evidence", str(run / "worker_train.json"),
                       "--seed", str(config["seed"]), "--", *common, *config["training_arguments"]]
            record["command"] = command
            write_new_json(run / "started.json", record)
            with (run / "train.log").open("x") as log:
                subprocess.run(command, env=env, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True)
            record["training"] = json.loads((run / "worker_train.json").read_text())
            if record["training"].get("actual_train") != sorted(rgb["train_rgb_400"]):
                raise ValueError("No verified native training access")
            record["artifacts"] = {str(f.relative_to(run)): sha256(f) for f in sorted((run / "model").rglob("*")) if f.is_file() and f.suffix in (".ply", ".pth", ".pt")}
            if not record["artifacts"]:
                raise RuntimeError("Native training produced no checkpoint")
            # Execute in the configured scientific runtime, not the lightweight
            # launcher Python. Keep every rejected model and its numeric inventory.
            numeric_command = [str(python), "-c",
                "import sys; from pathlib import Path; from src.quality_numeric import ply_inventory; "
                "from src.quality_protocol import write_new_json; root=Path(sys.argv[1]); "
                "write_new_json(root/'numeric_inventory.json', {str(p.relative_to(root)): ply_inventory(p) "
                "for p in sorted((root/'model/point_cloud').glob('iteration_*/point_cloud.ply'))})", str(run)]
            subprocess.run(numeric_command, cwd=ROOT, env=env, check=True)
            # The validation itself uses stdlib only; avoid requiring NumPy in launcher.
            record["numeric_inventory"] = json.loads((run / "numeric_inventory.json").read_text())
            if not record["numeric_inventory"] or any(not v["finite"] for v in record["numeric_inventory"].values()):
                raise FloatingPointError("Nonfinite or missing saved model parameters; see numeric_inventory.json")
            if config["backbone"] == "pgsr":
                record["multi_view_access"] = audit_pgsr_neighbors(run / "model/multi_view.json", rgb["train_rgb_400"])
            verify_hashes(args.artifact_root, frozen_hashes)
            if source_manifest(source) != source_files:
                raise ValueError("Upstream source changed during training")
            for entry, extra in zip(record["extra_sources"], args.extra_pythonpath):
                if source_manifest(extra.resolve()) != json.loads((run / entry["manifest"]).read_text()):
                    raise ValueError("Supplemental dependency changed during training")
            verify_hashes(ROOT, record["code_sha256"])
            verify_hashes(ROOT, record["inputs"])
            record["frozen_count_verified_after"] = len(frozen_hashes)
            record["status"] = "completed_native_training_pending_independent_reload_and_evaluation"
        except BaseException as exc:
            record.update(status="execution_failed_not_quality_failure", error=f"{type(exc).__name__}: {exc}")
            evidence = run / "worker_train.json"
            if evidence.is_file():
                record["training"] = json.loads(evidence.read_text())
                record["failure_category"] = record["training"].get("failure_category", "runtime")
            else:
                record["failure_category"] = "preflight"
            if isinstance(exc, FloatingPointError):
                record["failure_category"] = "numeric_saved_model_integrity"
            raise
        finally:
            record["seconds"] = time.monotonic() - start
            write_new_json(run / "result.json", record)
            print(json.dumps({"status": record["status"], "run": str(run), "seconds": record["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
