#!/usr/bin/env python3
"""Run the registered engineering preflight for RGB147-149."""

from __future__ import annotations

from _artifact_paths import artifact_root_default, parse_artifact_root

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_protocol import sha256, write_new_json


CONFIGS = {
    "control": ROOT
    / "experiments/runs/RGB-147-STANDARD-GT-SUPERVISION-CTRL/config.yaml",
    "region_rgb": ROOT / "experiments/runs/RGB-148-STANDARD-GT-REGION-RGB/config.yaml",
    "region_gradient": ROOT
    / "experiments/runs/RGB-149-STANDARD-GT-REGION-GRADIENT/config.yaml",
}


def artifact_environment(artifact_root: Path, source: Path) -> tuple[Path, dict]:
    cuda = artifact_root / "r1/toolchain/cuda-12.8"
    python = artifact_root / "r1/envs/r1-cuda128/bin/python"
    environment = os.environ.copy()
    environment.update(
        CUDA_HOME=str(cuda),
        PATH=str(cuda / "bin") + os.pathsep + environment.get("PATH", ""),
        LD_LIBRARY_PATH=str(cuda / "lib64")
        + os.pathsep
        + environment.get("LD_LIBRARY_PATH", ""),
        PYTHONPATH=os.pathsep.join(
            map(
                str,
                [
                    source,
                    source / "submodules/diff-gaussian-rasterization",
                    source / "submodules/simple-knn",
                    ROOT,
                ],
            )
        ),
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        PYTHONDONTWRITEBYTECODE="1",
    )
    if not python.is_file():
        raise FileNotFoundError(python)
    return python, environment


def run_logged(command: list[str], cwd: Path, environment: dict, log: Path) -> None:
    with log.open("x") as stream:
        subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def tensor_deltas(left, right, path="root") -> dict[str, float]:
    import torch

    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"Checkpoint tensor metadata differs at {path}")
        return {
            path: 0.0
            if not left.numel()
            else float((left.detach() - right.detach()).abs().max())
        }
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"Checkpoint array metadata differs at {path}")
        return {path: 0.0 if not left.size else float(np.max(np.abs(left - right)))}
    if isinstance(left, dict) and isinstance(right, dict):
        if list(left) != list(right):
            raise ValueError(f"Checkpoint keys differ at {path}")
        output = {}
        for key in left:
            output.update(tensor_deltas(left[key], right[key], f"{path}.{key}"))
        return output
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        if len(left) != len(right):
            raise ValueError(f"Checkpoint length differs at {path}")
        output = {}
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            output.update(tensor_deltas(left_item, right_item, f"{path}[{index}]"))
        return output
    if left != right or type(left) is not type(right):
        raise ValueError(f"Checkpoint metadata differs at {path}")
    return {}


def validate_labels() -> dict:
    root = ROOT / "outputs/quality/all_data/SEM-370-r1"
    manifest_path = root / "label_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["class_mapping"] != {
        "background": 0,
        "deck": 1,
        "foundation": 4,
        "stay_cable": 2,
        "tower": 3,
    }:
        raise ValueError("Unexpected semantic class mapping")
    if (
        manifest["image_count"] != 300
        or manifest["output_grid"] != "R1 full-frame PINHOLE, 1320x989, same K, no crop"
    ):
        raise ValueError("Unexpected semantic label grid")
    names = []
    pixel_counts = {str(index): 0 for index in range(5)}
    for record in manifest["records"]:
        path = root / record["label_png"]
        if sha256(path) != record["label_png_sha256"]:
            raise ValueError(f"Label hash changed: {path}")
        label = np.asarray(Image.open(path))
        if label.shape != (989, 1320) or label.dtype != np.uint8:
            raise ValueError(
                f"Unexpected label grid: {path} {label.shape} {label.dtype}"
            )
        values, counts = np.unique(label, return_counts=True)
        if not set(values.tolist()).issubset(set(range(5))):
            raise ValueError(f"Unexpected class ID: {path}")
        for value, count in zip(values, counts):
            pixel_counts[str(int(value))] += int(count)
        names.append(path.name)
    if len(names) != 300 or len(set(names)) != 300:
        raise ValueError("Label manifest does not contain 300 unique images")
    return {
        "manifest_sha256": sha256(manifest_path),
        "verified_images": len(names),
        "shape": [989, 1320],
        "grid": manifest["output_grid"],
        "class_mapping": manifest["class_mapping"],
        "pixel_counts": pixel_counts,
    }


def read_camera_sequence(path: Path) -> list[tuple[int, str]]:
    sequence = []
    for line in path.read_text().splitlines():
        iteration, name = line.split("\t")
        sequence.append((int(iteration), name))
    return sequence


def check_gradient_summary(path: Path, expected_calls: int) -> dict:
    summary = json.loads(path.read_text())
    if summary["calls"] != expected_calls or summary["labeled_calls"] <= 0:
        raise ValueError(f"Auxiliary path was not exercised: {path}")
    probe = summary["parameter_gradient_probe"]
    for name in ("xyz", "features_dc"):
        if not probe[name]["finite"] or probe[name]["norm"] <= 0:
            raise ValueError(f"Missing finite nonzero {name} auxiliary gradient")
    geometry = [probe[name] for name in ("xyz", "scaling", "rotation")]
    if not any(item["finite"] and item["norm"] > 0 for item in geometry):
        raise ValueError("No finite nonzero geometry auxiliary gradient")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root", type=Path, default=artifact_root_default()
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/quality/rgb147/preflight-r4"
    )
    parser.add_argument("--iterations", type=int, default=32)
    args = parser.parse_args()
    if args.iterations < 2:
        raise ValueError("Preflight needs at least two iterations")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {
        "schema_version": 1,
        "status": "running",
        "iterations": args.iterations,
        "quality_gate": False,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    try:
        configs = {name: json.loads(path.read_text()) for name, path in CONFIGS.items()}
        source = ROOT / configs["control"]["source"]
        scene = ROOT / configs["control"]["data"]["scene"]
        python, environment = artifact_environment(args.artifact_root.resolve(), source)
        result["labels"] = validate_labels()

        calibration_path = ROOT / "outputs/quality/rgb147/calibration-r2/result.json"
        calibration = json.loads(calibration_path.read_text())
        if calibration["status"] != "completed_pretraining_calibration":
            raise ValueError("Calibration is incomplete")
        if calibration["synthetic_mask_edge_check"]["correct_full_image_first"] != 0:
            raise ValueError("Full-image-first synthetic gradient check failed")
        if calibration["synthetic_mask_edge_check"]["wrong_mask_first"] <= 0:
            raise ValueError("Mask-first negative control did not expose a false edge")
        if len(calibration["diagnostic_panels"]) != 9:
            raise ValueError("Expected nine fixed-view diagnostic panels")
        for record in calibration["diagnostic_panels"]:
            if not (calibration_path.parent / record["path"]).is_file():
                raise FileNotFoundError(record["path"])
        result["calibration"] = {
            "path": str(calibration_path.relative_to(ROOT)),
            "sha256": sha256(calibration_path),
            "lambda_region_rgb": calibration["lambda_region_rgb"],
            "lambda_region_gradient": calibration["lambda_region_gradient"],
            "gradient_norm_medians": calibration["gradient_norm_medians"],
            "synthetic_mask_edge_check": calibration["synthetic_mask_edge_check"],
            "diagnostic_panel_count": len(calibration["diagnostic_panels"]),
        }
        for mode, key in (
            ("region_rgb", "lambda_region_rgb"),
            ("region_gradient", "lambda_region_gradient"),
        ):
            config = configs[mode]
            if (
                config["auxiliary_supervision"]["coefficient_status"]
                != "frozen_pretraining"
            ):
                raise ValueError(f"{mode} coefficient is not frozen")
            if config["auxiliary_supervision"]["coefficient"] != calibration[key]:
                raise ValueError(f"{mode} coefficient differs from calibration")

        native_runs = []
        for repeat in (1, 2):
            native = args.output / f"native-control-{repeat}"
            native.mkdir()
            native_evidence = native / "worker.json"
            native_command = [
                str(python),
                str(ROOT / "scripts/backbone_all400_worker.py"),
                "--source",
                str(source),
                "--access",
                str(ROOT / "outputs/quality/all_data/RGB-120-r2/access.json"),
                "--evidence",
                str(native_evidence),
                "--seed",
                "0",
                "--expected-init-policy",
                "same RGB130 original 303-view DLT initialization; all400 photometric training",
                "--",
                "--source_path",
                str(scene),
                "--model_path",
                str(native / "model"),
                "--resolution",
                "1",
                "--data_device",
                "cpu",
                "--iterations",
                str(args.iterations),
                "--densify_grad_threshold",
                "0.0002",
                "--densify_grad_abs_threshold",
                "0.0004",
                "--test_iterations",
                "-1",
                "--save_iterations",
                str(args.iterations),
                "--checkpoint_iterations",
                str(args.iterations),
                "--quiet",
            ]
            with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                applications = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,process_name,used_memory",
                        "--format=csv,noheader",
                    ],
                    text=True,
                )
                if any(
                    line.strip() and "ToDesk" not in line
                    for line in applications.splitlines()
                ):
                    raise RuntimeError("Other GPU compute process active")
                run_logged(native_command, source, environment, native / "train.log")
            native_runs.append(native)

        hooked = {}
        run_specs = (
            ("control_1", "control"),
            ("control_2", "control"),
            ("region_rgb", "region_rgb"),
            ("region_gradient", "region_gradient"),
        )
        for run_name, mode in run_specs:
            output = args.output / f"hooked-{run_name}"
            command = [
                str(python),
                str(ROOT / "scripts/run_standard_rgb_supervision.py"),
                "--artifact-root",
                str(args.artifact_root.resolve()),
                "--config",
                str(CONFIGS[mode]),
                "--output",
                str(output),
                "--iterations",
                str(args.iterations),
                "--preflight",
            ]
            run_logged(
                command, ROOT, environment, args.output / f"launch-{run_name}.log"
            )
            hooked[run_name] = output

        import torch
        from src.standard_rgb_supervision import AuxLossController

        native_checkpoints = [
            torch.load(
                output / f"model/chkpnt{args.iterations}.pth",
                map_location="cpu",
                weights_only=False,
            )
            for output in native_runs
        ]
        hooked_checkpoints = [
            torch.load(
                hooked[name] / f"model/chkpnt{args.iterations}.pth",
                map_location="cpu",
                weights_only=False,
            )
            for name in ("control_1", "control_2")
        ]
        native_repeat = tensor_deltas(native_checkpoints[0], native_checkpoints[1])
        hooked_repeat = tensor_deltas(hooked_checkpoints[0], hooked_checkpoints[1])
        native_hooked = tensor_deltas(native_checkpoints[0], hooked_checkpoints[0])
        if not native_repeat.keys() == hooked_repeat.keys() == native_hooked.keys():
            raise ValueError("Checkpoint tensor structure changed")
        envelope_violations = {
            path: {
                "native_hooked": native_hooked[path],
                "native_repeat": native_repeat[path],
                "hooked_repeat": hooked_repeat[path],
            }
            for path in native_hooked
            if native_hooked[path]
            > 2.0 * max(native_repeat[path], hooked_repeat[path]) + 1e-8
        }
        if envelope_violations:
            raise ValueError(
                "Hooked control exceeds CUDA repeatability envelope: "
                f"{list(envelope_violations)[:10]}"
            )

        control_controller = AuxLossController(configs["control"], args.output)
        base_loss = torch.tensor(1.0, requires_grad=True)
        if (
            control_controller.apply(0, None, None, None, base_loss, None)
            is not base_loss
        ):
            raise ValueError("Control auxiliary path is not an object-identity no-op")

        sequences = {
            run_name: read_camera_sequence(output / "camera_sequence.txt")
            for run_name, output in hooked.items()
        }
        if any(len(sequence) != args.iterations for sequence in sequences.values()):
            raise ValueError("Unexpected preflight camera-sequence length")
        if any(sequence != sequences["control_1"] for sequence in sequences.values()):
            raise ValueError("A/B/C camera sequences differ")

        summaries = {
            mode: check_gradient_summary(
                hooked[mode] / "aux_summary.json", args.iterations
            )
            for mode in ("region_rgb", "region_gradient")
        }
        result.update(
            status="passed",
            exact_checkpoint_identity_required=False,
            exact_checkpoint_identity_rejection=(
                "Independent native repeats are not bit-exact under the pinned CUDA rasterizer"
            ),
            control_loss_object_identity_noop=True,
            checkpoint_iteration=args.iterations,
            checkpoint_repeatability={
                "criterion": "native-hooked max-abs tensor deltas <= 2x max(native-native, hooked-hooked) + 1e-8",
                "tensor_count": len(native_hooked),
                "native_repeat_nonzero_tensors": sum(
                    value > 0 for value in native_repeat.values()
                ),
                "hooked_repeat_nonzero_tensors": sum(
                    value > 0 for value in hooked_repeat.values()
                ),
                "native_hooked_nonzero_tensors": sum(
                    value > 0 for value in native_hooked.values()
                ),
                "envelope_violations": envelope_violations,
                "native_repeat_maxima": native_repeat,
                "hooked_repeat_maxima": hooked_repeat,
                "native_hooked_maxima": native_hooked,
            },
            camera_sequences_exact=True,
            camera_sequence_sha256={
                run_name: sha256(output / "camera_sequence.txt")
                for run_name, output in hooked.items()
            },
            auxiliary_summaries=summaries,
            run_results={
                run_name: json.loads((output / "result.json").read_text())
                for run_name, output in hooked.items()
            },
            native_workers=[
                json.loads((output / "worker.json").read_text())
                for output in native_runs
            ],
        )
    except BaseException as error:
        result.update(
            status="failed_preserved", error=f"{type(error).__name__}: {error}"
        )
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(args.output / "result.json", result)
        print(json.dumps({"status": result["status"], "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
