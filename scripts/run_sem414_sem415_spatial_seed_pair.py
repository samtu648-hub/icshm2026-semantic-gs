#!/usr/bin/env python3
"""Execute one immutable SEM414/415 pair, sealing both endpoints before scoring."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_sem396_spatial as base
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.spatial_seed_replication import (
    PAIR_CONFIG,
    validate_pair_configs,
    validate_pair_endpoints,
)

EXTRA_CODE = (
    "scripts/run_sem414_sem415_spatial_seed_pair.py",
    "scripts/run_sem414_rgb154_spatial_seed20260919.py",
    "scripts/run_sem415_rgb155_spatial_seed20260919.py",
    "scripts/audit_qual013_spatial_v2_seed.py",
    "src/spatial_seed_replication.py",
    "src/quality_protocol.py",
    "src/quality_experts.py",
    "src/semantic_cable_binary.py",
    "src/semantic2d.py",
    "src/quality_cameras.py",
    "src/quality_native_gaussian.py",
)


def read(path):
    return json.loads(Path(path).read_text())


def context():
    pair = read(ROOT / PAIR_CONFIG)
    arms = {k: read(ROOT / v["config"]) for k, v in pair["arms"].items()}
    split = read(ROOT / pair["split_path"])
    validate_pair_configs(arms, split)
    return pair, arms, split, ROOT / pair["pair_output_path"]


def guard():
    pair, arms, split, output = context()
    verify_hashes(ROOT, pair["frozen_inputs"])
    verify_hashes(ROOT, read(output / "binding.json")["files"])
    return pair, arms, split, output


def gpu_lock():
    lock = (ROOT / "outputs/quality/gpu.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
    )
    if any(line.strip() and "ToDesk" not in line for line in processes.splitlines()):
        lock.close()
        raise RuntimeError("Another GPU compute process is active")
    return lock


def prepare_pair():
    pair, _, _, output = context()
    verify_hashes(ROOT, pair["frozen_inputs"])
    output.mkdir(parents=True, exist_ok=False)
    paths = sorted(set(base.CODE_PATHS) | set(EXTRA_CODE))
    paths += [
        PAIR_CONFIG,
        pair["protocol"],
        "experiments/runs/SEM-370-ALL300/config.yaml",
    ]
    paths += [v["config"] for v in pair["arms"].values()]
    files = {p: sha256(ROOT / p) for p in paths}
    for relative in paths:
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    write_new_json(
        output / "binding.json",
        {
            "schema_version": 1,
            "experiment_id": pair["experiment_id"],
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "files": files,
            "source_snapshot": "source",
            "frozen_inputs": pair["frozen_inputs"],
            "created_unix": time.time(),
            "free_bytes": shutil.disk_usage(output).free,
        },
    )


def prepared_plans(arms):
    plans = {
        k: read(ROOT / c["output_path"] / "sampling-plan.json") for k, c in arms.items()
    }
    if plans["absgs"]["samples"] != plans["standard"]["samples"]:
        raise ValueError("Paired sample plans differ")
    if any(p["seed"] != 20260919 or len(p["samples"]) != 6000 for p in plans.values()):
        raise ValueError("Unexpected seed or sample count")
    return plans


def preflight(artifact_root):
    pair, arms, split, output = guard()
    plans = prepared_plans(arms)
    destination = output / "preflight"
    destination.mkdir(exist_ok=False)
    result = {
        "status": "starting",
        "arms": {},
        "holdout_label_accesses": 0,
        "holdout_rgb_accesses": 0,
    }
    started = time.monotonic()
    try:
        with gpu_lock():
            runtime, torch, np = base.setup_runtime(artifact_root)
            from src.quality_mask2former import (
                config_for_model,
                load_official_model,
                author_optimizer,
                class_masks,
            )
            from src.quality_mask2former_training import (
                sampled_crop,
                amp_update,
                poly_factor,
            )
            from src.quality_seed_initialization import (
                tensor_manifest,
                validate_same_seed_pair,
                validate_cross_seed_reference,
            )
            from src.quality_static_training import TrainingLabels
            from src.semantic_cable_binary import read_rgb
            from detectron2.structures import Instances  # type: ignore[import-not-found]

            torch.set_num_threads(4)
            reference = read(
                ROOT
                / arms["absgs"]["seed_aware_initialization"][
                    "training_initialization_reference"
                ]
            )
            manifests = {}
            for arm, cfg in arms.items():
                random.seed(cfg["seed"])
                np.random.seed(cfg["seed"])
                torch.manual_seed(cfg["seed"])
                torch.cuda.manual_seed_all(cfg["seed"])
                torch.backends.cudnn.benchmark = False
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                c = config_for_model(
                    runtime / "vendor/Mask2Former",
                    "configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml",
                    False,
                )
                c.defrost()
                c.DATASETS.TRAIN = ("sem341_rendered_training",)
                c.SOLVER.MAX_ITER = 6000
                c.SOLVER.POLY_LR_POWER = 0.9
                c.freeze()
                if c.dump() != (base.SEM378_TRAIN / "resolved_model.yaml").read_text():
                    raise ValueError("Preflight model configuration drift")
                model, loaded = load_official_model(
                    c, runtime / "weights/model_final_6b4a3a.pkl"
                )
                state = tensor_manifest(model.state_dict())
                validation = validate_same_seed_pair(
                    reference["current_seed_state_manifest"],
                    state,
                    reference["current_seed_pretrained_load"],
                    loaded,
                )
                cross = validate_cross_seed_reference(
                    state,
                    reference["reference_seed_state_manifest"],
                    loaded,
                    reference["reference_seed_pretrained_load"],
                )
                if validation["tensor_count"] != 650:
                    raise ValueError("Unexpected initial state tensor count")
                manifests[arm] = state
                write_new_json(
                    destination / f"{arm}-initialization.json",
                    {
                        "state_manifest": state,
                        "pretrained_load": loaded,
                        "same_seed": validation,
                        "cross_seed": cross,
                    },
                )
                torch.cuda.reset_peak_memory_stats()
                model.cuda()
                optimizer = author_optimizer(model, c)
                rates = [g["lr"] for g in optimizer.param_groups]
                scaler = torch.amp.GradScaler("cuda", init_scale=1024)
                names = sorted(split["primary_fold"]["semantic_train_labeled"])
                wanted = {
                    row["name"]
                    for row in plans[arm]["samples"][
                        : pair["preflight_updates_per_arm"]
                    ]
                }
                labels = TrainingLabels(
                    ROOT / cfg["label_root"],
                    names,
                    read(ROOT / cfg["label_manifest"]),
                    [1320, 989],
                )
                images = {
                    n: read_rgb(ROOT / cfg["output_path"] / "export/images" / n)
                    for n in wanted
                }
                targets = {n: labels.read(n) for n in wanted}
                rng = random.Random(cfg["seed"])
                model.train()
                rows = []
                for step in range(pair["preflight_updates_per_arm"]):
                    x, y, sample = sampled_crop(rng, names, images, targets, 768)
                    expected = {
                        k: plans[arm]["samples"][step][k]
                        for k in ("name", "top", "left", "flip")
                    }
                    if sample != expected:
                        raise ValueError("Preflight native sampler drift")
                    ids, masks = class_masks(y)
                    instances = Instances(tuple(y.shape))
                    instances.gt_classes = torch.from_numpy(ids)
                    instances.gt_masks = torch.from_numpy(masks)
                    for group, rate in zip(optimizer.param_groups, rates):
                        group["lr"] = rate * poly_factor(step)
                    row = amp_update(
                        model,
                        optimizer,
                        scaler,
                        torch.from_numpy(x.transpose(2, 0, 1).copy()).float(),
                        instances,
                        max_attempts=12,
                        clip=0.01,
                    )
                    rows.append({"step": step + 1, "sample": sample, **row})
                result["arms"][arm] = {
                    "same_seed": validation,
                    "cross_seed": cross,
                    "updates": rows,
                    "training_labels_accessed": sorted(labels.accessed),
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                    "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                }
                del model, optimizer, scaler, images, targets, labels
                torch.cuda.empty_cache()
            if manifests["absgs"] != manifests["standard"]:
                raise ValueError("Preflight arm initial states differ")
        guard()
        result["status"] = "passed_discarded_eight_update_pair"
    except BaseException as error:
        result.update(
            status="failed_preserved", error=f"{type(error).__name__}: {error}"
        )
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(destination / "result.json", result)


def rgb_check(artifact_root, arm, phase):
    pair, arms, split, output = guard()
    cfg, settings = arms[arm], pair["arms"][arm]
    names = pair["rgb_replay_training_views"]
    if not set(names) <= set(split["primary_fold"]["train_rgb_400"]):
        raise ValueError("RGB replay camera is not a training camera")
    run = ROOT / cfg["rgb_run"]
    parent = read(run / "result.json")
    verify_hashes(run, parent["artifacts"])
    source = (
        ROOT if settings["rgb_source_base"] == "project" else artifact_root
    ) / settings["rgb_source_root"]
    verify_hashes(source, parent["source"])
    result = {
        "status": "starting",
        "arm": arm,
        "phase": phase,
        "views": {},
        "holdout_accesses": 0,
    }
    started = time.monotonic()
    try:
        with gpu_lock():
            _, torch, np = base.setup_runtime(artifact_root)
            sys.path[:0] = [
                str(source),
                str(source / "submodules/diff-gaussian-rasterization"),
                str(source / "submodules/simple-knn"),
            ]
            from gaussian_renderer import GaussianModel, render  # type: ignore[import-not-found]
            from arguments import PipelineParams  # type: ignore[import-not-found]
            from PIL import Image
            from src.quality_cameras import read_camera_requests
            from src.quality_native_gaussian import make_native_view

            torch.set_num_threads(4)
            model = GaussianModel(3)
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            model.load_ply(
                str(run / "model/point_cloud/iteration_30000/point_cloud.ply")
            )
            parser = argparse.ArgumentParser()
            pipeline = PipelineParams(parser).extract(parser.parse_args([]))
            background = torch.zeros(3, device="cuda")
            cameras = read_camera_requests(
                ROOT / "outputs/quality/RGB-154/scene/sparse/0"
            )
            with torch.inference_mode():
                for name in names:
                    rgb = render(
                        make_native_view(cameras[name]), model, pipeline, background
                    )["render"]
                    actual = np.rint(
                        rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
                    ).astype(np.uint8)
                    expected = np.asarray(
                        Image.open(
                            ROOT / cfg["rgb_evaluation"] / "images" / name
                        ).convert("RGB")
                    )
                    maximum = int(
                        np.abs(
                            actual.astype(np.int16) - expected.astype(np.int16)
                        ).max()
                    )
                    if maximum:
                        raise ValueError(
                            f"Frozen RGB replay differs: {arm} {name}, max={maximum}"
                        )
                    result["views"][name] = {
                        "max_pixel_difference": maximum,
                        "exact": True,
                    }
        verify_hashes(run, parent["artifacts"])
        guard()
        result["status"] = "passed_exact_rgb_replay"
    except BaseException as error:
        result.update(
            status="failed_preserved", error=f"{type(error).__name__}: {error}"
        )
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(output / f"rgb-{phase}-{arm}.json", result)


def seal_pair():
    _, arms, _, output = guard()
    prepared_plans(arms)
    records = {
        k: read(ROOT / c["output_path"] / "mask2former/result.json")
        for k, c in arms.items()
    }
    validate_pair_endpoints(records["absgs"], records["standard"])
    traces, checkpoints, training_hashes = {}, {}, {}
    for arm, cfg in arms.items():
        train = ROOT / cfg["output_path"] / "mask2former"
        verify_hashes(train, records[arm]["artifacts"])
        traces[arm] = [
            json.loads(line)["sample"]
            for line in (train / "training.jsonl").read_text().splitlines()
        ]
        checkpoints[arm] = sha256(train / "model-6000.pt")
        training_hashes[arm] = sha256(train / "result.json")
    if traces["absgs"] != traces["standard"] or len(traces["absgs"]) != 6000:
        raise ValueError("Actual successful-update traces are not paired")
    write_new_json(
        output / "pair-seal.json",
        {
            "status": "both_endpoints_sealed",
            "sealed_unix": time.time(),
            "checkpoint_hashes": checkpoints,
            "training_result_hashes": training_hashes,
            "initial_state_sha256": records["absgs"]["initial_model_state_sha256"],
            "actual_6000_samples_exact": True,
        },
    )


def require_seal(arms, output):
    seal = read(output / "pair-seal.json")
    if seal["status"] != "both_endpoints_sealed":
        raise ValueError("Missing pair seal")
    for arm, cfg in arms.items():
        train = ROOT / cfg["output_path"] / "mask2former"
        if (
            sha256(train / "model-6000.pt") != seal["checkpoint_hashes"][arm]
            or sha256(train / "result.json") != seal["training_result_hashes"][arm]
        ):
            raise ValueError("A sealed endpoint changed")


def arm_main(arm):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("prepare", "train", "inference"), required=True
    )
    args = parser.parse_args()
    pair, arms, _, output = guard()
    cfg = arms[arm]
    base.EID = cfg["experiment_id"]
    base.CONFIG = ROOT / pair["arms"][arm]["config"]
    base.CODE_PATHS = tuple(sorted(set(base.CODE_PATHS) | set(EXTRA_CODE)))
    if args.stage == "prepare":
        base.prepare(cfg)
    elif args.stage == "train":
        if (
            read(output / "preflight/result.json")["status"]
            != "passed_discarded_eight_update_pair"
        ):
            raise ValueError("Paired preflight has not passed")
        for name in arms:
            if (
                read(output / f"rgb-before-{name}.json")["status"]
                != "passed_exact_rgb_replay"
            ):
                raise ValueError("Missing pre-training RGB replay")
        base.train(cfg, args.artifact_root.resolve())
    else:
        require_seal(arms, output)
        base.inference(cfg, args.artifact_root.resolve())
    guard()


def run_pair(artifact_root):
    prepare_pair()
    pair, _, _, output = guard()
    started = time.monotonic()
    result = {"status": "running", "stages": [], "started_unix": time.time()}
    python = sys.executable
    driver = str(Path(__file__))

    def execute(name, command):
        guard()
        row = {"name": name, "command": command, "started_unix": time.time()}
        begin = time.monotonic()
        with (output / f"{name}.log").open("x") as log:
            process = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )
        row.update(
            exit_status=process.returncode,
            seconds=time.monotonic() - begin,
            finished_unix=time.time(),
        )
        write_new_json(output / f"{name}-execution.json", row)
        result["stages"].append(row)
        print(json.dumps(row), flush=True)
        if process.returncode:
            raise RuntimeError(
                f"Stage {name} exited {process.returncode}; preserved without automatic retry"
            )

    common = ["--artifact-root", str(artifact_root)]
    try:
        for arm, settings in pair["arms"].items():
            execute(
                f"prepare-{arm}",
                [python, str(ROOT / settings["script"]), *common, "--stage", "prepare"],
            )
        execute("preflight", [python, driver, *common, "--stage", "preflight"])
        for arm in pair["arms"]:
            execute(
                f"rgb-before-{arm}",
                [
                    python,
                    driver,
                    *common,
                    "--stage",
                    "rgb-check",
                    "--arm",
                    arm,
                    "--phase",
                    "before",
                ],
            )
        for arm, settings in pair["arms"].items():
            execute(
                f"train-{arm}",
                [python, str(ROOT / settings["script"]), *common, "--stage", "train"],
            )
        execute("seal", [python, driver, *common, "--stage", "seal"])
        for arm, settings in pair["arms"].items():
            execute(
                f"inference-{arm}",
                [
                    python,
                    str(ROOT / settings["script"]),
                    *common,
                    "--stage",
                    "inference",
                ],
            )
        for arm in pair["arms"]:
            execute(
                f"rgb-after-{arm}",
                [
                    python,
                    driver,
                    *common,
                    "--stage",
                    "rgb-check",
                    "--arm",
                    arm,
                    "--phase",
                    "after",
                ],
            )
        execute(
            "audit", [python, str(ROOT / "scripts/audit_qual013_spatial_v2_seed.py")]
        )
        result["status"] = "completed_pair_and_independent_audit"
    except BaseException as error:
        result.update(
            status="failed_preserved", error=f"{type(error).__name__}: {error}"
        )
        raise
    finally:
        result.update(seconds=time.monotonic() - started, finished_unix=time.time())
        write_new_json(output / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("run", "preflight", "rgb-check", "seal"), required=True
    )
    parser.add_argument("--arm", choices=("absgs", "standard"))
    parser.add_argument("--phase", choices=("before", "after"))
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()
    if args.stage == "run":
        run_pair(artifact_root)
    elif args.stage == "preflight":
        preflight(artifact_root)
    elif args.stage == "seal":
        seal_pair()
    else:
        if args.arm is None or args.phase is None:
            parser.error("RGB check requires --arm and --phase")
        rgb_check(artifact_root, args.arm, args.phase)


if __name__ == "__main__":
    main()
