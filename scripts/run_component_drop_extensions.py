#!/usr/bin/env python3
"""Finite, independently registered component-quota and Gaussian-dropout study."""

from __future__ import annotations

import argparse
import copy
import csv
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
import run_sem396_spatial as sem
import run_rgb152_spatial as rgb
from src.quality_protocol import sha256, verify_hashes, write_new_json

SUITE = "QUAL-014-COMPONENT-DROP-EXTENSIONS"
DIAG = "SEM-416-COMPONENT-POINT-DIAG"
NEW_CODE = [
    "scripts/run_component_drop_extensions.py",
    "scripts/worker_gaussian_dropout.py",
    "scripts/audit_component_drop_extensions.py",
    "src/component_point_sampling.py",
    "src/gaussian_training_dropout.py",
]


def read(path):
    return json.loads(Path(path).read_text())


def merge(a, b):
    result = copy.deepcopy(a)
    for k, v in b.items():
        result[k] = (
            merge(result[k], v)
            if isinstance(v, dict) and isinstance(result.get(k), dict)
            else v
        )
    return result


def config(eid):
    current = read(ROOT / "experiments/runs" / eid / "config.yaml")
    if "base_config" in current:
        current = merge(read(ROOT / current["base_config"]), current)
    return current


def out():
    return ROOT / config(SUITE)["output_path"]


def guard():
    verify_hashes(ROOT, read(out() / "binding.json")["files"])


def gpu_lock():
    lock = (ROOT / "outputs/quality/gpu.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    text = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ],
        text=True,
    )
    if any(line.strip() and "ToDesk" not in line for line in text.splitlines()):
        lock.close()
        raise RuntimeError("Other GPU compute process is active")
    return lock


def prepare():
    suite = config(SUITE)
    output = out()
    output.mkdir(parents=True, exist_ok=False)
    ids = (
        [SUITE, DIAG]
        + suite["semantic_pair"]
        + suite["rgb_pair"]
        + suite["rgb_semantic_pair"]
    )
    paths = set(NEW_CODE + list(sem.CODE_PATHS) + list(rgb.CODE_PATHS))
    paths.update(
        [
            suite["protocol"],
            "data/manifests/contest_dataset_v1.json",
            "experiments/splits/spatial_confirmation_v1.json",
            "experiments/splits/spatial_confirmation_v2.json",
        ]
    )
    resolved = {}
    for eid in ids:
        p = f"experiments/runs/{eid}/config.yaml"
        paths.add(p)
        c = read(ROOT / p)
        if "base_config" in c:
            paths.add(c["base_config"])
        resolved[eid] = config(eid)
    for eid in ["SEM-412", "SEM-414", "SEM-415"]:
        paths.add(f"outputs/quality/{eid}/r1/mask2former/result.json")
    for rel in [
        "RGB-154/full-r1",
        "RGB-155/full-r1",
        "RGB-152/full-r2",
        "rgb140/full-r2",
    ]:
        paths.add(f"outputs/quality/{rel}/result.json")
    paths.add("outputs/quality/SEM-378/r1/mask2former/result.json")
    for path in sorted(paths):
        if path.startswith(("scripts/", "src/", "experiments/runs/", "docs/")):
            target = output / "source" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / path, target)
    write_new_json(output / "resolved-configs.json", resolved)
    write_new_json(
        output / "binding.json",
        {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "files": {p: sha256(ROOT / p) for p in sorted(paths)},
            "created_unix": time.time(),
            "free_bytes": shutil.disk_usage(output).free,
        },
    )


def seed_everything(seed, torch, np):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def diagnostic(artifact_root):
    guard()
    c = config(DIAG)
    parent = read(ROOT / c["parent_config"])
    output = ROOT / c["output_path"]
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "starting",
        "experiment_id": DIAG,
        "snapshots": {},
        "holdout_accesses": 0,
    }
    begin = time.monotonic()
    try:
        with gpu_lock():
            runtime, torch, np = sem.setup_runtime(artifact_root)
            from PIL import Image
            from detectron2.structures import Instances
            from src.component_point_sampling import ComponentSampler
            from src.quality_mask2former import (
                config_for_model,
                load_official_model,
                class_masks,
            )
            from src.quality_seed_initialization import tensor_manifest
            from src.semantic_cable_binary import read_rgb

            plan = read(ROOT / c["sampling_plan"])
            fold = read(ROOT / parent["split_path"])["primary_fold"]
            allowed = sorted(fold["semantic_train_labeled"])
            selected = []
            for name in allowed:
                selected.extend(
                    [row for row in plan["samples"] if row["name"] == name][:2]
                )
            if len(selected) != 510:
                raise ValueError("Expected exactly two fixed crops per training image")
            write_new_json(output / "samples.json", selected)
            label_manifest = read(ROOT / parent["label_manifest"])
            hashes = {
                r["image"]: r["label_png_sha256"] for r in label_manifest["records"]
            }
            verify_hashes(ROOT / parent["label_root"], {n: hashes[n] for n in allowed})
            images = {
                n: read_rgb(ROOT / parent["rgb_evaluation"] / "images" / n)
                for n in allowed
            }
            labels = {
                n: np.asarray(Image.open(ROOT / parent["label_root"] / n))
                for n in allowed
            }
            parent_train = ROOT / parent["output_path"] / "mask2former"
            verify_hashes(parent_train, read(parent_train / "result.json")["artifacts"])
            torch.cuda.reset_peak_memory_stats()
            rows = []
            sampler = ComponentSampler(callback=rows.append)
            sampler.install()
            with (output / "coverage.jsonl").open("x") as journal:
                for snapshot in c["snapshots"]:
                    seed_everything(parent["seed"], torch, np)
                    mc = config_for_model(
                        runtime / "vendor/Mask2Former",
                        "configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml",
                        False,
                    )
                    model, _ = load_official_model(
                        mc, runtime / "weights/model_final_6b4a3a.pkl"
                    )
                    if snapshot:
                        model.load_state_dict(
                            torch.load(
                                parent_train / f"model-{snapshot}.pt",
                                map_location="cpu",
                                weights_only=True,
                            )["model"]
                        )
                    before = tensor_manifest(model.state_dict())
                    model.cuda().train()
                    accum = {"loss": {}, "matcher": {}}
                    low_views = set()
                    for index, sample in enumerate(selected):
                        name = sample["name"]
                        sampler.set_sample(sample, labels[name])
                        y, x = sample["top"], sample["left"]
                        image = images[name][y : y + 768, x : x + 768].copy()
                        target = labels[name][y : y + 768, x : x + 768].copy()
                        if sample["flip"]:
                            image, target = (
                                image[:, ::-1].copy(),
                                target[:, ::-1].copy(),
                            )
                        ids, masks = class_masks(target)
                        instances = Instances(target.shape)
                        instances.gt_classes, instances.gt_masks = (
                            torch.from_numpy(ids),
                            torch.from_numpy(masks),
                        )
                        torch.manual_seed(c["diagnostic_seed"] + index)
                        torch.cuda.manual_seed_all(c["diagnostic_seed"] + index)
                        rows.clear()
                        with (
                            torch.no_grad(),
                            torch.autocast("cuda", dtype=torch.float16),
                        ):
                            losses = model(
                                [
                                    {
                                        "image": torch.from_numpy(
                                            image.transpose(2, 0, 1).copy()
                                        ).float(),
                                        "instances": instances,
                                    }
                                ]
                            )
                        if any(not torch.isfinite(v).all() for v in losses.values()):
                            raise FloatingPointError("Nonfinite diagnostic loss")
                        for row in rows:
                            journal.write(
                                json.dumps({"snapshot": snapshot, "crop": index, **row})
                                + "\n"
                            )
                            key = f"layer-{row['layer']}"
                            group = accum[row["phase"]].setdefault(
                                key,
                                {"components": 0, "zero": 0, "below4": 0, "points": 0},
                            )
                            group["components"] += 1
                            group["zero"] += row["points"] == 0
                            group["below4"] += row["points"] < 4
                            group["points"] += row["points"]
                            if (
                                row["phase"] == "loss"
                                and row["layer"] == 0
                                and row["points"] < 4
                            ):
                                low_views.add(name)
                        if index % 50 == 0:
                            journal.flush()
                            print(
                                json.dumps({"snapshot": snapshot, "crops": index + 1}),
                                flush=True,
                            )
                    if before != tensor_manifest(model.state_dict()):
                        raise ValueError("Read-only diagnostic changed model state")
                    main = accum["loss"].get("layer-0", {"components": 0, "below4": 0})
                    gate = c["support_gate"]
                    supported = (
                        main["below4"]
                        >= gate["late_snapshot_min_low_coverage_components"]
                        and len(low_views)
                        >= gate["late_snapshot_min_low_coverage_views"]
                        and main["below4"] / max(main["components"], 1)
                        >= gate["late_snapshot_min_low_coverage_fraction"]
                    )
                    result["snapshots"][str(snapshot)] = {
                        "coverage": accum,
                        "low_coverage_views": sorted(low_views),
                        "supports_loss_quota": supported,
                    }
                    del model
                    torch.cuda.empty_cache()
            sampler.restore()
            result.update(
                status="completed",
                training_supported=any(
                    result["snapshots"][str(s)]["supports_loss_quota"]
                    for s in (3000, 6000)
                ),
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                training_label_count=len(allowed),
                coverage_sha256=sha256(output / "coverage.jsonl"),
                coordinate_hit_definition="component-specific bilinear weight >=0.5 at actual normalized sample coordinates",
            )
        guard()
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - begin
        write_new_json(output / "result.json", result)


def semantic(eid, stage, artifact_root):
    guard()
    c = config(eid)
    sem.EID, sem.CONFIG = eid, ROOT / "experiments/runs" / eid / "config.yaml"
    sem.CODE_PATHS = tuple(sorted(set(sem.CODE_PATHS) | set(NEW_CODE)))
    if stage == "prepare":
        sem.prepare(c)
    elif stage == "train":
        if (
            c.get("component_quota")
            and not read(ROOT / config(DIAG)["output_path"] / "result.json")[
                "training_supported"
            ]
        ):
            raise ValueError("Component diagnostic did not support training")
        original = sem.setup_runtime
        hook = None

        def setup(root):
            nonlocal hook
            runtime = original(root)
            if "component_quota" in c:
                from src.component_point_sampling import ComponentSampler

                hook = ComponentSampler(enabled=c["component_quota"])
                hook.install()
            return runtime

        sem.setup_runtime = setup
        try:
            sem.train(c, artifact_root)
        finally:
            sem.setup_runtime = original
            if hook:
                hook.restore()
                write_new_json(
                    ROOT / c["output_path"] / "quota-statistics.json", hook.stats
                )
    else:
        pair = config(SUITE)[
            "semantic_pair"
            if eid.startswith(("SEM-417", "SEM-418"))
            else "rgb_semantic_pair"
        ]
        for arm in pair:
            ac = config(arm)
            train = ROOT / ac["output_path"] / "mask2former"
            record = read(train / "result.json")
            if record["status"] != "completed_sealed_6000":
                raise ValueError("Both semantic endpoints must complete before scoring")
            verify_hashes(train, record["artifacts"])
        sem.inference(c, artifact_root)
    guard()


def rgb_stage(eid, stage, artifact_root):
    guard()
    c = config(eid)
    rgb.EID, rgb.CONFIG = eid, ROOT / "experiments/runs" / eid / "config.yaml"
    rgb.CODE_PATHS = tuple(sorted(set(rgb.CODE_PATHS) | set(NEW_CODE)))
    if stage == "preflight":
        rgb.preflight(c, artifact_root)
        return
    if stage == "evaluate":
        for arm in config(SUITE)["rgb_pair"]:
            record = read(ROOT / config(arm)["training"]["output"] / "result.json")
            if record["status"] != "completed_spatial_training":
                raise ValueError("Both RGB endpoints must be sealed before scoring")
        rgb.render_and_evaluate(c, artifact_root)
        return
    root = ROOT / c["output"]["root"]
    run = (
        ROOT / c["training"]["output"] if stage == "train" else root / "engineering-701"
    )
    run.mkdir(parents=True, exist_ok=False)
    source = ROOT / c["output"]["source"]
    environment, python = rgb.runtime_environment(artifact_root, source)
    steps = 30000 if stage == "train" else config(SUITE)["rgb_preflight_steps"]
    command = [
        str(python),
        str(ROOT / "scripts/worker_gaussian_dropout.py"),
        "--maximum",
        str(c["dropout_max"]),
        "--dropout-seed",
        str(c["dropout_seed"]),
        "--dropout-record",
        str(run / "dropout.json"),
        "--",
        "--source",
        str(source),
        "--access",
        str(root / "access.json"),
        "--evidence",
        str(run / "worker.json"),
        "--seed",
        str(c["seed"]),
        "--partition",
        c["split_id"],
        "--expected-train",
        "353",
        "--expected-evaluation",
        "30",
        "--expected-universe",
        "383",
        "--",
        "--source_path",
        str(ROOT / c["data"]["scene"]),
        "--model_path",
        str(run / "model"),
        "--resolution",
        "1",
        "--data_device",
        "cpu",
        "--iterations",
        str(steps),
        "--densify_grad_threshold",
        "0.0002",
        "--densify_grad_abs_threshold",
        "0.0004",
        "--test_iterations",
        "-1",
        "--save_iterations",
        str(steps),
        "--checkpoint_iterations",
        str(steps),
        "--quiet",
        "--eval",
    ]
    result = {
        "status": "starting",
        "experiment_id": eid,
        "command": command,
        "seed": c["seed"],
        "git_commit": c["git_commit"],
        "config_sha256": sha256(rgb.CONFIG),
        "split_sha256": sha256(ROOT / c["split_path"]),
        "initialization_ply_sha256": sha256(
            ROOT / c["data"]["scene"] / "sparse/0/points3D.ply"
        ),
        "source": rgb.source_manifest(source),
        "code_sha256": {p: sha256(ROOT / p) for p in rgb.CODE_PATHS},
    }
    begin = time.monotonic()
    try:
        with gpu_lock():
            with (
                (run / "train.log").open("x") as log,
                (run / "resources.jsonl").open("x") as monitor,
            ):
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                while process.poll() is None:
                    gpu = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=memory.used,utilization.gpu,power.draw",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                    )
                    monitor.write(
                        json.dumps(
                            {
                                "seconds": time.monotonic() - begin,
                                "gpu": gpu.stdout.strip(),
                            }
                        )
                        + "\n"
                    )
                    monitor.flush()
                    time.sleep(10)
            result["exit_status"] = process.returncode
            if process.returncode:
                raise RuntimeError(f"RGB worker failed {process.returncode}")
        worker = read(run / "worker.json")
        drop = read(run / "dropout.json")
        if (
            worker["status"] != "completed_native_execution"
            or drop["stats"]["training_renders"] != steps
        ):
            raise ValueError("Incomplete RGB execution")
        if stage == "engineering" and drop["stats"]["densification_events"] < 2:
            raise ValueError("Engineering probe did not exercise densification")
        result.update(
            status="completed_spatial_training"
            if stage == "train"
            else "passed_engineering_probe",
            worker=worker,
            dropout=drop,
            artifacts={
                str(p.relative_to(run)): sha256(p)
                for p in (run / "model").rglob("*")
                if p.is_file()
            },
        )
        guard()
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - begin
        write_new_json(run / "result.json", result)


def rgb_gate():
    suite = config(SUITE)
    a, b = [
        read(ROOT / config(eid)["training"]["output"] / "evaluation/result.json")
        for eid in suite["rgb_pair"]
    ]
    delta = {
        k: b["metrics"]["holdout"][k]["mean"] - a["metrics"]["holdout"][k]["mean"]
        for k in ("PSNR_dB", "SSIM", "LPIPS_Alex")
    }
    region = {
        k: b["group_metrics"][k]["PSNR_dB"]["mean"]
        - a["group_metrics"][k]["PSNR_dB"]["mean"]
        for k in a["group_metrics"]
    }
    passed = (
        delta["PSNR_dB"] >= 0
        and delta["LPIPS_Alex"] <= 0
        and (delta["PSNR_dB"] > 0 or delta["LPIPS_Alex"] < 0)
        and min(region.values()) >= -0.3
    )
    write_new_json(
        out() / "rgb-gate.json",
        {"passed": passed, "delta": delta, "region_psnr_delta": region},
    )


def run(artifact_root):
    prepare()
    suite = config(SUITE)
    result = {"status": "running", "stages": []}
    begin = time.monotonic()

    def execute(name, stage, eid=None):
        guard()
        command = [
            sys.executable,
            str(Path(__file__)),
            "--artifact-root",
            str(artifact_root),
            "--stage",
            stage,
        ]
        if eid:
            command += ["--experiment", eid]
        start = time.monotonic()
        with (out() / f"{name}.log").open("x") as log:
            proc = subprocess.run(
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
        record = {
            "name": name,
            "command": command,
            "seconds": time.monotonic() - start,
            "exit_status": proc.returncode,
        }
        write_new_json(out() / f"{name}-execution.json", record)
        result["stages"].append(record)
        print(json.dumps(record), flush=True)
        if proc.returncode:
            raise RuntimeError(f"{name} failed; preserved")

    try:
        execute("diagnostic", "diagnostic")
        if read(ROOT / config(DIAG)["output_path"] / "result.json")[
            "training_supported"
        ]:
            execute("semantic-preflight", "semantic-preflight")
            for stage in ("prepare", "train", "inference"):
                for eid in suite["semantic_pair"]:
                    execute(f"{eid}-{stage}", "semantic-" + stage, eid)
        else:
            result["A_training"] = "not_run_diagnostic_gate_failed"
        execute("dropout-unit-preflight", "dropout-preflight")
        for stage in ("preflight", "engineering", "train", "evaluate"):
            for eid in suite["rgb_pair"]:
                execute(f"{eid}-{stage}", "rgb-" + stage, eid)
        execute("rgb-gate", "rgb-gate")
        if read(out() / "rgb-gate.json")["passed"]:
            for stage in ("prepare", "train", "inference"):
                for eid in suite["rgb_semantic_pair"]:
                    execute(f"{eid}-{stage}", "semantic-" + stage, eid)
        else:
            result["B_semantic_training"] = "not_run_rgb_gate_failed"
        execute("audit", "audit")
        result["status"] = "completed_both_lines"
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - begin
        write_new_json(out() / "execution-result.json", result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--experiment")
    a = p.parse_args()
    artifact_root = a.artifact_root.resolve()
    if a.stage == "run":
        run(artifact_root)
    elif a.stage == "diagnostic":
        diagnostic(artifact_root)
    elif a.stage in ("semantic-preflight", "dropout-preflight", "audit"):
        from audit_component_drop_extensions import (
            preflight_semantic,
            preflight_dropout,
            audit,
        )

        {
            "semantic-preflight": preflight_semantic,
            "dropout-preflight": preflight_dropout,
            "audit": audit,
        }[a.stage](artifact_root)
    elif a.stage.startswith("semantic-"):
        semantic(a.experiment, a.stage.split("-", 1)[1], artifact_root)
    elif a.stage == "rgb-gate":
        rgb_gate()
    elif a.stage.startswith("rgb-"):
        rgb_stage(a.experiment, a.stage.split("-", 1)[1], artifact_root)
    else:
        raise ValueError("Unknown stage")


if __name__ == "__main__":
    main()
