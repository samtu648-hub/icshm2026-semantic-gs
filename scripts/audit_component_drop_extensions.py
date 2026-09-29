#!/usr/bin/env python3
"""Engineering gates and independent saved-output audits for QUAL014."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from run_component_drop_extensions import (
    config,
    guard,
    gpu_lock,
    out,
    read,
    seed_everything,
    SUITE,
    DIAG,
)
from src.quality_protocol import sha256, verify_hashes, write_new_json


def model_config(runtime):
    from src.quality_mask2former import config_for_model

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
    return c


def preflight_semantic(artifact_root):
    import run_sem396_spatial as base

    guard()
    output = out() / "semantic-preflight"
    output.mkdir(exist_ok=False)
    record = {"status": "starting", "arms": {}}
    start = time.monotonic()
    try:
        with gpu_lock():
            runtime, torch, np = base.setup_runtime(artifact_root)
            from PIL import Image
            from types import SimpleNamespace
            from detectron2.structures import Instances
            from src.quality_mask2former import (
                load_official_model,
                author_optimizer,
                class_masks,
            )
            from src.quality_mask2former_training import amp_update, poly_factor
            from src.quality_seed_initialization import tensor_manifest
            from src.component_point_sampling import ComponentSampler

            c = config(config(SUITE)["semantic_pair"][0])
            parent = config(DIAG)
            samples = read(ROOT / parent["sampling_plan"])["samples"][:8]
            initialization = None
            for enabled in (False, True):
                seed_everything(c["seed"], torch, np)
                mc = model_config(runtime)
                model, _ = load_official_model(
                    mc, runtime / "weights/model_final_6b4a3a.pkl"
                )
                manifest = tensor_manifest(model.state_dict())
                if initialization is None:
                    initialization = manifest
                elif initialization != manifest:
                    raise ValueError("Initial states differ")
                model.cuda().train()
                optimizer = author_optimizer(model, mc)
                rates = [g["lr"] for g in optimizer.param_groups]
                scaler = torch.amp.GradScaler("cuda", init_scale=1024)
                sampler = ComponentSampler(enabled=enabled)
                sampler.install()
                torch.cuda.reset_peak_memory_stats()
                updates = []
                for step, sample in enumerate(samples):
                    image = np.asarray(
                        Image.open(
                            ROOT / c["rgb_evaluation"] / "images" / sample["name"]
                        ).convert("RGB")
                    )
                    full = np.asarray(
                        Image.open(ROOT / c["label_root"] / sample["name"])
                    )
                    sampler.set_sample(sample, full)
                    y, x = sample["top"], sample["left"]
                    image, target = (
                        image[y : y + 768, x : x + 768].copy(),
                        full[y : y + 768, x : x + 768].copy(),
                    )
                    if sample["flip"]:
                        image, target = image[:, ::-1].copy(), target[:, ::-1].copy()
                    ids, masks = class_masks(target)
                    inst = Instances(target.shape)
                    inst.gt_classes, inst.gt_masks = (
                        torch.from_numpy(ids),
                        torch.from_numpy(masks),
                    )
                    for g, lr in zip(optimizer.param_groups, rates):
                        g["lr"] = lr * poly_factor(step)
                    updates.append(
                        amp_update(
                            model,
                            optimizer,
                            scaler,
                            torch.from_numpy(image.transpose(2, 0, 1).copy()).float(),
                            inst,
                            max_attempts=12,
                            clip=0.01,
                        )
                    )
                sampler.restore()
                record["arms"][str(enabled)] = {
                    "updates": len(updates),
                    "skips": sum(len(v["skips"]) for v in updates),
                    "statistics": sampler.stats,
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                    "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                }
                if enabled and sampler.stats["replaced_points"] == 0:
                    raise ValueError("No quota point replacement in preflight")
                del model, optimizer, scaler
                torch.cuda.empty_cache()
            record["status"] = "passed_eight_update_pair"
        guard()
    except BaseException as exc:
        record.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["seconds"] = time.monotonic() - start
        write_new_json(output / "result.json", record)


def preflight_dropout(artifact_root):
    guard()
    output = out() / "dropout-preflight"
    output.mkdir(exist_ok=False)
    record = {"status": "starting"}
    start = time.monotonic()
    try:
        with gpu_lock():
            import run_sem396_spatial as base

            _, torch, np = base.setup_runtime(artifact_root)
            from gaussian_renderer import GaussianModel, render
            from arguments import PipelineParams
            from src.quality_native_gaussian import make_native_view
            from src.quality_cameras import read_camera_requests
            from src.gaussian_training_dropout import GaussianDropout

            seed_everything(0, torch, np)
            model = GaussianModel(3)
            model.load_ply(
                str(
                    ROOT
                    / "outputs/quality/RGB-152/full-r2/model/point_cloud/iteration_30000/point_cloud.ply"
                )
            )
            pp = argparse.ArgumentParser()
            pipeline = PipelineParams(pp).extract(pp.parse_args([]))
            camera = read_camera_requests(
                ROOT / "outputs/quality/RGB-152/scene/sparse/0"
            )["001.png"]
            view = make_native_view(camera)
            view.image_name = "001.png"
            background = torch.zeros(3, device="cuda")
            native = render(view, model, pipeline, background)["render"]
            loss = native.mean()
            loss.backward()
            gradients = {
                k: getattr(model, k).grad.detach().clone()
                for k in ("_xyz", "_opacity", "_features_dc", "_scaling")
            }
            for p in vars(model).values():
                if isinstance(p, torch.Tensor) and p.grad is not None and p.is_leaf:
                    p.grad = None
            disabled = GaussianDropout(0)
            disabled.install()
            disabled.step = 1000
            cpu_state, gpu_state = torch.get_rng_state(), torch.cuda.get_rng_state()
            replay = disabled.render(view, model, pipeline, background)["render"]
            if not torch.equal(replay, native):
                raise ValueError("Zero-drop RGB differs")
            replay.mean().backward()
            gradient_max = max(
                float((getattr(model, k).grad - gradients[k]).abs().max())
                for k in gradients
            )
            if any(
                not torch.allclose(
                    getattr(model, k).grad, gradients[k], atol=5e-5, rtol=0.002
                )
                for k in gradients
            ):
                raise ValueError("Zero-drop gradients differ")
            if not torch.equal(cpu_state, torch.get_rng_state()) or not torch.equal(
                gpu_state, torch.cuda.get_rng_state()
            ):
                raise ValueError("Disabled hook changed RNG")
            disabled.restore()
            hook = GaussianDropout(0.1)
            hook.install()
            hook.step = 15000
            cpu_state, gpu_state = torch.get_rng_state(), torch.cuda.get_rng_state()
            # The private mask stream must replay identically at the same seed.
            hooked = hook.render(view, model, pipeline, background)
            first = hooked["render"].detach().clone()
            hooked["render"].mean().backward()
            if any(not torch.isfinite(getattr(model, k).grad).all() for k in gradients):
                raise ValueError("Nonfinite dropout gradient")
            hook.generator.manual_seed(157)
            second = hook.render(view, model, pipeline, background)
            if not torch.equal(first, second["render"]):
                raise ValueError("Private-mask replay differs")
            if not torch.equal(cpu_state, torch.get_rng_state()) or not torch.equal(
                gpu_state, torch.cuda.get_rng_state()
            ):
                raise ValueError("Dropout changed global RNG")
            if hasattr(model, "_temporary_dropout_factor"):
                raise ValueError("Temporary opacity state leaked")
            with torch.no_grad():
                restored = hook.render(view, model, pipeline, background)["render"]
            if not torch.equal(native, restored):
                raise ValueError("Evaluation RGB changed")
            hook.restore()
            record.update(
                status="passed",
                zero_rate_rgb_exact=True,
                zero_rate_gradient_max_diff=gradient_max,
                global_rng_exact=True,
                fixed_mask_rgb_exact=True,
                normal_evaluation_exact=True,
                dropout_stats=hook.stats,
                dropped_radii_zero=True,
            )
        guard()
    except BaseException as exc:
        record.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["seconds"] = time.monotonic() - start
        write_new_json(output / "result.json", record)


def recount_extension(c, split, saved):
    import numpy as np
    from PIL import Image
    from scipy.ndimage import distance_transform_edt
    from audit_qual013_spatial_v2_seed import inner_boundary, compare_metrics
    from src.spatial_seed_replication import confusion_metrics
    from src.component_point_sampling import full_small_components

    fold = split["primary_fold"]
    names = set(fold["semantic_train_labeled"]) | set(
        fold["semantic_validation_labeled"]
    )
    hold = set(fold["semantic_validation_labeled"])
    groups = {n: g["group_id"] for g in split["groups"] for n in g["images"]}
    manifest = {
        r["image"]: r["label_png_sha256"]
        for r in read(ROOT / c["label_manifest"])["records"]
    }
    label_root = ROOT / c["label_root"]
    pred_root = ROOT / c["output_path"] / "inference/predictions"
    verify_hashes(label_root, {n: manifest[n] for n in names})
    verify_hashes(pred_root, saved["prediction_hashes"])
    views = {r["name"]: r for r in saved["per_view"]}
    if set(views) != names:
        raise ValueError("Label universe changed")
    totals = {k: np.zeros((5, 5), np.int64) for k in saved["metrics"]}
    boundaries = {k: np.zeros(4, np.int64) for k in totals}
    small = {
        str(k): {"components": 0, "pixels": 0, "tp": 0, "zero_detected": 0}
        for k in range(1, 5)
    }
    for index, n in enumerate(sorted(names)):
        t = np.asarray(Image.open(label_root / n))
        p = np.asarray(Image.open(pred_root / n))
        if (
            t.shape != (989, 1320)
            or p.shape != t.shape
            or p.dtype != np.uint8
            or p.max() > 4
        ):
            raise ValueError("Invalid prediction grid")
        matrix = np.zeros((5, 5), np.int64)
        np.add.at(matrix, (t.ravel(), p.ravel()), 1)
        if matrix.tolist() != views[n]["confusion"]:
            raise ValueError("Independent confusion differs")
        compare_metrics(confusion_metrics(matrix.tolist()), views[n])
        te, pe = inner_boundary(t == 2), inner_boundary(p == 2)
        counts = np.array(
            [
                np.count_nonzero(pe & (np.asarray(distance_transform_edt(~te)) <= 2)),
                pe.sum(),
                np.count_nonzero(te & (np.asarray(distance_transform_edt(~pe)) <= 2)),
                te.sum(),
            ],
            np.int64,
        )
        roles = ["holdout", groups[n]] if n in hold else ["train"]
        for role in roles:
            totals[role] += matrix
            boundaries[role] += counts
        if n in hold:
            for component in full_small_components(t):
                yx = component["yx"]
                k = component["class_id"]
                tp = int(np.count_nonzero(p[yx[:, 0], yx[:, 1]] == k))
                row = small[str(k)]
                row["components"] += 1
                row["pixels"] += len(yx)
                row["tp"] += tp
                row["zero_detected"] += tp == 0
        if index % 75 == 0:
            print(
                json.dumps({"audit": c["experiment_id"], "views": index + 1}),
                flush=True,
            )
    for role in totals:
        if (
            totals[role].tolist() != saved["metrics"][role]["confusion"]
            or boundaries[role].tolist() != saved["boundary_metrics"][role]["counts"]
        ):
            raise ValueError("Independent pooled or boundary counts differ")
        compare_metrics(
            confusion_metrics(totals[role].tolist()), saved["metrics"][role]
        )
    return {
        "views_recounted": len(names),
        "prediction_hashes_verified": len(saved["prediction_hashes"]),
        "confusion_and_boundary_exact": True,
        "small_component_holdout": small,
    }


def audit(artifact_root):
    guard()
    from src.spatial_seed_replication import flatten

    suite = config(SUITE)
    record = {
        "status": "completed",
        "semantic": {},
        "rgb": {},
        "scope": "historically exposed single-seed development comparisons",
    }
    start = time.monotonic()
    for label, pair in [
        ("A", suite["semantic_pair"]),
        ("B", suite["rgb_semantic_pair"]),
    ]:
        if not all(
            (ROOT / config(e)["output_path"] / "inference/result.json").is_file()
            for e in pair
        ):
            record["semantic"][label] = {"status": "not_run_parent_gate_failed"}
            continue
        raw = []
        audits = []
        for eid in pair:
            c = config(eid)
            saved = read(ROOT / c["output_path"] / "inference/result.json")
            split = read(ROOT / c["split_path"])
            audits.append(recount_extension(c, split, saved))
            raw.append(saved)
        a, b = raw
        roles = list(a["metrics"])
        delta = {
            r: {k: flatten(b, r)[k] - flatten(a, r)[k] for k in flatten(a, r)}
            for r in roles
        }
        classes = {
            n: b["metrics"]["holdout"]["per_class"][n]["IoU"]
            - a["metrics"]["holdout"]["per_class"][n]["IoU"]
            for n in ("deck", "stay_cable", "tower", "foundation")
        }
        d = delta["holdout"]
        passed = (
            d["foreground_miou"] > 0
            and d["boundary_f1"] >= 0
            and d["stay_cable_precision"] >= -0.001
            and min(classes.values()) >= -0.005
        )
        record["semantic"][label] = {
            "status": "completed",
            "independent_audits": audits,
            "delta": delta,
            "class_iou_delta": classes,
            "retention_gate_passed": passed,
            "control": {r: flatten(a, r) for r in roles},
            "candidate": {r: flatten(b, r) for r in roles},
        }
    ar, br = [
        read(ROOT / config(e)["training"]["output"] / "evaluation/result.json")
        for e in suite["rgb_pair"]
    ]
    for eid in suite["rgb_pair"]:
        c = config(eid)
        run = ROOT / c["training"]["output"]
        r = read(run / "result.json")
        verify_hashes(run, r["artifacts"])
        for role in ("train", "holdout"):
            with (run / f"evaluation/{role}.csv").open() as handle:
                import csv

                rows = list(csv.DictReader(handle))
            for k in ("PSNR_dB", "SSIM", "LPIPS_Alex"):
                mean = sum(float(row[k]) for row in rows) / len(rows)
                if (
                    abs(
                        mean
                        - read(run / "evaluation/result.json")["metrics"][role][k][
                            "mean"
                        ]
                    )
                    > 1e-10
                ):
                    raise ValueError("RGB CSV summary differs")
    drops = [
        read(ROOT / config(e)["training"]["output"] / "dropout.json")
        for e in suite["rgb_pair"]
    ]
    if drops[0]["camera_trace"] != drops[1]["camera_trace"]:
        raise ValueError("RGB camera sequences differ")
    record["rgb"] = {
        "gate": read(out() / "rgb-gate.json"),
        "camera_trace_exact": True,
        "control": ar["metrics"],
        "candidate": br["metrics"],
        "control_regions": ar["group_metrics"],
        "candidate_regions": br["group_metrics"],
        "dropout_statistics": [d["stats"] for d in drops],
    }
    guard()
    record["seconds"] = time.monotonic() - start
    write_new_json(out() / "audit-result.json", record)
    print(
        json.dumps(
            {
                "rgb_gate": record["rgb"]["gate"],
                "semantic": {
                    k: v.get("delta", {}).get("holdout", v.get("status"))
                    for k, v in record["semantic"].items()
                },
            }
        ),
        flush=True,
    )
