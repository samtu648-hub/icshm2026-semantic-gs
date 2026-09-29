#!/usr/bin/env python3
"""Run the SEM-425 frozen-RGB152 direct five-logit engineering preflight."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_direct_semantics import (  # noqa: E402
    build_direct_logit_field,
    direct_logit_sha256,
    render_direct_probabilities,
)
from src.quality_feature_splatting import crop_and_flip, crop_camera  # noqa: E402
from src.quality_protocol import sha256  # noqa: E402


EID = "SEM-425-RGB152-DIRECT5-PF"
CONFIG = ROOT / "experiments/runs" / EID / "config.yaml"
RGB_PARAMETER_NAMES = (
    "_xyz",
    "_features_dc",
    "_features_rest",
    "_scaling",
    "_rotation",
    "_opacity",
    "_exposure",
)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tensor_sha256(tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def rgb_state_sha256(model):
    return {name: tensor_sha256(getattr(model, name)) for name in RGB_PARAMETER_NAMES}


def roles(config):
    split = read(ROOT / config["split_path"])
    fold = split["primary_fold"]
    training = sorted(fold["semantic_train_labeled"])
    confirmation = sorted(fold["semantic_validation_labeled"])
    buffer_names = sorted(fold["buffer_excluded_all_cameras"])
    if (
        len(training) != config["semantic_train_views"]
        or len(confirmation) != config["semantic_confirmation_views"]
        or len(buffer_names) != config["buffer_excluded_views"]
        or set(training) & set(confirmation)
        or set(training) & set(buffer_names)
        or set(confirmation) & set(buffer_names)
    ):
        raise ValueError("SEM425 registered role isolation failed")
    return training, confirmation, buffer_names


def setup_runtime(config, artifact_root: Path):
    source = ROOT / config["rgb_source"]
    cuda = artifact_root / "r1/toolchain/cuda-12.8"
    os.environ.update(
        CUDA_HOME=str(cuda),
        PATH=str(cuda / "bin") + os.pathsep + os.environ.get("PATH", ""),
        LD_LIBRARY_PATH=str(cuda / "lib64") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""),
        PYTHONDONTWRITEBYTECODE="1",
        WANDB_MODE="disabled",
    )
    sys.path[:0] = [
        str(source),
        str(source / "submodules/diff-gaussian-rasterization"),
        str(source / "submodules/simple-knn"),
    ]
    np = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    Image = importlib.import_module("PIL.Image")
    arguments = importlib.import_module("arguments")
    gaussian_renderer = importlib.import_module("gaussian_renderer")
    from src.quality_cameras import read_camera_requests

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    parser = argparse.ArgumentParser(add_help=False)
    pipeline_params = arguments.PipelineParams(parser)
    optimization_params = arguments.OptimizationParams(parser)
    parsed = parser.parse_args([])
    model = gaussian_renderer.GaussianModel(3)
    model._exposure = torch.nn.Parameter(torch.eye(3, 4, device="cuda")[None])
    model.pretrained_exposures = None
    captured, iteration = torch.load(
        ROOT / config["rgb_model"] / "chkpnt30000.pth",
        map_location="cuda",
        weights_only=False,
    )
    if iteration != 30000:
        raise ValueError("RGB152 checkpoint is not the sealed 30000-step endpoint")
    model.restore(captured, optimization_params.extract(parsed))
    for name in RGB_PARAMETER_NAMES:
        getattr(model, name).requires_grad_(False)
    if any(getattr(model, name).requires_grad for name in RGB_PARAMETER_NAMES):
        raise ValueError("A frozen RGB parameter remains trainable")
    cameras = read_camera_requests(ROOT / config["scene"] / "sparse/0")
    if len(cameras) != config["scene_views"]:
        raise ValueError("RGB152 scene camera count differs from the registered split")
    return SimpleNamespace(
        torch=torch,
        np=np,
        Image=Image,
        model=model,
        render=gaussian_renderer.render,
        pipe=pipeline_params.extract(parsed),
        background=torch.zeros(3, device="cuda"),
        cameras=cameras,
    )


def sampling_plan(names, config):
    rng = random.Random(config["seed"])
    crop = config["crop"]
    return [
        {
            "step": step,
            "name": names[rng.randrange(len(names))],
            "top": rng.randint(0, 989 - crop),
            "left": rng.randint(0, 1320 - crop),
            "flip": rng.random() < 0.5,
            "crop": crop,
        }
        for step in range(1, config["preflight"]["successful_updates"] + 1)
    ]


def class_weights(runtime, names, label_root):
    counts = runtime.np.zeros(5, dtype=runtime.np.int64)
    hashes = {}
    for name in names:
        path = label_root / name
        label = runtime.np.asarray(runtime.Image.open(path), dtype=runtime.np.uint8)
        if label.shape != (989, 1320) or int(label.max()) >= 5:
            raise ValueError(f"Invalid training label: {name}")
        counts += runtime.np.bincount(label.reshape(-1), minlength=5)
        hashes[name] = sha256(path)
    values = runtime.np.median(counts) / counts
    values = runtime.np.clip(values, 0.05, 20.0).astype(runtime.np.float32)
    return counts, values, hashes


def make_crop_view(runtime, sample):
    from src.quality_native_gaussian import make_native_view

    camera = crop_camera(
        runtime.cameras[sample["name"]],
        top=sample["top"],
        left=sample["left"],
        size=sample["crop"],
    )
    return make_native_view(camera)


def semantic_crop(runtime, logits, sample):
    probabilities = render_direct_probabilities(
        runtime.render,
        make_crop_view(runtime, sample),
        runtime.model,
        runtime.pipe,
        logits,
    )
    if sample["flip"]:
        probabilities = runtime.torch.flip(probabilities, dims=(-1,))
    return probabilities


def fresh_rgb(runtime, name):
    from src.quality_native_gaussian import make_native_view

    with runtime.torch.inference_mode():
        value = runtime.render(
            make_native_view(runtime.cameras[name]),
            runtime.model,
            runtime.pipe,
            runtime.background,
        )["render"]
    return runtime.torch.round(value.clamp(0, 1) * 255).to(runtime.torch.uint8).cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    config = read(CONFIG)
    output = ROOT / config["output_path"]
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    result = {
        "schema_version": 1,
        "experiment_id": EID,
        "status": "running",
        "quality_gate": False,
        "confirmation_rgb_accesses": 0,
        "confirmation_label_accesses": 0,
        "buffer_rgb_accesses": 0,
        "buffer_label_accesses": 0,
    }
    write(output / "result.json", result)
    started = time.monotonic()
    try:
        training, confirmation, buffer_names = roles(config)
        rgb_result_path = ROOT / config["rgb_run"] / "result.json"
        rgb_result = read(rgb_result_path)
        if (
            rgb_result.get("experiment_id")
            != "RGB-152-STANDARD-ABSGRAD-SPATIAL30"
            or rgb_result.get("status") != "completed_spatial_training"
        ):
            raise ValueError("The sealed RGB152 spatial parent is not complete")
        runtime = setup_runtime(config, args.artifact_root.resolve())
        runtime.torch.cuda.reset_peak_memory_stats()
        label_root = ROOT / config["label_root"]
        counts, weights_np, label_hashes = class_weights(runtime, training, label_root)
        label_manifest = read(ROOT / config["label_manifest"])
        registered_labels = {
            row["image"]: row["label_png_sha256"]
            for row in label_manifest["records"]
        }
        if any(label_hashes[name] != registered_labels.get(name) for name in training):
            raise ValueError("A training label differs from the frozen manifest")
        plan = sampling_plan(training, config)
        write(output / "sampling-plan.json", {"samples": plan})
        before_state = rgb_state_sha256(runtime.model)
        fixed_name = training[0]
        before_rgb = fresh_rgb(runtime, fixed_name)
        logits = build_direct_logit_field(
            runtime.model.get_xyz.shape[0],
            device="cuda",
            dtype=runtime.model.get_xyz.dtype,
        )
        initial_hash = direct_logit_sha256(logits)
        optimizer = runtime.torch.optim.Adam(
            [logits],
            lr=config["preflight"]["learning_rate"],
            weight_decay=config["preflight"]["weight_decay"],
        )
        weights = runtime.torch.from_numpy(weights_np).cuda()
        rows = []
        first_gradient_norm = None
        probability_sum_max_error = 0.0
        for sample in plan:
            optimizer.zero_grad(set_to_none=True)
            probabilities = semantic_crop(runtime, logits, sample)
            target_np = crop_and_flip(
                runtime.np.asarray(runtime.Image.open(label_root / sample["name"]), dtype=runtime.np.uint8),
                sample,
            )
            target = runtime.torch.from_numpy(target_np.astype("int64")).cuda()
            loss = runtime.torch.nn.functional.nll_loss(
                probabilities.clamp_min(runtime.torch.finfo(probabilities.dtype).tiny).log()[None],
                target[None],
                weight=weights,
            )
            loss.backward()
            if logits.grad is None or not runtime.torch.isfinite(logits.grad).all():
                raise FloatingPointError("Direct semantic gradient is missing or non-finite")
            gradient_norm = float(logits.grad.norm())
            if first_gradient_norm is None:
                first_gradient_norm = gradient_norm
            if gradient_norm <= 0:
                raise ValueError("Direct semantic gradient is zero")
            probability_error = float((probabilities.sum(dim=0) - 1).abs().max())
            probability_sum_max_error = max(probability_sum_max_error, probability_error)
            optimizer.step()
            if not runtime.torch.isfinite(logits).all():
                raise FloatingPointError("Direct semantic logits became non-finite")
            rows.append(
                {
                    "step": sample["step"],
                    "sample": sample,
                    "loss": float(loss.detach()),
                    "gradient_norm": gradient_norm,
                    "probability_sum_max_error": probability_error,
                }
            )
        checkpoint = output / "direct-logits-step8.pt"
        runtime.torch.save(
            {
                "schema_version": 1,
                "experiment_id": EID,
                "step": len(plan),
                "config_sha256": sha256(CONFIG),
                "direct_logits": logits.detach().cpu(),
            },
            checkpoint,
        )
        payload = runtime.torch.load(checkpoint, map_location="cpu", weights_only=True)
        replay = build_direct_logit_field(
            runtime.model.get_xyz.shape[0],
            device="cuda",
            dtype=runtime.model.get_xyz.dtype,
        )
        with runtime.torch.no_grad():
            replay.copy_(payload["direct_logits"].cuda())
            reference_probability = semantic_crop(runtime, logits, plan[0])
            replay_probability = semantic_crop(runtime, replay, plan[0])
        if direct_logit_sha256(logits) != direct_logit_sha256(replay):
            raise ValueError("Direct semantic checkpoint state replay differs")
        if not runtime.torch.equal(reference_probability, replay_probability):
            raise ValueError("Direct semantic checkpoint render replay differs")
        after_state = rgb_state_sha256(runtime.model)
        after_rgb = fresh_rgb(runtime, fixed_name)
        if before_state != after_state or not runtime.torch.equal(before_rgb, after_rgb):
            raise ValueError("Frozen RGB isolation gate failed")
        prediction = reference_probability.argmax(dim=0).cpu().numpy().astype(runtime.np.uint8)
        runtime.Image.fromarray(prediction, mode="L").save(output / "training-probe-semantic.png")
        allocated = int(runtime.torch.cuda.max_memory_allocated())
        reserved = int(runtime.torch.cuda.max_memory_reserved())
        if reserved >= 12 * 1024**3:
            raise RuntimeError("SEM425 exceeded the registered 12 GiB resource gate")
        result.update(
            status="completed_engineering_preflight",
            split_id=config["split_id"],
            training_views=len(training),
            confirmation_views=len(confirmation),
            buffer_views=len(buffer_names),
            training_label_hashes_read=len(label_hashes),
            class_pixel_counts=counts.tolist(),
            class_weights=weights_np.tolist(),
            point_count=int(logits.shape[0]),
            trainable_parameter_count=int(logits.numel()),
            trainable_parameters=["direct_logits"],
            successful_updates=len(rows),
            first_gradient_norm=first_gradient_norm,
            probability_sum_max_error=probability_sum_max_error,
            initial_direct_logit_sha256=initial_hash,
            final_direct_logit_sha256=direct_logit_sha256(logits),
            checkpoint_sha256=sha256(checkpoint),
            checkpoint_state_replay_exact=True,
            checkpoint_render_replay_exact=True,
            rgb_state_hashes_before=before_state,
            rgb_state_hashes_after=after_state,
            rgb_state_hashes_equal=before_state == after_state,
            fixed_rgb_view=fixed_name,
            fresh_rgb_exact=True,
            frozen_rgb_max_absolute_pixel_difference=int(
                (before_rgb.to(runtime.torch.int16) - after_rgb.to(runtime.torch.int16)).abs().max()
            ),
            peak_allocated_bytes=allocated,
            peak_reserved_bytes=reserved,
            peak_reserved_mib=reserved / 1024**2,
            update_trace=rows,
            config_sha256=sha256(CONFIG),
            split_sha256=sha256(ROOT / config["split_path"]),
            label_manifest_sha256=sha256(ROOT / config["label_manifest"]),
            rgb_result_sha256=sha256(rgb_result_path),
            rgb_checkpoint_sha256=sha256(ROOT / config["rgb_model"] / "chkpnt30000.pth"),
            code_sha256={
                path: sha256(ROOT / path)
                for path in (
                    "scripts/run_sem425_direct5_preflight.py",
                    "src/quality_direct_semantics.py",
                    "src/quality_feature_splatting.py",
                    "src/quality_native_gaussian.py",
                    "src/quality_cameras.py",
                    "src/quality_protocol.py",
                )
            },
            seconds=time.monotonic() - started,
            next_authorized_action="Register a separate spatial training experiment only after independent audit of this preflight.",
        )
        write(output / "result.json", result)
        print(json.dumps({key: result[key] for key in ("status", "point_count", "successful_updates", "first_gradient_norm", "probability_sum_max_error", "peak_reserved_mib", "seconds")}, indent=2))
    except BaseException as exc:
        result.update(
            status="failed_preserved",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            seconds=time.monotonic() - started,
        )
        write(output / "result.json", result)
        raise


if __name__ == "__main__":
    main()
