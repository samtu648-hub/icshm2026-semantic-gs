#!/usr/bin/env python3
"""Run the matched SEM387 frozen-RGB150 Feature Splatting experiment."""

from __future__ import annotations

import argparse
from dataclasses import replace
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_dinov3_mask2former import module_state_sha256, validate_model_dir
from src.quality_feature_splatting import (
    build_feature_splatting_field,
    crop_and_flip,
    crop_camera,
    field_state_sha256,
    final_patch_tokens,
    patch_cosine_loss,
    render_signed_features,
)
from src.quality_protocol import CLASSES, sha256, verify_hashes, write_new_json


EID = "SEM-387-RGB150-FEATURE-SPLATTING-DINOV3"
CONFIG = ROOT / "experiments/runs" / EID / "config.yaml"
CODE_PATHS = (
    "scripts/run_sem387_feature_splatting.py",
    "src/quality_feature_splatting.py",
    "src/quality_dinov3_mask2former.py",
    "src/quality_native_gaussian.py",
    "src/quality_cameras.py",
    "src/quality_semantic_audit.py",
    "src/quality_protocol.py",
)
RGB_PARAMETER_NAMES = (
    "_xyz",
    "_features_dc",
    "_features_rest",
    "_scaling",
    "_rotation",
    "_opacity",
    "_exposure",
)


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def tensor_sha256(tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def rgb_state_sha256(model) -> dict[str, str]:
    return {name: tensor_sha256(getattr(model, name)) for name in RGB_PARAMETER_NAMES}


def validate_dinov3_provenance(config: dict[str, Any], model_dir: Path):
    hashes = validate_model_dir(model_dir)
    if hashes != config["dinov3"]["file_sha256"]:
        raise ValueError("DINOv3 files differ from the registered source")
    manifest_path = model_dir / "download_manifest.json"
    if sha256(manifest_path) != config["dinov3"]["source_manifest_sha256"]:
        raise ValueError("DINOv3 source manifest differs from SEM381")
    manifest = read(manifest_path)
    if (
        manifest["model_id"] != config["dinov3"]["model_id"]
        or manifest["hf_reference_revision"]
        != config["dinov3"]["hf_reference_revision"]
        or not manifest["official_weight_hash_matches_modelscope"]
        or manifest["status"] != "downloaded_and_verified"
    ):
        raise ValueError("DINOv3 source manifest provenance mismatch")
    return hashes, sha256(manifest_path)


def make_sampling_plan(names: list[str], seed: int, steps: int, crop: int):
    if not names or names != sorted(set(names)):
        raise ValueError("Sampling role must be sorted, unique, and non-empty")
    rng = random.Random(seed)
    return [
        {
            "step": step,
            "name": names[rng.randrange(len(names))],
            "top": rng.randint(0, 989 - crop),
            "left": rng.randint(0, 1320 - crop),
            "flip": rng.random() < 0.5,
            "crop": crop,
        }
        for step in range(1, steps + 1)
    ]


def normalized_parent_samples(record: dict, crop: int) -> list[dict[str, Any]]:
    samples = []
    for row in record["samples"]:
        samples.append(
            {
                "step": int(row["step"]),
                "name": str(row["name"]),
                "top": int(row["top"]),
                "left": int(row["left"]),
                "flip": bool(row["flip"]),
                "crop": crop,
            }
        )
    return samples


def roles(config: dict[str, Any]):
    split = read(ROOT / config["split_path"])
    fold = split["primary_fold"]
    labeled = sorted(fold["semantic_train_labeled"])
    holdout = sorted(fold["semantic_validation_labeled"])
    rgb_train = sorted(fold["train_rgb_400"])
    unlabeled = sorted(set(rgb_train) - set(labeled))
    if (
        len(labeled) != 270
        or len(unlabeled) != 100
        or len(holdout) != 30
        or set(labeled) & set(unlabeled)
        or (set(labeled) | set(unlabeled)) & set(holdout)
        or set(rgb_train) != set(labeled) | set(unlabeled)
    ):
        raise ValueError("SEM387 requires disjoint 270/100/30 registered roles")
    return split, labeled, unlabeled, holdout, rgb_train


def prepare(config: dict[str, Any]) -> None:
    output = ROOT / config["output_path"]
    if output.exists():
        raise FileExistsError(f"SEM387 output already exists: {output}")
    _, labeled, unlabeled, holdout, rgb_train = roles(config)
    rgb_run = ROOT / config["rgb_run"]
    rgb_evaluation = ROOT / config["rgb_evaluation"]
    rgb_result = read(rgb_run / "result.json")
    rgb_eval_result = read(rgb_evaluation / "result.json")
    if (
        rgb_result["status"] != "completed_dev30_training"
        or rgb_eval_result["status"] != "completed"
    ):
        raise ValueError("Sealed RGB150 training and evaluation are required")
    required_rgb_artifacts = {
        name: rgb_result["artifacts"][name]
        for name in (
            "model/chkpnt30000.pth",
            "model/point_cloud/iteration_30000/point_cloud.ply",
        )
    }
    verify_hashes(rgb_run, required_rgb_artifacts)
    rgb_source = ROOT / config["rgb_source"]
    verified_rgb_source = verify_hashes(rgb_source, rgb_result["source"])
    training_rgb_hashes = {
        name: rgb_eval_result["image_hashes"][name] for name in rgb_train
    }
    verify_hashes(rgb_evaluation / "images", training_rgb_hashes)

    label_root = ROOT / config["label_root"]
    label_manifest = ROOT / config["label_manifest"]
    manifest = read(label_manifest)
    records = {row["image"]: row for row in manifest["records"]}
    label_hashes = {name: sha256(label_root / name) for name in labeled}
    if any(label_hashes[name] != records[name]["label_png_sha256"] for name in labeled):
        raise ValueError("A training label differs from the frozen manifest")

    parent_plan_path = ROOT / config["parent_sampling_plan"]
    if sha256(parent_plan_path) != config["parent_sampling_plan_sha256"]:
        raise ValueError("SEM384 sampling plan hash changed")
    parent_plan = read(parent_plan_path)
    labeled_plan = normalized_parent_samples(parent_plan, config["crop"])
    if (
        len(labeled_plan) != config["iterations"]
        or [row["step"] for row in labeled_plan]
        != list(range(1, config["iterations"] + 1))
        or not {row["name"] for row in labeled_plan} <= set(labeled)
    ):
        raise ValueError("SEM384 labeled plan does not match SEM387")
    unlabeled_plan = make_sampling_plan(
        unlabeled,
        config["unlabeled_sampling_seed"],
        config["iterations"],
        config["crop"],
    )

    model_dir_env = config["dinov3"]["model_dir_env"]
    if model_dir_env not in os.environ:
        raise ValueError(f"{model_dir_env} is required for offline source binding")
    model_dir = Path(os.environ[model_dir_env]).resolve()
    dino_hashes, dino_manifest_hash = validate_dinov3_provenance(config, model_dir)

    output.mkdir(parents=True)
    prepare_output = output / "prepare"
    prepare_output.mkdir()
    labeled_plan_path = output / "labeled-sampling-plan.json"
    unlabeled_plan_path = output / "unlabeled-sampling-plan.json"
    write_new_json(
        labeled_plan_path,
        {
            "schema_version": 1,
            "experiment_id": EID,
            "role": "shared_labeled",
            "source": config["parent_sampling_plan"],
            "source_sha256": config["parent_sampling_plan_sha256"],
            "samples": labeled_plan,
            "holdout_names_excluded": holdout,
        },
    )
    write_new_json(
        unlabeled_plan_path,
        {
            "schema_version": 1,
            "experiment_id": EID,
            "role": "candidate_only_unlabeled_teacher",
            "seed": config["unlabeled_sampling_seed"],
            "samples": unlabeled_plan,
            "holdout_names_excluded": holdout,
        },
    )
    binding = {
        "schema_version": 1,
        "experiment_id": EID,
        "config_sha256": sha256(CONFIG),
        "code_sha256": {path: sha256(ROOT / path) for path in CODE_PATHS},
        "split_sha256": sha256(ROOT / config["split_path"]),
        "label_manifest_sha256": sha256(label_manifest),
        "training_label_hashes": label_hashes,
        "training_rgb_hashes": training_rgb_hashes,
        "rgb_result_sha256": sha256(rgb_run / "result.json"),
        "rgb_evaluation_result_sha256": sha256(rgb_evaluation / "result.json"),
        "rgb_artifacts": required_rgb_artifacts,
        "rgb_source_file_count": len(verified_rgb_source),
        "labeled_sampling_plan_sha256": sha256(labeled_plan_path),
        "unlabeled_sampling_plan_sha256": sha256(unlabeled_plan_path),
        "dinov3_source_sha256": dino_hashes,
        "dinov3_source_manifest_sha256": dino_manifest_hash,
        "seed": config["seed"],
        "access": {
            "labeled_train": labeled,
            "unlabeled_teacher": unlabeled,
            "holdout_sealed": holdout,
        },
    }
    binding_path = output / "binding.json"
    write_new_json(binding_path, binding)
    write_new_json(
        prepare_output / "result.json",
        {
            "schema_version": 1,
            "status": "completed_holdout_sealed",
            "experiment_id": EID,
            "config_sha256": binding["config_sha256"],
            "binding_sha256": sha256(binding_path),
            "labeled_sampling_plan_sha256": binding["labeled_sampling_plan_sha256"],
            "unlabeled_sampling_plan_sha256": binding["unlabeled_sampling_plan_sha256"],
            "training_rgb_hashes_read": len(training_rgb_hashes),
            "training_label_hashes_read": len(label_hashes),
            "rgb_source_files_verified": len(verified_rgb_source),
            "holdout_rgb_hashes_read": 0,
            "holdout_label_hashes_read": 0,
            "dinov3_files_exact": len(dino_hashes),
        },
    )


def setup_runtime(config: dict[str, Any], artifact_root: Path, *, load_teacher: bool):
    source = ROOT / config["rgb_source"]
    runtime = ROOT / "outputs/quality/mask2former/SEM-340/runtime"
    cuda = artifact_root / "r1/toolchain/cuda-12.8"
    os.environ.update(
        CUDA_HOME=str(cuda),
        PATH=str(cuda / "bin") + os.pathsep + os.environ.get("PATH", ""),
        LD_LIBRARY_PATH=str(cuda / "lib64")
        + os.pathsep
        + os.environ.get("LD_LIBRARY_PATH", ""),
        TORCH_HOME=str(artifact_root / "r1/weights/torch"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        WANDB_MODE="disabled",
        PYTHONDONTWRITEBYTECODE="1",
    )
    sys.path[:0] = [
        str(source),
        str(source / "submodules/diff-gaussian-rasterization"),
        str(source / "submodules/simple-knn"),
        str(runtime),
    ]
    np = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    Image = importlib.import_module("PIL.Image")
    arguments = importlib.import_module("arguments")
    gaussian_renderer = importlib.import_module("gaussian_renderer")
    OptimizationParams = arguments.OptimizationParams
    PipelineParams = arguments.PipelineParams
    GaussianModel = gaussian_renderer.GaussianModel
    render = gaussian_renderer.render
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
    pipeline_params = PipelineParams(parser)
    optimization_params = OptimizationParams(parser)
    parsed = parser.parse_args([])
    model = GaussianModel(3)
    model._exposure = torch.nn.Parameter(torch.eye(3, 4, device="cuda")[None])
    model.pretrained_exposures = None
    checkpoint = ROOT / config["rgb_model"] / "chkpnt30000.pth"
    captured, iteration = torch.load(
        checkpoint, map_location="cuda", weights_only=False
    )
    if iteration != 30000:
        raise ValueError("RGB150 checkpoint is not the fixed 30000-step endpoint")
    model.restore(captured, optimization_params.extract(parsed))
    for name in RGB_PARAMETER_NAMES:
        getattr(model, name).requires_grad_(False)
    cameras = read_camera_requests(ROOT / config["scene"] / "sparse/0")
    if len(cameras) != 400:
        raise ValueError("RGB150 official camera set must contain 400 views")

    teacher = None
    teacher_loading = None
    if load_teacher:
        AutoModel = importlib.import_module("transformers").AutoModel

        model_dir = Path(os.environ[config["dinov3"]["model_dir_env"]]).resolve()
        validate_dinov3_provenance(config, model_dir)
        teacher, teacher_loading = AutoModel.from_pretrained(
            model_dir, local_files_only=True, output_loading_info=True
        )
        if any(
            teacher_loading[key]
            for key in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
        ):
            raise ValueError(f"Incomplete DINOv3 load: {teacher_loading}")
        teacher = teacher.cuda().eval().requires_grad_(False)

    return SimpleNamespace(
        torch=torch,
        np=np,
        Image=Image,
        source=source,
        model=model,
        render=render,
        pipe=pipeline_params.extract(parsed),
        background=torch.zeros(3, device="cuda"),
        cameras=cameras,
        teacher=teacher,
        teacher_loading=teacher_loading,
    )


class Lazy(dict):
    def __init__(self, loader):
        super().__init__()
        self.loader = loader

    def __missing__(self, key):
        self[key] = self.loader(key)
        return self[key]


def load_inputs(runtime, config, names):
    label_root = ROOT / config["label_root"]
    rgb_root = ROOT / config["rgb_evaluation"] / "images"
    images = Lazy(
        lambda name: runtime.np.asarray(
            runtime.Image.open(rgb_root / name).convert("RGB"), dtype=runtime.np.uint8
        ).copy()
    )
    labels = Lazy(
        lambda name: runtime.np.asarray(
            runtime.Image.open(label_root / name), dtype=runtime.np.uint8
        ).copy()
    )
    for name in names:
        if name not in runtime.cameras:
            raise ValueError(f"Missing registered camera: {name}")
    return images, labels


def render_sample(runtime, field, sample):
    camera = crop_camera(
        runtime.cameras[sample["name"]],
        top=sample["top"],
        left=sample["left"],
        size=sample["crop"],
    )
    from src.quality_native_gaussian import make_native_view

    rendered, alpha = render_signed_features(
        runtime.render,
        make_native_view(camera),
        runtime.model,
        runtime.pipe,
        field.latent,
        runtime.background,
    )
    if sample["flip"]:
        rendered = runtime.torch.flip(rendered, dims=(-1,))
        alpha = runtime.torch.flip(alpha, dims=(-1,))
    return rendered, alpha


def field_optimizer(field, config):
    torch = __import__("torch")
    decoder = [
        *field.trunk.parameters(),
        *field.class_head.parameters(),
        *field.teacher_head.parameters(),
    ]
    return torch.optim.Adam(
        [
            {
                "params": [field.latent],
                "lr": config["feature_splatting"]["latent_learning_rate"],
            },
            {
                "params": decoder,
                "lr": config["feature_splatting"]["decoder_learning_rate"],
            },
        ],
        weight_decay=config["feature_splatting"]["weight_decay"],
    )


def build_initial_state(runtime, config):
    runtime.torch.manual_seed(config["seed"])
    field = build_feature_splatting_field(
        runtime.model.get_xyz.shape[0],
        latent_dim=config["feature_splatting"]["latent_dim"],
        hidden_dim=256,
        class_count=len(CLASSES),
        teacher_dim=config["dinov3"]["hidden_size"],
        device="cuda",
        dtype=runtime.model.get_xyz.dtype,
    )
    digest = field_state_sha256(field)
    state = {
        name: value.detach().cpu().clone() for name, value in field.state_dict().items()
    }
    del field
    runtime.torch.cuda.empty_cache()
    return state, digest


def new_field(runtime, config, initial_state):
    field = build_feature_splatting_field(
        runtime.model.get_xyz.shape[0],
        latent_dim=config["feature_splatting"]["latent_dim"],
        hidden_dim=256,
        class_count=len(CLASSES),
        teacher_dim=config["dinov3"]["hidden_size"],
        device="cuda",
        dtype=runtime.model.get_xyz.dtype,
    )
    field.load_state_dict(initial_state, strict=True)
    return field


def save_checkpoint(path, field, optimizer, *, arm, step, binding_sha256):
    if path.exists():
        raise FileExistsError(path)
    __import__("torch").save(
        {
            "schema_version": 1,
            "experiment_id": EID,
            "arm": arm,
            "step": step,
            "binding_sha256": binding_sha256,
            "field": {
                name: value.detach().cpu() for name, value in field.state_dict().items()
            },
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def load_field_checkpoint(
    runtime, config, path, *, expected_arm, expected_step, binding_sha256
):
    payload = runtime.torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload["experiment_id"] != EID
        or payload["arm"] != expected_arm
        or payload["step"] != expected_step
        or payload["binding_sha256"] != binding_sha256
    ):
        raise ValueError("Feature Splatting checkpoint identity mismatch")
    field = new_field(runtime, config, payload["field"])
    return field, payload


def verify_rgb_training_view(runtime, config, name):
    from src.quality_native_gaussian import make_native_view

    with runtime.torch.no_grad():
        rgb = runtime.render(
            make_native_view(runtime.cameras[name]),
            runtime.model,
            runtime.pipe,
            runtime.background,
        )["render"]
    actual = runtime.np.rint(rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(
        runtime.np.uint8
    )
    expected = runtime.np.asarray(
        runtime.Image.open(ROOT / config["rgb_evaluation"] / "images" / name).convert(
            "RGB"
        ),
        dtype=runtime.np.uint8,
    )
    if not runtime.np.array_equal(actual, expected):
        raise ValueError("Fresh frozen RGB150 render differs from its sealed PNG")
    return True


def finite_field(field):
    torch = __import__("torch")
    for parameter in field.parameters():
        if not torch.isfinite(parameter).all() or (
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ):
            raise FloatingPointError(
                "Non-finite Feature Splatting parameter or gradient"
            )


def train_arm(
    runtime,
    config,
    binding,
    output,
    arm,
    initial_state,
    initial_hash,
    labeled_plan,
    unlabeled_plan,
    images,
    labels,
    *,
    steps,
):
    torch = runtime.torch
    functional = torch.nn.functional
    field = new_field(runtime, config, initial_state).train()
    if field_state_sha256(field) != initial_hash:
        raise ValueError("Matched arm initialization differs")
    optimizer = field_optimizer(field, config)
    arm_output = output / arm
    arm_output.mkdir()
    result = {
        "schema_version": 1,
        "status": "running",
        "experiment_id": EID,
        "arm": arm,
        "steps_requested": steps,
        "successful_steps": 0,
        "initial_state_sha256": initial_hash,
        "labeled_sampling_plan_sha256": binding["labeled_sampling_plan_sha256"],
        "unlabeled_sampling_plan_sha256": binding["unlabeled_sampling_plan_sha256"]
        if arm == "dinov3_distilled"
        else None,
        "teacher_rgb_accesses": 0,
        "label_accesses": 0,
        "first_step_gradient_norms": None,
    }
    teacher_before = None
    if arm == "dinov3_distilled":
        if (
            runtime.teacher is None
            or runtime.teacher.training
            or any(
                parameter.requires_grad for parameter in runtime.teacher.parameters()
            )
        ):
            raise ValueError("Candidate requires an immutable eval-mode DINOv3 teacher")
        teacher_before = module_state_sha256(runtime.teacher)
    started = time.monotonic()
    try:
        with (arm_output / "training.jsonl").open("x", encoding="utf-8") as journal:
            for step in range(1, steps + 1):
                labeled_sample = labeled_plan[step - 1]
                optimizer.zero_grad(set_to_none=True)
                rendered, alpha = render_sample(runtime, field, labeled_sample)
                logits = field.classify(rendered)
                target_np = crop_and_flip(
                    labels[labeled_sample["name"]], labeled_sample
                )
                target = torch.from_numpy(target_np.astype("int64")).cuda()
                semantic_loss = functional.cross_entropy(logits, target[None])
                semantic_loss.backward()
                result["label_accesses"] += 1
                probability_error = float(
                    (logits.softmax(dim=1).sum(dim=1) - 1).abs().max().detach()
                )
                alpha_range = [float(alpha.min()), float(alpha.max())]
                dino_loss = None
                unlabeled_sample = None
                if arm == "dinov3_distilled":
                    unlabeled_sample = unlabeled_plan[step - 1]
                    rgb_np = crop_and_flip(
                        images[unlabeled_sample["name"]], unlabeled_sample
                    )
                    rgb = (
                        torch.from_numpy(rgb_np.transpose(2, 0, 1).copy())
                        .cuda()
                        .float()[None]
                        / 255.0
                    )
                    teacher_features = final_patch_tokens(runtime.teacher, rgb)
                    del rgb
                    rendered_teacher, _ = render_sample(
                        runtime, field, unlabeled_sample
                    )
                    patch_latent = functional.interpolate(
                        rendered_teacher[None],
                        size=teacher_features.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    student_features = field.project_teacher(patch_latent)
                    dino_loss = patch_cosine_loss(student_features, teacher_features)
                    (config["loss"]["distillation_weight"] * dino_loss).backward()
                    result["teacher_rgb_accesses"] += 1
                finite_field(field)
                if step == 1:
                    result["first_step_gradient_norms"] = {
                        "latent": float(field.latent.grad.norm()),
                        "trunk": float(next(field.trunk.parameters()).grad.norm()),
                        "class_head": float(
                            next(field.class_head.parameters()).grad.norm()
                        ),
                        "teacher_head": (
                            None
                            if next(field.teacher_head.parameters()).grad is None
                            else float(
                                next(field.teacher_head.parameters()).grad.norm()
                            )
                        ),
                    }
                    if (
                        min(
                            result["first_step_gradient_norms"][name]
                            for name in ("latent", "trunk", "class_head")
                        )
                        <= 0
                    ):
                        raise ValueError("Required supervised gradient is zero")
                    if (
                        arm == "label_only"
                        and result["first_step_gradient_norms"]["teacher_head"]
                        is not None
                    ):
                        raise ValueError("Control teacher head received a gradient")
                    if arm == "dinov3_distilled" and not (
                        result["first_step_gradient_norms"]["teacher_head"]
                        and result["first_step_gradient_norms"]["teacher_head"] > 0
                    ):
                        raise ValueError("Candidate teacher head gradient is zero")
                optimizer.step()
                finite_field(field)
                result["successful_steps"] = step
                row = {
                    "step": step,
                    "labeled_sample": labeled_sample,
                    "unlabeled_teacher_sample": unlabeled_sample,
                    "semantic_loss": float(semantic_loss.detach()),
                    "dino_cosine_loss": None
                    if dino_loss is None
                    else float(dino_loss.detach()),
                    "probability_sum_max_error": probability_error,
                    "alpha_range": alpha_range,
                    "elapsed_seconds": time.monotonic() - started,
                }
                journal.write(json.dumps(row) + "\n")
                if step == 1 or step % 100 == 0:
                    journal.flush()
                    print(json.dumps({"arm": arm, **row}), flush=True)
                if steps == config["iterations"] and step in (
                    config["analysis_checkpoint"],
                    config["primary_checkpoint"],
                ):
                    save_checkpoint(
                        arm_output / f"model-{step}.pt",
                        field,
                        optimizer,
                        arm=arm,
                        step=step,
                        binding_sha256=sha256(
                            ROOT / config["output_path"] / "binding.json"
                        ),
                    )
        final_step = steps
        if steps != config["iterations"]:
            save_checkpoint(
                arm_output / f"diagnostic-{steps}.pt",
                field,
                optimizer,
                arm=arm,
                step=steps,
                binding_sha256=sha256(ROOT / config["output_path"] / "binding.json"),
            )
            checkpoint = arm_output / f"diagnostic-{steps}.pt"
        else:
            checkpoint = arm_output / f"model-{final_step}.pt"
        replay, _ = load_field_checkpoint(
            runtime,
            config,
            checkpoint,
            expected_arm=arm,
            expected_step=final_step,
            binding_sha256=sha256(ROOT / config["output_path"] / "binding.json"),
        )
        if field_state_sha256(field) != field_state_sha256(replay):
            raise ValueError("Fresh checkpoint field replay differs")
        with torch.no_grad():
            reference_latent, _ = render_sample(runtime, field, labeled_plan[0])
            replay_latent, _ = render_sample(runtime, replay, labeled_plan[0])
            reference_logits = field.classify(reference_latent)
            replay_logits = replay.classify(replay_latent)
        if not torch.equal(reference_latent, replay_latent) or not torch.equal(
            reference_logits, replay_logits
        ):
            raise ValueError("Fresh checkpoint render replay differs")
        result.update(
            status="completed",
            final_state_sha256=field_state_sha256(field),
            checkpoint_state_replay_exact=True,
            checkpoint_render_replay_exact=True,
            checkpoint_sha256=sha256(checkpoint),
            training_seconds=time.monotonic() - started,
        )
        if steps == config["iterations"]:
            result["checkpoint_sha256s"] = {
                str(step): sha256(arm_output / f"model-{step}.pt")
                for step in (
                    config["analysis_checkpoint"],
                    config["primary_checkpoint"],
                )
            }
        if arm == "dinov3_distilled":
            teacher_after = module_state_sha256(runtime.teacher)
            if teacher_after != teacher_before or any(
                parameter.grad is not None for parameter in runtime.teacher.parameters()
            ):
                raise ValueError("DINOv3 teacher changed or accumulated gradients")
            result.update(
                initial_dinov3_state_sha256=teacher_before,
                final_dinov3_state_sha256=teacher_after,
                dinov3_frozen_exact=True,
            )
        write_new_json(arm_output / "result.json", result)
        return result
    except BaseException as exc:
        result.update(
            status="failed_preserved",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            seconds=time.monotonic() - started,
        )
        if not (arm_output / "result.json").exists():
            write_new_json(arm_output / "result.json", result)
        try:
            save_checkpoint(
                arm_output / "failure-state.pt",
                field,
                optimizer,
                arm=arm,
                step=result["successful_steps"],
                binding_sha256=sha256(ROOT / config["output_path"] / "binding.json"),
            )
        except BaseException:
            pass
        raise
    finally:
        del field, optimizer
        runtime.torch.cuda.empty_cache()


def verify_binding(config, binding):
    output = ROOT / config["output_path"]
    if sha256(CONFIG) != binding["config_sha256"]:
        raise ValueError("SEM387 config changed after preparation")
    verify_hashes(ROOT, binding["code_sha256"])
    if sha256(ROOT / config["split_path"]) != binding["split_sha256"]:
        raise ValueError("SEM387 split changed after preparation")
    if sha256(ROOT / config["label_manifest"]) != binding["label_manifest_sha256"]:
        raise ValueError("Training label manifest changed after preparation")
    if sha256(ROOT / config["rgb_run"] / "result.json") != binding["rgb_result_sha256"]:
        raise ValueError("RGB150 training result changed after preparation")
    if (
        sha256(ROOT / config["rgb_evaluation"] / "result.json")
        != binding["rgb_evaluation_result_sha256"]
    ):
        raise ValueError("RGB150 evaluation result changed after preparation")
    verify_hashes(ROOT / config["label_root"], binding["training_label_hashes"])
    verify_hashes(
        ROOT / config["rgb_evaluation"] / "images", binding["training_rgb_hashes"]
    )
    verify_hashes(ROOT / config["rgb_run"], binding["rgb_artifacts"])
    rgb_result = read(ROOT / config["rgb_run"] / "result.json")
    verified_source = verify_hashes(ROOT / config["rgb_source"], rgb_result["source"])
    if len(verified_source) != binding["rgb_source_file_count"]:
        raise ValueError("RGB150 renderer source file count changed")
    if (
        sha256(output / "labeled-sampling-plan.json")
        != binding["labeled_sampling_plan_sha256"]
    ):
        raise ValueError("Labeled sampling plan changed")
    if (
        sha256(output / "unlabeled-sampling-plan.json")
        != binding["unlabeled_sampling_plan_sha256"]
    ):
        raise ValueError("Unlabeled sampling plan changed")
    model_dir_env = config["dinov3"]["model_dir_env"]
    if model_dir_env not in os.environ:
        raise ValueError(f"{model_dir_env} is required for offline source verification")
    model_dir = Path(os.environ[model_dir_env]).resolve()
    dino_hashes, manifest_hash = validate_dinov3_provenance(config, model_dir)
    if (
        dino_hashes != binding["dinov3_source_sha256"]
        or manifest_hash != binding["dinov3_source_manifest_sha256"]
    ):
        raise ValueError("DINOv3 source changed after preparation")


def run_fit(config, artifact_root, *, preflight):
    output_root = ROOT / config["output_path"]
    binding_path = output_root / "binding.json"
    binding = read(binding_path)
    verify_binding(config, binding)
    stage = "preflight" if preflight else "training"
    output = output_root / stage
    output.mkdir(exist_ok=False)
    result = {
        "schema_version": 1,
        "status": "waiting_gpu_lock",
        "experiment_id": EID,
        "stage": stage,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": binding["config_sha256"],
        "binding_sha256": sha256(binding_path),
        "seed": config["seed"],
        "holdout_rgb_accessed": False,
        "holdout_labels_accessed": False,
    }
    write_new_json(output / "started.json", result)
    runtime = None
    started = time.monotonic()
    try:
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            queued = time.monotonic()
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
                raise RuntimeError(
                    "Other GPU compute process active outside the shared lock"
                )
            result["queue_seconds"] = time.monotonic() - queued
            result["gpu_processes_before"] = applications
            runtime = setup_runtime(config, artifact_root, load_teacher=True)
            runtime.torch.cuda.reset_peak_memory_stats()
            result["runtime"] = {
                "python": sys.version,
                "torch": runtime.torch.__version__,
                "cuda": runtime.torch.version.cuda,
                "transformers": importlib.metadata.version("transformers"),
                "gpu": runtime.torch.cuda.get_device_name(0),
                "total_memory_mib": runtime.torch.cuda.get_device_properties(
                    0
                ).total_memory
                / 2**20,
            }
            initial_rgb = rgb_state_sha256(runtime.model)
            ply_model = __import__(
                "gaussian_renderer", fromlist=["GaussianModel"]
            ).GaussianModel(3)
            ply_model.load_ply(
                str(
                    ROOT
                    / config["rgb_model"]
                    / "point_cloud/iteration_30000/point_cloud.ply"
                )
            )
            result["checkpoint_ply_parameter_exact"] = {
                name: bool(
                    runtime.torch.equal(
                        getattr(runtime.model, name), getattr(ply_model, name)
                    )
                )
                for name in RGB_PARAMETER_NAMES[:-1]
            }
            if not all(result["checkpoint_ply_parameter_exact"].values()):
                raise ValueError("RGB150 checkpoint and deployment PLY differ")
            del ply_model
            _, labeled, unlabeled, _, _ = roles(config)
            images, labels = load_inputs(runtime, config, set(labeled) | set(unlabeled))
            labeled_plan = read(output_root / "labeled-sampling-plan.json")["samples"]
            unlabeled_plan = read(output_root / "unlabeled-sampling-plan.json")[
                "samples"
            ]
            initial_state, initial_hash = build_initial_state(runtime, config)
            result["matched_initial_state_sha256"] = initial_hash
            steps = (
                config["preflight"]["successful_updates_per_arm"]
                if preflight
                else config["iterations"]
            )
            if not preflight:
                preflight_result = read(output_root / "preflight/result.json")
                if (
                    preflight_result["status"] != "completed"
                    or preflight_result["matched_initial_state_sha256"] != initial_hash
                ):
                    raise ValueError(
                        "A matched successful engineering preflight is required"
                    )
            result["fresh_rgb_before"] = {
                name: verify_rgb_training_view(runtime, config, name)
                for name in ("057.png", "159.png", "301.png")
            }
            if preflight:
                probe = new_field(runtime, config, initial_state).eval()
                with runtime.torch.no_grad():
                    zero_image, _ = render_sample(runtime, probe, labeled_plan[0])
                    constants = runtime.torch.linspace(
                        -1.5,
                        1.5,
                        probe.latent.shape[1],
                        device=probe.latent.device,
                        dtype=probe.latent.dtype,
                    )
                    probe.latent.copy_(constants[None].expand_as(probe.latent))
                    constant_image, alpha = render_sample(
                        runtime, probe, labeled_plan[0]
                    )
                    expected = constants[:, None, None] * alpha
                    chunk_error = float((constant_image - expected).abs().max())
                    zero_error = float(zero_image.abs().max())
                if zero_error > 2e-6 or chunk_error > 2e-5:
                    raise ValueError(
                        "Signed native feature rendering failed exactness gate"
                    )
                result["signed_renderer"] = {
                    "zero_field_max_abs": zero_error,
                    "constant_channel_chunk_reassembly_max_abs": chunk_error,
                    "channels": int(probe.latent.shape[1]),
                    "native_channels_per_pass": 3,
                }
                del probe
                runtime.torch.cuda.empty_cache()
            arm_results = {}
            for arm in config["arms"]:
                arm_results[arm] = train_arm(
                    runtime,
                    config,
                    binding,
                    output,
                    arm,
                    initial_state,
                    initial_hash,
                    labeled_plan,
                    unlabeled_plan,
                    images,
                    labels,
                    steps=steps,
                )
            if (
                arm_results["label_only"]["initial_state_sha256"]
                != arm_results["dinov3_distilled"]["initial_state_sha256"]
            ):
                raise ValueError("Arm initial states differ")
            if (
                arm_results["label_only"]["label_accesses"]
                != arm_results["dinov3_distilled"]["label_accesses"]
            ):
                raise ValueError("Arm labeled update counts differ")
            if rgb_state_sha256(runtime.model) != initial_rgb:
                raise ValueError("Frozen RGB150 parameter changed")
            result["fresh_rgb_after"] = {
                name: verify_rgb_training_view(runtime, config, name)
                for name in ("057.png", "159.png", "301.png")
            }
            runtime.torch.cuda.synchronize()
            peak_reserved = runtime.torch.cuda.max_memory_reserved() / 2**20
            total_memory = (
                runtime.torch.cuda.get_device_properties(0).total_memory / 2**20
            )
            if peak_reserved >= total_memory:
                raise ValueError(
                    "Feature Splatting exceeded the physical GPU memory gate"
                )
            result.update(
                status="completed",
                successful_steps_per_arm=steps,
                arm_results={
                    arm: output.name + "/" + arm + "/result.json"
                    for arm in config["arms"]
                },
                matched_initial_state_exact=True,
                matched_labeled_plan_exact=True,
                frozen_rgb_parameters_exact=True,
                resource_gate_12gb_passed=True,
                resource_gate_peak_reserved_mib=peak_reserved,
                resource_gate_total_memory_mib=total_memory,
                holdout_rgb_accessed=False,
                holdout_labels_accessed=False,
            )
    except BaseException as exc:
        result.update(
            status="failed_preserved",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        if runtime is not None:
            result["peak_allocated_mib"] = (
                runtime.torch.cuda.max_memory_allocated() / 2**20
            )
            result["peak_reserved_mib"] = (
                runtime.torch.cuda.max_memory_reserved() / 2**20
            )
        write_new_json(output / "result.json", result)


def boundary_metrics(values) -> dict[str, Any]:
    precision = float(values[0] / values[1]) if values[1] else None
    recall = float(values[2] / values[3]) if values[3] else None
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "tolerance_px": 2,
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "counts": values.tolist(),
    }


def infer(config, artifact_root):
    output_root = ROOT / config["output_path"]
    binding_path = output_root / "binding.json"
    binding = read(binding_path)
    verify_binding(config, binding)
    training = read(output_root / "training/result.json")
    if (
        training["status"] != "completed"
        or training["successful_steps_per_arm"] != config["iterations"]
    ):
        raise ValueError("Both sealed 6000-step arm endpoints are required")
    checkpoints = {
        arm: output_root / "training" / arm / f"model-{config['primary_checkpoint']}.pt"
        for arm in config["arms"]
    }
    checkpoint_hashes = {arm: sha256(path) for arm, path in checkpoints.items()}
    for arm, path in checkpoints.items():
        arm_result = read(path.parent / "result.json")
        if (
            arm_result["status"] != "completed"
            or arm_result["checkpoint_sha256"] != checkpoint_hashes[arm]
        ):
            raise ValueError(f"Unsealed endpoint for {arm}")

    output = output_root / "inference"
    output.mkdir(exist_ok=False)
    result = {
        "schema_version": 1,
        "status": "waiting_gpu_lock",
        "experiment_id": EID,
        "checkpoint_sha256_before_holdout_access": checkpoint_hashes,
        "holdout_access_begins_after_both_checkpoints_sealed": True,
        "binding_sha256": sha256(binding_path),
    }
    write_new_json(output / "started.json", result)
    runtime = None
    started = time.monotonic()
    try:
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            queued = time.monotonic()
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
                raise RuntimeError(
                    "Other GPU compute process active outside the shared lock"
                )
            result["queue_seconds"] = time.monotonic() - queued
            runtime = setup_runtime(config, artifact_root, load_teacher=False)
            runtime.torch.cuda.reset_peak_memory_stats()
            initial_rgb = rgb_state_sha256(runtime.model)
            result["initial_rgb_state_sha256"] = initial_rgb
            result["fresh_rgb_before"] = {
                name: verify_rgb_training_view(runtime, config, name)
                for name in ("057.png", "159.png", "301.png")
            }
            from PIL import Image
            from scipy.ndimage import binary_erosion, distance_transform_edt
            from src.quality_native_gaussian import make_native_view
            from src.quality_semantic_audit import independent_confusion, metrics

            split, labeled, _, holdout, _ = roles(config)
            labeled_set, holdout_set = set(labeled), set(holdout)
            group_by_name = {
                name: group["group_id"]
                for group in split["groups"]
                for name in group["images"]
            }
            trajectory_root = artifact_root / config["trajectory"]
            from src.quality_cameras import read_camera_requests

            trajectory = read_camera_requests(trajectory_root)
            if set(trajectory) & set(runtime.cameras):
                raise ValueError(
                    "Trajectory names collide with registered dataset cameras"
                )
            requests = {
                **runtime.cameras,
                **{
                    name: replace(camera, model="PINHOLE", radial_k=0.0)
                    for name, camera in trajectory.items()
                },
            }
            label_root = ROOT / config["label_root"]
            arm_summaries = {}
            for arm in config["arms"]:
                arm_output = output / arm
                prediction_root = arm_output / "predictions"
                prediction_root.mkdir(parents=True)
                field, _ = load_field_checkpoint(
                    runtime,
                    config,
                    checkpoints[arm],
                    expected_arm=arm,
                    expected_step=config["primary_checkpoint"],
                    binding_sha256=sha256(binding_path),
                )
                field.eval()
                totals = {
                    "train": runtime.np.zeros((5, 5), dtype=runtime.np.int64),
                    "holdout": runtime.np.zeros((5, 5), dtype=runtime.np.int64),
                    **{
                        group["group_id"]: runtime.np.zeros(
                            (5, 5), dtype=runtime.np.int64
                        )
                        for group in split["groups"]
                    },
                }
                boundary = {
                    key: runtime.np.zeros(4, dtype=runtime.np.int64) for key in totals
                }
                per_view = []
                times = []
                trajectory_predictions = []
                with runtime.torch.inference_mode():
                    for index, (name, camera) in enumerate(sorted(requests.items())):
                        runtime.torch.cuda.synchronize()
                        infer_started = time.monotonic()
                        rendered, _ = render_signed_features(
                            runtime.render,
                            make_native_view(camera),
                            runtime.model,
                            runtime.pipe,
                            field.latent,
                            runtime.background,
                        )
                        probability = field.classify(rendered).softmax(dim=1)[0]
                        if float((probability.sum(dim=0) - 1).abs().max()) > 2e-5:
                            raise ValueError("Semantic probabilities do not sum to one")
                        prediction = probability.argmax(dim=0).byte().cpu().numpy()
                        runtime.torch.cuda.synchronize()
                        times.append(time.monotonic() - infer_started)
                        Image.fromarray(prediction).save(prediction_root / name)
                        if name in trajectory:
                            trajectory_predictions.append(prediction)
                        if name in labeled_set or name in holdout_set:
                            target = runtime.np.asarray(
                                Image.open(label_root / name), dtype=runtime.np.uint8
                            )
                            confusion = independent_confusion(target, prediction)
                            role = "train" if name in labeled_set else "holdout"
                            keys = [role]
                            if role == "holdout":
                                keys.append(group_by_name[name])
                            cable = target == 2
                            gt_boundary = cable ^ binary_erosion(cable)
                            gt_distance = runtime.np.asarray(
                                distance_transform_edt(~gt_boundary)
                            )
                            predicted_cable = prediction == 2
                            predicted_boundary = predicted_cable ^ binary_erosion(
                                predicted_cable
                            )
                            predicted_distance = runtime.np.asarray(
                                distance_transform_edt(~predicted_boundary)
                            )
                            counts = runtime.np.asarray(
                                [
                                    runtime.np.sum(
                                        predicted_boundary & (gt_distance <= 2)
                                    ),
                                    runtime.np.sum(predicted_boundary),
                                    runtime.np.sum(
                                        gt_boundary & (predicted_distance <= 2)
                                    ),
                                    runtime.np.sum(gt_boundary),
                                ],
                                dtype=runtime.np.int64,
                            )
                            for key in keys:
                                totals[key] += confusion
                                boundary[key] += counts
                            per_view.append(
                                {
                                    "name": name,
                                    "role": role,
                                    "group": group_by_name.get(name),
                                    **metrics(confusion),
                                }
                            )
                        if index % 25 == 0:
                            print(
                                json.dumps(
                                    {
                                        "arm": arm,
                                        "inferred": index + 1,
                                        "total": len(requests),
                                    }
                                ),
                                flush=True,
                            )
                adjacent = [
                    float(runtime.np.mean(a != b))
                    for a, b in zip(trajectory_predictions, trajectory_predictions[1:])
                ]
                arm_summaries[arm] = {
                    "metrics": {key: metrics(matrix) for key, matrix in totals.items()},
                    "boundary_metrics": {
                        key: boundary_metrics(values)
                        for key, values in boundary.items()
                    },
                    "per_view": per_view,
                    "inference_seconds": times,
                    "inference_median_seconds": float(runtime.np.median(times[1:])),
                    "trajectory_adjacent_label_change": adjacent,
                    "trajectory_adjacent_label_change_mean": float(
                        runtime.np.mean(adjacent)
                    ),
                    "checkpoint_bytes": checkpoints[arm].stat().st_size,
                    "prediction_hashes": {
                        name: sha256(prediction_root / name)
                        for name in sorted(requests)
                    },
                }
                del field
                runtime.torch.cuda.empty_cache()
            paired = []
            for name in sorted(runtime.cameras):
                a = runtime.np.asarray(
                    Image.open(output / "label_only/predictions" / name),
                    dtype=runtime.np.uint8,
                )
                b = runtime.np.asarray(
                    Image.open(output / "dinov3_distilled/predictions" / name),
                    dtype=runtime.np.uint8,
                )
                paired.append(
                    {
                        "name": name,
                        "changed_pixels": int(runtime.np.count_nonzero(a != b)),
                        "changed_fraction": float(runtime.np.mean(a != b)),
                    }
                )
            if {
                arm: sha256(path) for arm, path in checkpoints.items()
            } != checkpoint_hashes:
                raise ValueError("Sealed endpoint changed during holdout inference")
            final_rgb = rgb_state_sha256(runtime.model)
            if final_rgb != initial_rgb:
                raise ValueError("Frozen RGB150 parameter changed during inference")
            result["final_rgb_state_sha256"] = final_rgb
            result["fresh_rgb_after"] = {
                name: verify_rgb_training_view(runtime, config, name)
                for name in ("057.png", "159.png", "301.png")
            }
            comparator_paths = {
                "SEM-384-RGB150-DEV270": ROOT
                / "outputs/quality/SEM-384/r1/inference/result.json",
                "SEM-386-RGB150-DEV270-CONTEXT50-MEMORY": ROOT
                / "outputs/quality/SEM-386/r2/inference/result.json",
            }
            comparators = {
                name: {
                    "result_sha256": sha256(path),
                    "metrics": read(path)["metrics"],
                }
                for name, path in comparator_paths.items()
            }
            result.update(
                status="completed",
                checkpoint_sha256_after_holdout_access=checkpoint_hashes,
                arms=arm_summaries,
                paired_prediction_changes=paired,
                comparators=comparators,
                sem386_visual_review_status="pending",
                train_labeled_count=len(labeled),
                holdout_labeled_count=len(holdout),
                unlabeled_count=100,
                trajectory_count=len(trajectory),
                frozen_rgb_parameters_exact=True,
            )
    except BaseException as exc:
        result.update(
            status="failed_preserved",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        if runtime is not None:
            result["peak_allocated_mib"] = (
                runtime.torch.cuda.max_memory_allocated() / 2**20
            )
            result["peak_reserved_mib"] = (
                runtime.torch.cuda.max_memory_reserved() / 2**20
            )
        write_new_json(output / "result.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("prepare", "preflight", "train", "inference"), required=True
    )
    args = parser.parse_args()
    config = read(CONFIG)
    if config["experiment_id"] != EID:
        raise ValueError("SEM387 config identity changed")
    if args.stage == "prepare":
        prepare(config)
    elif args.stage == "preflight":
        run_fit(config, args.artifact_root.resolve(), preflight=True)
    elif args.stage == "train":
        run_fit(config, args.artifact_root.resolve(), preflight=False)
    else:
        infer(config, args.artifact_root.resolve())


if __name__ == "__main__":
    main()
