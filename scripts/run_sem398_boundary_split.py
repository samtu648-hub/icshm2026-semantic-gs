#!/usr/bin/env python3
"""SEM-398 matched semantic-conflict versus RGB split-ranking experiment."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/runs/SEM-398-RGB152-BOUNDARY-SPLIT/config.yaml"
sys.path.insert(0, str(ROOT))


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def schedule(names, steps: int, seed: int) -> list[str]:
    names = sorted(names)
    output: list[str] = []
    rng = random.Random(seed)
    while len(output) < steps:
        epoch = list(names)
        rng.shuffle(epoch)
        output.extend(epoch)
    return output[:steps]


class GeometryView:
    """Expose the native model while detaching geometry from semantic loss."""

    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        if name in {"get_xyz", "get_scaling", "get_rotation", "get_opacity"}:
            return getattr(self.model, name).detach()
        return getattr(self.model, name)


def setup(config: dict, captured=None):
    source = ROOT / config["source"]
    sys.path[:0] = [
        str(source),
        str(source / "submodules/diff-gaussian-rasterization"),
        str(source / "submodules/simple-knn"),
    ]
    import numpy as np
    import torch
    from PIL import Image
    from arguments import OptimizationParams, PipelineParams
    from gaussian_renderer import GaussianModel, render
    from src.quality_cameras import read_camera_requests
    from src.quality_native_gaussian import make_native_view

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parser = argparse.ArgumentParser(add_help=False)
    pipeline_params = PipelineParams(parser)
    optimization_params = OptimizationParams(parser)
    empty = parser.parse_args([])
    model = GaussianModel(3)
    model._exposure = torch.nn.Parameter(torch.eye(3, 4, device="cuda")[None])
    model.pretrained_exposures = None
    if captured is None:
        captured, iteration = torch.load(
            ROOT / config["rgb_parent"] / "chkpnt30000.pth",
            map_location="cuda",
            weights_only=False,
        )
        if iteration != 30000:
            raise ValueError(f"Expected RGB152 iteration 30000, got {iteration}")
    model.restore(captured, optimization_params.extract(empty))
    model.update_learning_rate(30000)
    cameras = read_camera_requests(ROOT / config["scene"] / "sparse/0")
    views = {name: make_native_view(camera) for name, camera in cameras.items()}
    split = read(ROOT / config["split_path"])
    fold = split["primary_fold"]
    rgb_names = sorted(fold["train_rgb_400"])
    semantic_names = sorted(fold["semantic_train_labeled"])
    holdout_names = sorted(fold["semantic_validation_labeled"])
    expected = config.get(
        "role_counts", {"rgb_train": 353, "semantic_train": 256, "holdout": 30}
    )
    if (len(rgb_names), len(semantic_names), len(holdout_names)) != (
        expected["rgb_train"],
        expected["semantic_train"],
        expected["holdout"],
    ):
        raise ValueError("Unexpected spatial-confirmation role counts")
    if set(rgb_names) & set(holdout_names) or set(semantic_names) & set(holdout_names):
        raise ValueError("Confirmation roles overlap training")
    manifest = read(ROOT / config["label_manifest"])
    records = {record["image"]: record for record in manifest["records"]}

    class Lazy(dict):
        def __init__(self, loader):
            super().__init__()
            self.loader = loader

        def __missing__(self, key):
            self[key] = self.loader(key)
            return self[key]

    def label_loader(name):
        if name not in records:
            raise KeyError(name)
        path = ROOT / config["labels"] / name
        if sha256(path) != records[name]["label_png_sha256"]:
            raise ValueError(f"Label hash mismatch: {name}")
        return np.asarray(Image.open(path), dtype=np.uint8).copy()

    labels = Lazy(label_loader)
    photos = Lazy(
        lambda name: np.asarray(
            Image.open(ROOT / config["scene"] / "images" / name).convert("RGB"), dtype=np.uint8
        ).copy()
    )
    return SimpleNamespace(
        torch=torch,
        np=np,
        Image=Image,
        model=model,
        render=render,
        pipe=pipeline_params.extract(empty),
        views=views,
        labels=labels,
        photos=photos,
        rgb_names=rgb_names,
        semantic_names=semantic_names,
        holdout_names=holdout_names,
        split=split,
        bg=torch.zeros(3, device="cuda"),
        config=config,
    )


def semantic_probability(state, logits, name: str):
    torch = state.torch
    probability = logits.softmax(1)
    geometry = GeometryView(state.model)
    first = state.render(
        state.views[name],
        geometry,
        state.pipe,
        torch.tensor([1.0, 0.0, 0.0], device="cuda"),
        override_color=probability[:, :3].contiguous(),
    )["render"]
    last = state.render(
        state.views[name],
        geometry,
        state.pipe,
        state.bg,
        override_color=torch.cat([probability[:, 3:], torch.zeros_like(probability[:, :1])], 1).contiguous(),
    )["render"][:2]
    return torch.cat([first, last], 0)


def semantic_loss(state, logits, name: str):
    if name not in state.semantic_names:
        raise RuntimeError(f"Training attempted to access non-training label {name}")
    target = state.torch.from_numpy(state.labels[name].astype("int64")).cuda()
    probability = semantic_probability(state, logits, name)
    return state.torch.nn.functional.nll_loss(probability.clamp_min(1e-8).log()[None], target[None])


def rgb_loss(state, name: str, return_pack: bool = False):
    from utils.loss_utils import l1_loss, ssim

    pack = state.render(state.views[name], state.model, state.pipe, state.bg)
    target = (
        state.torch.from_numpy(state.photos[name].transpose(2, 0, 1).copy()).cuda().float() / 255.0
    )
    loss = 0.8 * l1_loss(pack["render"], target) + 0.2 * (1.0 - ssim(pack["render"], target))
    return (loss, pack) if return_pack else loss


def require_finite(*values) -> None:
    for value in values:
        if not value.isfinite().all():
            raise FloatingPointError("Non-finite tensor")


def save_torch_new(torch, path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(path)
    torch.save(value, path)


def preflight(state, output: Path, result: dict) -> None:
    torch, np = state.torch, state.np
    point_count = len(state.model.get_xyz)
    logits = torch.nn.Parameter(torch.zeros((point_count, 5), device="cuda"))
    checks = []
    for name in ["001.png", "159.png"]:
        with torch.no_grad():
            rendered = state.render(state.views[name], state.model, state.pipe, state.bg)["render"]
            array = np.rint(rendered.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            expected = np.asarray(state.Image.open(ROOT / state.config["parent_export"] / name))
            probability = semantic_probability(state, logits, name)
            error = float((probability.sum(0) - 1.0).abs().max())
        if not np.array_equal(array, expected) or error >= 2e-5:
            raise AssertionError(f"Native replay failed for {name}")
        checks.append({"name": name, "rgb_png_exact": True, "probability_sum_max_error": error})
    with torch.no_grad():
        values = torch.arange(logits.numel(), device="cuda", dtype=logits.dtype).reshape_as(logits)
        logits.copy_(torch.sin(values * 0.17))
    loss = semantic_loss(state, logits, "001.png")
    loss.backward()
    require_finite(logits.grad)
    if logits.grad[:, 2].abs().max() == 0:
        raise AssertionError("stay_cable logit gradient is zero")
    stay_cable_gradient_l1 = float(logits.grad[:, 2].abs().sum())
    if any(getattr(state.model, name).grad is not None for name in ["_xyz", "_scaling", "_rotation", "_opacity"]):
        raise AssertionError("Semantic gradient leaked into geometry")
    logits.grad = None
    state.model.optimizer.zero_grad(set_to_none=True)
    loss, pack = rgb_loss(state, "001.png", return_pack=True)
    loss.backward()
    screen_gradient = pack["viewspace_points"].grad
    require_finite(screen_gradient)
    if screen_gradient.shape != (point_count, 4):
        raise AssertionError(f"Expected AbsGS [N,4] screen gradient, got {screen_gradient.shape}")
    large = state.model.get_scaling.max(1).values > state.model.percent_dense * state.model.spatial_lr_scale
    if int(large.sum()) < state.config["split_budget"]:
        raise AssertionError("Not enough large Gaussians for fixed split budget")
    result.update(
        point_count=point_count,
        checks=checks,
        semantic_gradient_geometry_blocked=True,
        stay_cable_gradient_l1=stay_cable_gradient_l1,
        absgs_screen_gradient_shape=list(screen_gradient.shape),
        large_gaussians=int(large.sum()),
        spatial_lr_scale=float(state.model.spatial_lr_scale),
        split_budget=int(state.config["split_budget"]),
    )


def common(state, output: Path, result: dict) -> None:
    torch = state.torch
    config = state.config
    point_count = len(state.model.get_xyz)
    for name in ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"]:
        getattr(state.model, name).requires_grad_(False)
    logits = torch.nn.Parameter(torch.zeros((point_count, 5), device="cuda"))
    optimizer = torch.optim.Adam([logits], lr=config["semantic_lr"])
    names = schedule(state.semantic_names, config["semantic_fit_steps"], config["seed"])
    write(output / "semantic-sampling.json", names)
    started = time.monotonic()
    with (output / "training.jsonl").open("x", encoding="utf-8") as log:
        for step, name in enumerate(names, 1):
            optimizer.zero_grad(set_to_none=True)
            loss = semantic_loss(state, logits, name)
            loss.backward()
            require_finite(loss, logits, logits.grad)
            optimizer.step()
            if step % 100 == 0 or step == len(names):
                row = {"step": step, "name": name, "semantic_nll": float(loss), "seconds": time.monotonic() - started}
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
    positive = torch.zeros(point_count, device="cuda", dtype=torch.float64)
    negative = torch.zeros_like(positive)
    for index, name in enumerate(state.semantic_names, 1):
        optimizer.zero_grad(set_to_none=True)
        loss = semantic_loss(state, logits, name)
        loss.backward()
        gradient = logits.grad[:, config["class_id_of_interest"]].detach().double()
        magnitude = gradient.abs()
        active = magnitude[magnitude > 0]
        scale = active.median() if len(active) else magnitude.new_tensor(1.0)
        normalized = gradient / (magnitude + scale.clamp_min(torch.finfo(torch.float64).tiny))
        positive.add_(normalized.clamp_min(0))
        negative.add_((-normalized).clamp_min(0))
        if index % 32 == 0:
            print(json.dumps({"semantic_evidence": index, "total": len(state.semantic_names)}), flush=True)
    conflict = 2.0 * torch.minimum(positive, negative) / (
        config["semantic_conflict_prior"] + positive + negative
    )
    rgb_sum = torch.zeros(point_count, device="cuda", dtype=torch.float64)
    rgb_views = torch.zeros_like(rgb_sum)
    state.model.optimizer.zero_grad(set_to_none=True)
    for index, name in enumerate(state.rgb_names, 1):
        state.model.optimizer.zero_grad(set_to_none=True)
        loss, pack = rgb_loss(state, name, return_pack=True)
        loss.backward()
        gradient = torch.linalg.vector_norm(pack["viewspace_points"].grad[:, 2:], dim=1).double()
        visible = pack["visibility_filter"]
        rgb_sum[visible] += gradient[visible]
        rgb_views[visible] += 1.0
        if index % 32 == 0:
            print(json.dumps({"rgb_evidence": index, "total": len(state.rgb_names)}), flush=True)
    rgb_score = rgb_sum / rgb_views.clamp_min(1.0)
    large = state.model.get_scaling.max(1).values > config["large_gaussian_fraction_of_extent"] * state.model.spatial_lr_scale
    eligible = large & (rgb_score >= config["rgb_absolute_gradient_threshold"])
    from src.semantic_boundary_split import select_matched_pair

    pair = select_matched_pair(
        rgb_score.cpu().numpy(), conflict.cpu().numpy(), eligible.cpu().numpy(), config["split_budget"]
    )
    different = config["split_budget"] - pair.overlap
    if different < config["promotion_gates"]["minimum_different_selected_parents"]:
        raise RuntimeError(f"Only {different} selected parents differ; mechanism gate requires 512")
    evidence = {
        "rgb_score": rgb_score.float().cpu(),
        "semantic_positive": positive.float().cpu(),
        "semantic_negative": negative.float().cpu(),
        "semantic_conflict": conflict.float().cpu(),
        "eligible": eligible.cpu(),
        "control": torch.from_numpy(pair.control),
        "candidate": torch.from_numpy(pair.candidate),
    }
    save_torch_new(
        torch,
        output / "common.pt",
        {"logits": logits.detach().cpu(), "semantic_optimizer": optimizer.state_dict(), "evidence": evidence},
    )
    write(
        output / "selection.json",
        {
            "point_count": point_count,
            "eligible": int(eligible.sum()),
            "split_budget": config["split_budget"],
            "overlap": pair.overlap,
            "different": different,
            "control": pair.control.tolist(),
            "candidate": pair.candidate.tolist(),
        },
    )
    result.update(
        semantic_steps=len(names),
        training_seconds=time.monotonic() - started,
        eligible=int(eligible.sum()),
        selection_overlap=pair.overlap,
        different_selected_parents=different,
        common_sha256=sha256(output / "common.pt"),
        selection_sha256=sha256(output / "selection.json"),
    )


def run_arm(state, output: Path, result: dict, arm: str, attempt: str) -> None:
    torch = state.torch
    config = state.config
    common_dir = ROOT / config["output"] / f"common-{attempt}"
    payload = torch.load(common_dir / "common.pt", map_location="cpu", weights_only=False)
    selection = read(common_dir / "selection.json")
    selected = torch.tensor(selection[arm], device="cuda", dtype=torch.long)
    old_logits = payload["logits"].cuda()
    keep = torch.ones(len(old_logits), device="cuda", dtype=torch.bool)
    keep[selected] = False
    inherited = torch.cat([old_logits[keep], old_logits[selected].repeat(config["split_children"], 1)], 0)
    state.model.tmp_radii = torch.zeros(len(old_logits), device="cuda")
    split_mask = torch.zeros(len(old_logits), device="cuda")
    split_mask[selected] = 1.0
    torch.manual_seed(config["seed"] + 1)
    torch.cuda.manual_seed_all(config["seed"] + 1)
    state.model.densify_and_split(
        split_mask,
        0.5,
        state.model.spatial_lr_scale,
        N=config["split_children"],
    )
    if len(state.model.get_xyz) != len(inherited):
        raise AssertionError("Semantic sidecar and native split point counts differ")
    logits = torch.nn.Parameter(inherited)
    semantic_optimizer = torch.optim.Adam([logits], lr=config["semantic_lr"])
    names = schedule(state.rgb_names, config["restoration_steps"], config["seed"] + 2)
    write(output / "restoration-sampling.json", names)
    started = time.monotonic()
    semantic_set = set(state.semantic_names)
    with (output / "training.jsonl").open("x", encoding="utf-8") as log:
        for step, name in enumerate(names, 1):
            state.model.optimizer.zero_grad(set_to_none=True)
            semantic_optimizer.zero_grad(set_to_none=True)
            photometric = rgb_loss(state, name)
            loss = photometric
            semantic = None
            if name in semantic_set:
                semantic = semantic_loss(state, logits, name)
                loss = loss + config["semantic_weight"] * semantic
            loss.backward()
            require_finite(loss, logits)
            state.model.optimizer.step()
            if semantic is not None:
                semantic_optimizer.step()
            if step % 100 == 0 or step == len(names):
                row = {
                    "step": step,
                    "name": name,
                    "rgb_loss": float(photometric),
                    "semantic_nll": float(semantic) if semantic is not None else None,
                    "seconds": time.monotonic() - started,
                }
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
    endpoint = output / "endpoint.pt"
    save_torch_new(
        torch,
        endpoint,
        {
            "model": state.model.capture(),
            "logits": logits.detach(),
            "semantic_optimizer": semantic_optimizer.state_dict(),
            "arm": arm,
            "selected": selected.cpu(),
        },
    )
    result.update(
        arm=arm,
        before_points=len(old_logits),
        after_points=len(logits),
        selected_parents=len(selected),
        restoration_steps=len(names),
        training_seconds=time.monotonic() - started,
        endpoint_sha256=sha256(endpoint),
        semantic_geometry_route="detached",
    )


def boundary_counts(target, prediction, np):
    from scipy.ndimage import binary_erosion, distance_transform_edt

    truth = target == 2
    pred = prediction == 2
    truth_boundary = truth ^ binary_erosion(truth)
    pred_boundary = pred ^ binary_erosion(pred)
    truth_distance = distance_transform_edt(~truth_boundary)
    pred_distance = distance_transform_edt(~pred_boundary)
    return np.asarray(
        [
            np.sum(pred_boundary & (truth_distance <= 2)),
            np.sum(pred_boundary),
            np.sum(truth_boundary & (pred_distance <= 2)),
            np.sum(truth_boundary),
        ],
        dtype=np.int64,
    )


def evaluate_endpoint(state, logits, group_by_name):
    from scripts.evaluate_rgb import psnr, ssim
    from src.semantic2d import confusion_metrics, update_confusion

    torch, np = state.torch, state.np
    totals = {
        "train": np.zeros((5, 5), dtype=np.int64),
        "holdout": np.zeros((5, 5), dtype=np.int64),
    }
    boundaries = {
        "train": np.zeros(4, dtype=np.int64),
        "holdout": np.zeros(4, dtype=np.int64),
    }
    for group in state.split["groups"]:
        totals[group["group_id"]] = np.zeros((5, 5), dtype=np.int64)
        boundaries[group["group_id"]] = np.zeros(4, dtype=np.int64)
    rows = []
    training = set(state.semantic_names)
    for name in [*state.semantic_names, *state.holdout_names]:
        with torch.no_grad():
            probability = semantic_probability(state, logits, name)
            prediction = probability.argmax(0).cpu().numpy().astype(np.uint8)
            rendered = state.render(state.views[name], state.model, state.pipe, state.bg)["render"]
            rgb = rendered.permute(1, 2, 0).cpu().numpy().clip(0, 1)
        target = state.labels[name]
        role = "train" if name in training else "holdout"
        group = group_by_name.get(name)
        keys = [role] + ([group] if group is not None else [])
        for key in keys:
            update_confusion(totals[key], target, prediction)
            boundaries[key] += boundary_counts(target, prediction, np)
        truth_rgb = state.photos[name].astype(np.float32) / 255.0
        rows.append({"name": name, "role": role, "group": group, "PSNR": psnr(truth_rgb, rgb), "SSIM": ssim(truth_rgb, rgb)})
    def summarize_boundary(counts):
        precision = float(counts[0] / counts[1]) if counts[1] else 0.0
        recall = float(counts[2] / counts[3]) if counts[3] else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": precision, "recall": recall, "f1": f1, "counts": counts.tolist()}
    return {
        "semantic": {key: confusion_metrics(value) for key, value in totals.items()},
        "boundary": {key: summarize_boundary(value) for key, value in boundaries.items()},
        "rgb": {
            role: {
                "PSNR": float(np.mean([row["PSNR"] for row in rows if row["role"] == role])),
                "SSIM": float(np.mean([row["SSIM"] for row in rows if row["role"] == role])),
            }
            for role in ["train", "holdout"]
        },
        "per_view_rgb": rows,
    }


def evaluate(config: dict, output: Path, result: dict, attempt: str) -> None:
    import torch

    control_path = ROOT / config["output"] / f"control-{attempt}" / "endpoint.pt"
    candidate_path = ROOT / config["output"] / f"candidate-{attempt}" / "endpoint.pt"
    common_path = ROOT / config["output"] / f"common-{attempt}" / "common.pt"
    if not control_path.exists() or not candidate_path.exists():
        raise RuntimeError("Both sealed arm endpoints are required before confirmation access")
    endpoint_hashes = {"control": sha256(control_path), "candidate": sha256(candidate_path)}
    group_by_name = {
        name: group["group_id"] for group in read(ROOT / config["split_path"])["groups"] for name in group["images"]
    }
    metrics = {}
    common_payload = torch.load(common_path, map_location="cpu", weights_only=False)
    common_state = setup(config)
    metrics["common"] = evaluate_endpoint(common_state, common_payload["logits"].cuda(), group_by_name)
    del common_state
    torch.cuda.empty_cache()
    for arm, path in [("control", control_path), ("candidate", candidate_path)]:
        payload = torch.load(path, map_location="cuda", weights_only=False)
        state = setup(config, captured=payload["model"])
        metrics[arm] = evaluate_endpoint(state, payload["logits"], group_by_name)
        del state, payload
        torch.cuda.empty_cache()
    if endpoint_hashes != {"control": sha256(control_path), "candidate": sha256(candidate_path)}:
        raise RuntimeError("A sealed endpoint changed during confirmation evaluation")
    control = metrics["control"]
    candidate = metrics["candidate"]
    c_sem = control["semantic"]["holdout"]
    x_sem = candidate["semantic"]["holdout"]
    deltas = {
        "boundary_f1": candidate["boundary"]["holdout"]["f1"] - control["boundary"]["holdout"]["f1"],
        "five_class_miou": x_sem["mIoU_with_background"] - c_sem["mIoU_with_background"],
        "stay_cable_iou": x_sem["per_class"]["stay_cable"]["iou"] - c_sem["per_class"]["stay_cable"]["iou"],
        "stay_cable_precision": x_sem["per_class"]["stay_cable"]["precision"] - c_sem["per_class"]["stay_cable"]["precision"],
        "PSNR": candidate["rgb"]["holdout"]["PSNR"] - control["rgb"]["holdout"]["PSNR"],
        "SSIM": candidate["rgb"]["holdout"]["SSIM"] - control["rgb"]["holdout"]["SSIM"],
    }
    gates = config["promotion_gates"]
    decisions = {
        "boundary_f1": deltas["boundary_f1"] >= gates["confirmation_boundary_f1_gain_at_least"],
        "five_class_miou": deltas["five_class_miou"] >= -gates["confirmation_five_class_miou_drop_at_most"],
        "stay_cable_iou": deltas["stay_cable_iou"] >= -gates["confirmation_stay_cable_iou_drop_at_most"],
        "stay_cable_precision": deltas["stay_cable_precision"] >= -gates["confirmation_stay_cable_precision_drop_at_most"],
        "PSNR": deltas["PSNR"] >= -gates["confirmation_rgb_psnr_drop_db_at_most"],
        "SSIM": deltas["SSIM"] >= -gates["confirmation_rgb_ssim_drop_at_most"],
    }
    result.update(
        both_endpoints_sealed_before_confirmation_access=True,
        endpoint_hashes=endpoint_hashes,
        metrics=metrics,
        candidate_minus_control=deltas,
        gate_pass=decisions,
        promotion_authorized=all(decisions.values()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["preflight", "common", "control", "candidate", "evaluate"], required=True)
    parser.add_argument("--attempt", default="r1")
    parser.add_argument("--source-attempt", default=None, help="Existing common/arm attempt used by evaluate")
    parser.add_argument("--config", type=Path, default=CONFIG, help="Registered SEM-398-family config")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = read(config_path)
    output = ROOT / config["output"] / f"{args.stage}-{args.attempt}"
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "experiment_id": config["experiment_id"],
        "stage": args.stage,
        "status": "starting",
        "seed": config["seed"],
        "config_sha256": sha256(config_path),
        "script_sha256": sha256(Path(__file__)),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    write(output / "started.json", result)
    state = None
    started = time.monotonic()
    try:
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if args.stage == "evaluate":
                evaluate(config, output, result, args.source_attempt or args.attempt)
            else:
                state = setup(config)
                state.torch.cuda.reset_peak_memory_stats()
                if args.stage == "preflight":
                    preflight(state, output, result)
                elif args.stage == "common":
                    common(state, output, result)
                else:
                    run_arm(state, output, result, args.stage, args.attempt)
            result["status"] = "completed"
    except BaseException as error:
        result.update(status="failed_preserved", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        if state is not None:
            result["peak_allocated_mib"] = state.torch.cuda.max_memory_allocated() / 2**20
            result["peak_reserved_mib"] = state.torch.cuda.max_memory_reserved() / 2**20
        write(output / "result.json", result)


if __name__ == "__main__":
    main()
