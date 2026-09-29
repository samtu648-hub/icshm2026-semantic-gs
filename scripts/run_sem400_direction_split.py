#!/usr/bin/env python3
"""SEM-400 deterministic semantic-gradient child-direction experiment."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sem398_boundary_split as base
from src.semantic_boundary_direction import deterministic_control_lines

CONFIG = ROOT / "experiments/runs/SEM-400-RGB152-SEMANTIC-DIRECTION-SPLIT/config.yaml"


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


class XyzDirectionView:
    """Allow diagnostic XYZ gradients while detaching every other geometry field."""

    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        if name == "get_xyz":
            return self.model.get_xyz
        if name in {"get_scaling", "get_rotation", "get_opacity"}:
            return getattr(self.model, name).detach()
        return getattr(self.model, name)


def direction_probability(state, logits, name: str):
    torch = state.torch
    probability = logits.softmax(1)
    geometry = XyzDirectionView(state.model)
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
        override_color=torch.cat(
            [probability[:, 3:], torch.zeros_like(probability[:, :1])], 1
        ).contiguous(),
    )["render"][:2]
    return torch.cat([first, last], 0)


def direction_loss(state, logits, name: str):
    if name not in state.semantic_names:
        raise RuntimeError(f"Direction evidence attempted forbidden label {name}")
    target = state.torch.from_numpy(state.labels[name].astype("int64")).cuda()
    probability = direction_probability(state, logits, name)
    return state.torch.nn.functional.nll_loss(
        probability.clamp_min(1e-8).log()[None], target[None]
    )


def collect_directions(state, output: Path, result: dict) -> None:
    torch = state.torch
    config = state.config
    payload = torch.load(output / "common.pt", map_location="cpu", weights_only=False)
    selected = payload["evidence"]["candidate"].long().cuda()
    logits = payload["logits"].cuda()
    logits.requires_grad_(False)
    state.model._xyz.requires_grad_(True)
    matrices = torch.zeros((len(selected), 3, 3), device="cuda", dtype=torch.float64)
    observations = torch.zeros(len(selected), device="cuda", dtype=torch.int32)
    started = time.monotonic()
    for index, name in enumerate(state.semantic_names, 1):
        loss = direction_loss(state, logits, name)
        gradient = torch.autograd.grad(loss, state.model._xyz, retain_graph=False)[0][selected].detach().double()
        base.require_finite(gradient)
        magnitude = torch.linalg.vector_norm(gradient, dim=1)
        active = magnitude[magnitude > 0]
        scale = active.median() if len(active) else magnitude.new_tensor(1.0)
        weight = magnitude / (magnitude + scale.clamp_min(torch.finfo(torch.float64).tiny))
        unit = gradient / magnitude.clamp_min(torch.finfo(torch.float64).tiny)[:, None]
        matrices.add_(weight[:, None, None] * torch.einsum("bi,bj->bij", unit, unit))
        observations.add_((magnitude > 0).to(observations.dtype))
        if index % 25 == 0 or index == len(state.semantic_names):
            print(json.dumps({"direction_evidence": index, "total": len(state.semantic_names)}), flush=True)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrices)
    candidate = eigenvectors[:, :, -1]
    control_np = deterministic_control_lines(len(selected), config["direction_control_seed"])
    control = torch.from_numpy(control_np).to(device="cuda", dtype=torch.float64)
    coherence = (eigenvalues[:, 2] - eigenvalues[:, 1]) / eigenvalues[:, 2].clamp_min(
        torch.finfo(torch.float64).tiny
    )
    nonzero = observations > 0
    dot = torch.sum(control * candidate, dim=1).abs().clamp(0, 1)
    angle = torch.rad2deg(torch.acos(dot))
    evidence_config = config["direction_evidence"]
    coverage = float(nonzero.float().mean())
    median_coherence = float(coherence.median())
    median_angle = float(angle.median())
    gates = {
        "nonzero_parent_fraction": coverage >= evidence_config["minimum_nonzero_parent_fraction"],
        "median_eigengap_fraction": median_coherence >= evidence_config["minimum_median_eigengap_fraction"],
        "median_control_candidate_line_angle_deg": median_angle
        >= evidence_config["minimum_median_control_candidate_line_angle_deg"],
    }
    if not all(gates.values()):
        raise RuntimeError(f"Direction evidence gate failed: {gates}")
    path = output / "directions.pt"
    base.save_torch_new(
        torch,
        path,
        {
            "selected": selected.cpu(),
            "control": control.float().cpu(),
            "candidate": candidate.float().cpu(),
            "eigenvalues": eigenvalues.float().cpu(),
            "observations": observations.cpu(),
        },
    )
    result.update(
        direction_seconds=time.monotonic() - started,
        direction_parent_count=len(selected),
        direction_nonzero_parent_fraction=coverage,
        direction_median_eigengap_fraction=median_coherence,
        direction_median_control_candidate_line_angle_deg=median_angle,
        direction_gates=gates,
        directions_sha256=sha256(path),
    )


def common(state, output: Path, result: dict) -> None:
    base.common(state, output, result)
    collect_directions(state, output, result)


def split_deterministic(state, logits, selected, directions):
    torch = state.torch
    from utils.general_utils import build_rotation

    model = state.model
    children = state.config["split_children"]
    if children != 2:
        raise ValueError("SEM-400 requires symmetric binary children")
    selected = selected.long()
    directions = directions.double()
    directions = directions / torch.linalg.vector_norm(directions, dim=1, keepdim=True).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    with torch.no_grad():
        scales = model.get_scaling[selected].double()
        rotations = build_rotation(model._rotation[selected]).double()
        local_line = torch.bmm(rotations.transpose(1, 2), directions[:, :, None]).squeeze(-1)
        radius = torch.rsqrt(
            torch.sum((local_line / scales.clamp_min(torch.finfo(torch.float64).tiny)) ** 2, dim=1)
        )
        offsets = radius[:, None] * directions
        repeated_offsets = torch.cat([offsets, -offsets], 0).to(model._xyz.dtype)
        new_xyz = model._xyz[selected].repeat(2, 1) + repeated_offsets
        new_scaling = model.scaling_inverse_activation(model.get_scaling[selected].repeat(2, 1) / 1.6)
        new_rotation = model._rotation[selected].repeat(2, 1)
        new_features_dc = model._features_dc[selected].repeat(2, 1, 1)
        new_features_rest = model._features_rest[selected].repeat(2, 1, 1)
        new_opacity = model._opacity[selected].repeat(2, 1)
        model.tmp_radii = torch.zeros(len(model.get_xyz), device="cuda")
        new_tmp_radii = model.tmp_radii[selected].repeat(2)
        model.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_tmp_radii,
        )
        prune = torch.cat(
            [
                torch.zeros(len(logits), device="cuda", dtype=torch.bool),
                torch.zeros(2 * len(selected), device="cuda", dtype=torch.bool),
            ]
        )
        prune[selected] = True
        model.prune_points(prune)
        keep = torch.ones(len(logits), device="cuda", dtype=torch.bool)
        keep[selected] = False
        inherited = torch.cat([logits[keep], logits[selected].repeat(2, 1)], 0)
    if len(inherited) != len(model.get_xyz):
        raise AssertionError("Deterministic split sidecar count mismatch")
    return inherited, radius.float()


def run_arm(state, output: Path, result: dict, arm: str, attempt: str) -> None:
    torch = state.torch
    config = state.config
    common_dir = ROOT / config["output"] / f"common-{attempt}"
    payload = torch.load(common_dir / "common.pt", map_location="cpu", weights_only=False)
    direction_payload = torch.load(common_dir / "directions.pt", map_location="cpu", weights_only=False)
    selected = direction_payload["selected"].cuda()
    expected = payload["evidence"]["candidate"].long()
    if not torch.equal(selected.cpu(), expected):
        raise AssertionError("Direction parents differ from sealed semantic-conflict parents")
    before = len(state.model.get_xyz)
    inherited, radii = split_deterministic(
        state,
        payload["logits"].cuda(),
        selected,
        direction_payload[arm].cuda(),
    )
    logits = torch.nn.Parameter(inherited)
    semantic_optimizer = torch.optim.Adam([logits], lr=config["semantic_lr"])
    names = base.schedule(state.rgb_names, config["restoration_steps"], config["seed"] + 2)
    write(output / "restoration-sampling.json", names)
    semantic_set = set(state.semantic_names)
    started = time.monotonic()
    with (output / "training.jsonl").open("x", encoding="utf-8") as log:
        for step, name in enumerate(names, 1):
            state.model.optimizer.zero_grad(set_to_none=True)
            semantic_optimizer.zero_grad(set_to_none=True)
            photometric = base.rgb_loss(state, name)
            loss = photometric
            semantic = None
            if name in semantic_set:
                semantic = base.semantic_loss(state, logits, name)
                loss = loss + config["semantic_weight"] * semantic
            loss.backward()
            base.require_finite(loss, logits)
            state.model.optimizer.step()
            if semantic is not None:
                semantic_optimizer.step()
            if step % 100 == 0 or step == len(names):
                row = {
                    "step": step,
                    "name": name,
                    "rgb_loss": float(photometric.detach()),
                    "semantic_nll": float(semantic.detach()) if semantic is not None else None,
                    "seconds": time.monotonic() - started,
                }
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
    endpoint = output / "endpoint.pt"
    base.save_torch_new(
        torch,
        endpoint,
        {
            "model": state.model.capture(),
            "logits": logits.detach(),
            "semantic_optimizer": semantic_optimizer.state_dict(),
            "arm": arm,
            "selected": selected.cpu(),
            "directions_sha256": sha256(common_dir / "directions.pt"),
        },
    )
    quantiles = torch.quantile(radii, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device="cuda")).tolist()
    result.update(
        arm=arm,
        before_points=before,
        after_points=len(logits),
        selected_parents=len(selected),
        offset_radius_quantiles=quantiles,
        restoration_steps=len(names),
        training_seconds=time.monotonic() - started,
        endpoint_sha256=sha256(endpoint),
        placement="symmetric_parent_ellipsoid_surface",
        semantic_geometry_route="detached_during_restoration",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["preflight", "common", "control", "candidate", "evaluate"], required=True)
    parser.add_argument("--attempt", default="r1")
    parser.add_argument("--source-attempt", default=None)
    args = parser.parse_args()
    config = read(CONFIG)
    output = ROOT / config["output"] / f"{args.stage}-{args.attempt}"
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "experiment_id": config["experiment_id"],
        "stage": args.stage,
        "status": "starting",
        "seed": config["seed"],
        "config_sha256": sha256(CONFIG),
        "script_sha256": sha256(Path(__file__)),
        "split_sha256": sha256(ROOT / config["split_path"]),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    write(output / "started.json", result)
    state = None
    started = time.monotonic()
    try:
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if args.stage == "evaluate":
                base.evaluate(config, output, result, args.source_attempt or args.attempt)
            else:
                state = base.setup(config)
                state.torch.cuda.reset_peak_memory_stats()
                if args.stage == "preflight":
                    base.preflight(state, output, result)
                elif args.stage == "common":
                    common(state, output, result)
                else:
                    run_arm(state, output, result, args.stage, args.source_attempt or args.attempt)
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
