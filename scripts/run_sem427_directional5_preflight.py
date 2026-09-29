#!/usr/bin/env python3
"""Run the SEM427 frozen-RGB directional semantic engineering preflight."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_sem425_direct5_preflight import (  # noqa: E402
    class_weights,
    fresh_rgb,
    make_crop_view,
    rgb_state_sha256,
    roles,
    setup_runtime,
)
from src.quality_feature_splatting import crop_and_flip  # noqa: E402
from src.quality_native_gaussian import make_native_view  # noqa: E402
from src.quality_protocol import sha256, verify_hashes  # noqa: E402
from src.quality_view_conditioned_semantics import (  # noqa: E402
    build_directional_logit_field,
    directional_logits,
    directional_logit_sha256,
    render_directional_probabilities,
)


EID = "SEM-427-RGB152-DIR5-PF"
CONFIG = ROOT / "experiments/runs" / EID / "config.yaml"
CODE_PATHS = (
    "scripts/run_sem427_directional5_preflight.py",
    "scripts/run_sem425_direct5_preflight.py",
    "src/quality_view_conditioned_semantics.py",
    "src/quality_feature_splatting.py",
    "src/quality_native_gaussian.py",
    "src/quality_cameras.py",
    "src/quality_protocol.py",
)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def replace(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def semantic_crop(runtime, coefficients, sample):
    probability = render_directional_probabilities(
        runtime.render,
        make_crop_view(runtime, sample),
        runtime.model,
        runtime.pipe,
        coefficients,
    )
    return runtime.torch.flip(probability, dims=(-1,)) if sample["flip"] else probability


def verify_binding(config, binding):
    if sha256(CONFIG) != binding["config_sha256"]:
        raise ValueError("SEM427 config changed after binding")
    if sha256(ROOT / config["split_path"]) != binding["split_sha256"]:
        raise ValueError("SEM427 split changed after binding")
    if sha256(ROOT / config["parent_sampling_plan"]) != binding["sampling_plan_sha256"]:
        raise ValueError("SEM426 sample plan changed after binding")
    verify_hashes(ROOT, binding["code_sha256"])


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
        parent_audit_path = ROOT / config["parent_audit"]
        parent_audit = read(parent_audit_path)
        if sha256(parent_audit_path) != config["parent_audit_sha256"] or parent_audit.get("status") != "passed":
            raise ValueError("A byte-matched passed SEM426 audit is required")
        training, confirmation, buffer_names = roles(config)
        parent_plan_path = ROOT / config["parent_sampling_plan"]
        if sha256(parent_plan_path) != config["parent_sampling_plan_sha256"]:
            raise ValueError("SEM426 sampling plan differs from registered hash")
        parent_samples = read(parent_plan_path)["samples"]
        plan = parent_samples[: config["preflight"]["successful_updates"]]
        if len(plan) != 8 or any(row["name"] not in set(training) for row in plan):
            raise ValueError("SEM427 did not receive eight training-only SEM426 samples")
        write(output / "sampling-plan.json", {"schema_version": 1, "experiment_id": EID, "samples": plan})
        binding = {
            "schema_version": 1,
            "experiment_id": EID,
            "config_sha256": sha256(CONFIG),
            "split_sha256": sha256(ROOT / config["split_path"]),
            "sampling_plan_sha256": sha256(parent_plan_path),
            "local_sampling_plan_sha256": sha256(output / "sampling-plan.json"),
            "parent_audit_sha256": sha256(parent_audit_path),
            "training_views": training,
            "confirmation_views_sealed": confirmation,
            "buffer_views_sealed": buffer_names,
            "code_sha256": {path: sha256(ROOT / path) for path in CODE_PATHS},
        }
        write(output / "binding.json", binding)
        verify_binding(config, binding)
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            runtime = setup_runtime(config, args.artifact_root.resolve())
            runtime.torch.cuda.reset_peak_memory_stats()
            label_root = ROOT / config["label_root"]
            counts, weights_np, label_hashes = class_weights(runtime, training, label_root)
            manifest = read(ROOT / config["label_manifest"])
            registered = {row["image"]: row["label_png_sha256"] for row in manifest["records"]}
            if any(label_hashes[name] != registered.get(name) for name in training):
                raise ValueError("A SEM427 training label differs from the frozen manifest")
            binding["training_label_hashes"] = label_hashes
            replace(output / "binding.json", binding)
            verify_binding(config, binding)
            before_state = rgb_state_sha256(runtime.model)
            fixed_name = training[0]
            before_rgb = fresh_rgb(runtime, fixed_name)
            coefficients = build_directional_logit_field(
                runtime.model.get_xyz.shape[0], device="cuda", dtype=runtime.model.get_xyz.dtype
            )
            initial_hash = directional_logit_sha256(coefficients)
            optimizer = runtime.torch.optim.Adam(
                [coefficients],
                lr=config["preflight"]["learning_rate"],
                weight_decay=config["preflight"]["weight_decay"],
            )
            weights = runtime.torch.from_numpy(weights_np).cuda()
            rows = []
            probability_error = 0.0
            for sample in plan:
                optimizer.zero_grad(set_to_none=True)
                probability = semantic_crop(runtime, coefficients, sample)
                label = runtime.np.asarray(runtime.Image.open(label_root / sample["name"]), dtype=runtime.np.uint8)
                target_np = crop_and_flip(label, sample)
                target = runtime.torch.from_numpy(target_np.astype("int64")).cuda()
                loss = runtime.torch.nn.functional.nll_loss(
                    probability.clamp_min(runtime.torch.finfo(probability.dtype).tiny).log()[None],
                    target[None],
                    weight=weights,
                )
                loss.backward()
                gradient = coefficients.grad
                if gradient is None or not runtime.torch.isfinite(gradient).all():
                    raise FloatingPointError("SEM427 coefficient gradient is missing or non-finite")
                bias_norm = float(gradient[:, :, 0].norm())
                direction_norm = float(gradient[:, :, 1:].norm())
                if bias_norm <= 0 or direction_norm <= 0:
                    raise ValueError("SEM427 bias or directional gradient is zero")
                step_error = float((probability.sum(dim=0) - 1).abs().max())
                probability_error = max(probability_error, step_error)
                optimizer.step()
                if not runtime.torch.isfinite(coefficients).all():
                    raise FloatingPointError("SEM427 coefficients became non-finite")
                rows.append({
                    "step": int(sample["step"]),
                    "sample": sample,
                    "loss": float(loss.detach()),
                    "bias_gradient_norm": bias_norm,
                    "direction_gradient_norm": direction_norm,
                    "probability_sum_max_error": step_error,
                })
            checkpoint = output / "directional-logits-step8.pt"
            runtime.torch.save({
                "schema_version": 1,
                "experiment_id": EID,
                "step": 8,
                "binding_sha256": sha256(output / "binding.json"),
                "directional_coefficients": coefficients.detach().cpu(),
            }, checkpoint)
            replay = build_directional_logit_field(
                runtime.model.get_xyz.shape[0], device="cuda", dtype=runtime.model.get_xyz.dtype
            )
            payload = runtime.torch.load(checkpoint, map_location="cpu", weights_only=True)
            with runtime.torch.no_grad():
                replay.copy_(payload["directional_coefficients"].cuda())
                reference_probability = semantic_crop(runtime, coefficients, plan[0])
                replay_probability = semantic_crop(runtime, replay, plan[0])
                view_a = make_native_view(runtime.cameras[training[0]])
                view_b = make_native_view(runtime.cameras[training[-1]])
                gaussian_probability_a = runtime.torch.softmax(directional_logits(coefficients, view_a, runtime.model), dim=1)
                gaussian_probability_b = runtime.torch.softmax(directional_logits(coefficients, view_b, runtime.model), dim=1)
                cross_delta = (gaussian_probability_a - gaussian_probability_b).abs()
            checkpoint_replay = directional_logit_sha256(coefficients) == directional_logit_sha256(replay) and runtime.torch.equal(reference_probability, replay_probability)
            after_state = rgb_state_sha256(runtime.model)
            after_rgb = fresh_rgb(runtime, fixed_name)
            rgb_exact = before_state == after_state and runtime.torch.equal(before_rgb, after_rgb)
            peak_reserved = runtime.torch.cuda.max_memory_reserved() / 2**20
            gates = {
                "eight_updates": len(rows) == 8,
                "finite_nonzero_bias_gradients": all(row["bias_gradient_norm"] > config["preflight"]["minimum_bias_gradient_norm"] for row in rows),
                "finite_nonzero_direction_gradients": all(row["direction_gradient_norm"] > config["preflight"]["minimum_direction_gradient_norm"] for row in rows),
                "cross_camera_probability_response": float(cross_delta.max()) > config["preflight"]["minimum_cross_camera_probability_delta"],
                "probability_conservation": probability_error <= config["preflight"]["probability_sum_max_error"],
                "checkpoint_replay_exact": checkpoint_replay,
                "frozen_rgb_exact": rgb_exact,
                "memory_within_12gib": peak_reserved < config["preflight"]["maximum_peak_reserved_mib"],
                "confirmation_and_buffer_sealed": True,
            }
            prediction = reference_probability.argmax(dim=0).cpu().numpy().astype(runtime.np.uint8)
            runtime.Image.fromarray(prediction, mode="L").save(output / "training-probe-semantic.png")
            result.update(
                status="completed_passed" if all(gates.values()) else "completed_gate_failed",
                gate_pass=gates,
                successful_updates=len(rows),
                point_count=int(coefficients.shape[0]),
                trainable_parameter_count=int(coefficients.numel()),
                initial_directional_logit_sha256=initial_hash,
                final_directional_logit_sha256=directional_logit_sha256(coefficients),
                checkpoint_sha256=sha256(checkpoint),
                checkpoint_replay_exact=checkpoint_replay,
                rgb_state_hashes_equal=before_state == after_state,
                fresh_rgb_exact=runtime.torch.equal(before_rgb, after_rgb),
                frozen_rgb_max_absolute_pixel_difference=int((before_rgb.to(runtime.torch.int16) - after_rgb.to(runtime.torch.int16)).abs().max()),
                probability_sum_max_error=probability_error,
                cross_camera_probe={
                    "camera_a": training[0],
                    "camera_b": training[-1],
                    "mean_absolute_probability_delta": float(cross_delta.mean()),
                    "max_absolute_probability_delta": float(cross_delta.max()),
                    "fraction_above_1e-6": float((cross_delta > 1e-6).float().mean()),
                },
                class_pixel_counts=counts.tolist(),
                class_weights=weights_np.tolist(),
                updates=rows,
                peak_allocated_mib=runtime.torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=peak_reserved,
                binding_sha256=sha256(output / "binding.json"),
                training_views_accessed=sorted({row["sample"]["name"] for row in rows}),
            )
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        replace(output / "result.json", result)


if __name__ == "__main__":
    main()
