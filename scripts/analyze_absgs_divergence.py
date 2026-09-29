#!/usr/bin/env python3
"""Compare observed trajectories and run frozen-state causal render probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from run_absgs_divergence import suite, output, guard, arm_config
from run_component_drop_extensions import gpu_lock
from src.quality_protocol import sha256, write_new_json, verify_hashes


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def compare():
    import numpy as np

    q = suite()
    arms = q["arms"]
    roots = [ROOT / arm_config(e)["training"]["output"] for e in arms]
    traces = [p / "trace" for p in roots]
    states = [{r["step"]: r for r in rows(p / "states.jsonl")} for p in traces]
    grads = [{r["step"]: r for r in rows(p / "gradients.jsonl")} for p in traces]
    cams = [read(p / "camera-trace.json") for p in traces]
    events = [rows(p / "events.jsonl") for p in traces]
    initial_equal = states[0][0]["parameters"] == states[1][0]["parameters"]
    result = {
        "initial_parameters_exact": initial_equal,
        "initial_rng_exact": states[0][0]["rng"] == states[1][0]["rng"],
        "runtime_exact": read(traces[0] / "runtime.json")
        == read(traces[1] / "runtime.json"),
        "camera_sequence_exact": cams[0] == cams[1],
        "loaded_image_tensors_exact": read(traces[0] / "input-tensor-hashes.json")
        == read(traces[1] / "input-tensor-hashes.json"),
        "camera_tensor_metadata_exact": read(traces[0] / "camera-tensor-metadata.json")
        == read(traces[1] / "camera-tensor-metadata.json"),
        "first_differences": {},
        "topology": [],
        "screen_pruning": {},
        "probe_curves": {},
    }
    for key in ("xyz_sha256", "opacity_sha256", "rng", "count"):
        diff = [
            s
            for s in sorted(states[0].keys() & states[1].keys())
            if states[0][s][key] != states[1][s][key]
        ]
        result["first_differences"]["state_" + key] = diff[0] if diff else None
    for key in (
        "xyz_grad_sha256",
        "opacity_grad_sha256",
        "screen_grad_sha256",
        "sampled_rgb_sha256",
        "rng",
    ):
        diff = [
            s
            for s in sorted(grads[0].keys() & grads[1].keys())
            if grads[0][s][key] != grads[1][s][key]
        ]
        result["first_differences"]["gradient_" + key] = diff[0] if diff else None
    topology_first = None
    for step in range(600, 15000, 100):
        for kind in ("clone", "split"):
            files = [p / f"events/{step:05d}-{kind}.npz" for p in traces]
            with np.load(files[0]) as a, np.load(files[1]) as b:
                pa, pb = a["parents"], b["parents"]
                same = np.array_equal(pa, pb)
                record = {
                    "step": step,
                    "kind": kind,
                    "parent_order_exact": same,
                    "children_a": len(pa),
                    "children_b": len(pb),
                    "parent_symmetric_difference": len(
                        set(pa.tolist()) ^ set(pb.tolist())
                    ),
                    "child_positions_exact": same
                    and np.array_equal(a["xyz"], b["xyz"]),
                }
                if not same and topology_first is None:
                    topology_first = {**record, "witnesses": []}
                    differing = sorted(set(pa.tolist()) ^ set(pb.tolist()))[:16]
                    scorefiles = [
                        np.load(p / f"events/{step:05d}-scores.npz") for p in traces
                    ]
                    key = "signed" if kind == "clone" else "absolute"
                    threshold = 0.0002 if kind == "clone" else 0.0004
                    for ident in differing:
                        witness = {"lineage_id": ident, "threshold": threshold}
                        for index, z in enumerate(scorefiles):
                            found = np.flatnonzero(z["ids"] == ident)
                            if len(found):
                                i = int(found[0])
                                witness["ab"[index]] = {
                                    "score": float(z[key][i]),
                                    "scale": float(z["scale"][i]),
                                    "selected": ident
                                    in set((pa if index == 0 else pb).tolist()),
                                }
                        topology_first["witnesses"].append(witness)
                    for z in scorefiles:
                        z.close()
                result["topology"].append(record)
    result["first_topology_selection_difference"] = topology_first
    prunes = [
        {(r["step"], r["file"]): r for r in ev if r["type"] == "prune"} for ev in events
    ]
    result["first_prune_difference"] = None
    for key in sorted(prunes[0].keys() & prunes[1].keys()):
        with np.load(traces[0] / key[1]) as a, np.load(traces[1] / key[1]) as b:
            if not np.array_equal(a["removed_ids"], b["removed_ids"]):
                result["first_prune_difference"] = {
                    "step": key[0],
                    "phase": prunes[0][key]["phase"],
                    "removed_a": len(a["removed_ids"]),
                    "removed_b": len(b["removed_ids"]),
                }
                break
    for arm, p, ev in zip(arms, traces, events):
        pre = [r for r in ev if r["type"] == "densify_begin" and r["screen_threshold"]]
        app = [r for r in ev if r["type"] == "append"]
        result["screen_pruning"][arm] = {
            "active_threshold_events": len(pre),
            "events_with_oversized_screen_points_before_append": sum(
                r["screen_big_before_append"] > 0 for r in pre
            ),
            "oversized_point_event_incidences": sum(
                r["screen_big_before_append"] for r in pre
            ),
            "all_append_calls_zero_max_radii": all(
                r["max_radii_nonzero_after_append"] == 0 for r in app
            ),
            "total_append_calls": len(app),
            "actual_final_prune_screen_big_incidences": sum(
                r["reasons"]["screen_big_at_actual_prune"]
                for r in ev
                if r["type"] == "prune" and r.get("reasons") is not None
            ),
        }
        result["probe_curves"][arm] = rows(p / "probes.jsonl")
    for p in roots:
        r = read(p / "result.json")
        verify_hashes(p, r["artifacts"])
        verify_hashes(p, r["trace_hashes"])
    return result


def setup(artifact_root):
    import run_sem396_spatial as runtime

    _, torch, np = runtime.setup_runtime(artifact_root)
    torch.set_num_threads(4)
    from arguments import PipelineParams

    p = argparse.ArgumentParser()
    pipe = PipelineParams(p).extract(p.parse_args([]))
    return torch, np, pipe


def repeated_backward(run, step, artifact_root):
    from src.absgs_trajectory import tensor_hash, rng_hashes

    torch, np, pipe = setup(artifact_root)
    from gaussian_renderer import GaussianModel, render
    from PIL import Image
    from utils.loss_utils import l1_loss, ssim

    try:
        from fused_ssim import fused_ssim
    except ImportError:
        fused_ssim = None
    trace = run / "trace"
    snap = trace / f"snapshots/step-{step:05d}"
    model = GaussianModel(3)
    model.load_ply(str(snap / "point_cloud.ply"))
    info = read(snap / "state.json")
    model.active_sh_degree = info["active_sh_degree"]
    capture = torch.load(snap / "capture.pt", map_location="cpu", weights_only=False)
    camrow = read(trace / "camera-trace.json")[step]
    name = camrow["camera"]
    m = read(trace / "camera-tensor-metadata.json")[name]
    camera = SimpleNamespace(
        image_width=m["image_width"],
        image_height=m["image_height"],
        FoVx=m["FoVx"],
        FoVy=m["FoVy"],
        image_name=name,
    )
    for k in (
        "world_view_transform",
        "projection_matrix",
        "full_proj_transform",
        "camera_center",
    ):
        setattr(camera, k, torch.tensor(m[k], dtype=torch.float32, device="cuda"))
    image_name = name if Path(name).suffix else name + ".png"
    photo = np.asarray(
        Image.open(ROOT / suite()["scene"] / "images" / image_name).convert("RGB")
    )
    gt = torch.from_numpy(photo.copy()).permute(2, 0, 1).float() / 255
    tensor_exact = tensor_hash(gt) == read(trace / "input-tensor-hashes.json")[name]
    if not tensor_exact:
        raise ValueError("Loaded frozen RGB tensor differs from original training")
    gt = gt.cuda()
    bg = torch.zeros(3, device="cuda")
    packet = render(camera, model, pipe, bg, **m["render_options"])
    image = packet["render"]
    dssim = (
        fused_ssim(image[None], gt[None]) if fused_ssim is not None else ssim(image, gt)
    )
    loss = 0.8 * l1_loss(image, gt) + 0.2 * (1 - dssim)
    upstream = torch.autograd.grad(loss, image)[0].detach()
    del packet, image, loss, dssim
    parameter_before = {
        k: tensor_hash(getattr(model, k))
        for k in (
            "_xyz",
            "_opacity",
            "_scaling",
            "_rotation",
            "_features_dc",
            "_features_rest",
        )
    }
    records = []
    reference = None
    selections = []
    for trial in range(suite()["repeated_backward_count"]):
        random.setstate(capture["python_rng"])
        np.random.set_state(capture["numpy_rng"])
        torch.set_rng_state(capture["torch_rng"])
        torch.cuda.set_rng_state(capture["cuda_rng"])
        for k in parameter_before:
            getattr(model, k).grad = None
        before = rng_hashes()
        p = render(camera, model, pipe, bg, **m["render_options"])
        (p["render"] * upstream).sum().backward()
        xyz = model._xyz.grad.detach().cpu()
        screen = p["viewspace_points"].grad.detach().cpu()
        if not torch.isfinite(xyz).all() or not torch.isfinite(screen).all():
            raise FloatingPointError("Nonfinite frozen backward")
        row = {
            "trial": trial,
            "rgb_sha256": tensor_hash(p["render"]),
            "xyz_gradient_sha256": tensor_hash(xyz),
            "screen_gradient_sha256": tensor_hash(screen),
            "rng_unchanged": before == rng_hashes(),
        }
        if reference is None:
            reference = (xyz.clone(), screen.clone())
        row["xyz_max_abs_difference_from_first"] = float(
            (xyz - reference[0]).abs().max()
        )
        row["screen_max_abs_difference_from_first"] = float(
            (screen - reference[1]).abs().max()
        )
        if step == 599:
            signed = capture["model"][8].cuda().clone()
            absolute = capture["model"][9].cuda().clone()
            denom = capture["model"][10].cuda().clone()
            visible = p["visibility_filter"]
            signed[visible] += torch.norm(
                p["viewspace_points"].grad[visible, :2], dim=-1, keepdim=True
            )
            absolute[visible] += torch.norm(
                p["viewspace_points"].grad[visible, 2:], dim=-1, keepdim=True
            )
            denom[visible] += 1
            small = (
                model.get_scaling.max(1).values
                <= info["training_args"]["percent_dense"] * capture["model"][12]
            )
            clone = (
                torch.nan_to_num(signed / denom, nan=0.0).flatten() >= 0.0002
            ) & small
            split = (
                torch.nan_to_num(absolute / denom, nan=0.0).flatten() >= 0.0004
            ) & ~small
            selected = {
                "clone": clone.detach().cpu().numpy(),
                "split": split.detach().cpu().numpy(),
            }
            if selections:
                row["selection_differences_from_first"] = {
                    k: int(np.count_nonzero(v != selections[0][k]))
                    for k, v in selected.items()
                }
            selections.append(selected)
            row["clone_count"], row["split_count"] = int(clone.sum()), int(split.sum())
        records.append(row)
    if parameter_before != {
        k: tensor_hash(getattr(model, k)) for k in parameter_before
    }:
        raise ValueError("Frozen probe changed parameters")
    return {
        "step": step,
        "camera": image_name,
        "input_tensor_exact": tensor_exact,
        "parameters_unchanged": True,
        "fixed_upstream_gradient_sha256": tensor_hash(upstream),
        "loss_implementation": "fused_ssim"
        if fused_ssim is not None
        else "native_ssim",
        "trials": records,
        "forward_exact": len({r["rgb_sha256"] for r in records}) == 1,
        "backward_exact": len({r["screen_gradient_sha256"] for r in records}) == 1,
    }


def contributions(run, step, artifact_root):
    from src.absgs_trajectory import tensor_hash

    torch, np, pipe = setup(artifact_root)
    from gaussian_renderer import GaussianModel, render
    from src.quality_cameras import read_camera_requests
    from src.quality_native_gaussian import make_native_view
    from PIL import Image

    trace = run / "trace"
    snap = trace / f"snapshots/step-{step:05d}"
    model = GaussianModel(3)
    model.load_ply(str(snap / "point_cloud.ply"))
    model.active_sh_degree = read(snap / "state.json")["active_sh_degree"]
    for k in (
        "_xyz",
        "_opacity",
        "_scaling",
        "_rotation",
        "_features_dc",
        "_features_rest",
    ):
        getattr(model, k).requires_grad_(False)
    ids = np.load(snap / "ids.npy")
    cameras = read_camera_requests(ROOT / suite()["scene"] / "sparse/0")
    with np.load(trace / "lineage.npz") as z:
        parent = {
            int(i): (int(p), int(b), int(k))
            for i, p, b, k in zip(z["ids"], z["parents"], z["birth_step"], z["kind"])
        }
    destination = output() / "contributions" / run.parent.name / f"step-{step:05d}"
    destination.mkdir(parents=True, exist_ok=False)
    result = []
    bg = torch.zeros(3, device="cuda")
    original_opacity = model._opacity.detach().clone()
    opacity_sha = tensor_hash(model._opacity)
    for name in suite()["contribution_views"]:
        camera = cameras[name]
        view = make_native_view(camera)
        with torch.no_grad():
            baseline = render(view, model, pipe, bg)["render"].detach()
        to_image = lambda value: np.rint(
            value.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
        ).astype(np.uint8)
        baseline_image = to_image(baseline)
        if not np.array_equal(
            baseline_image, np.asarray(Image.open(snap / name).convert("RGB"))
        ):
            raise ValueError("Frozen milestone PLY RGB replay differs")
        target = np.asarray(
            Image.open(ROOT / suite()["scene"] / "images" / name).convert("RGB")
        )

        def psnr(image):
            return float(
                -10
                * np.log10(
                    max(
                        np.mean(((image.astype(np.float64) - target) / 255) ** 2), 1e-30
                    )
                )
            )

        colors = torch.full((len(ids), 3), 0.5, device="cuda", requires_grad=True)
        p = render(view, model, pipe, bg, override_color=colors)
        weights = torch.autograd.grad(p["render"].mean(), colors)[0].sum(1)
        indices = torch.argsort(weights, descending=True)[:64]
        indices_cpu = indices.detach().cpu().numpy()
        total = float(weights.sum())
        top = []
        for index in indices_cpu:
            ident = int(ids[index])
            lineage = []
            current = ident
            while current in parent:
                par, birth, kind = parent[current]
                lineage.append(
                    {"id": current, "parent": par, "birth_step": birth, "kind": kind}
                )
                current = par
                if len(lineage) > 1000:
                    raise ValueError("Lineage cycle")
            xyz = model._xyz[index].cpu().numpy()
            depth = float((camera.pose.rotation @ xyz + camera.pose.translation)[2])
            top.append(
                {
                    "index": int(index),
                    "id": ident,
                    "initial_ancestor": current,
                    "ancestry": lineage,
                    "mean_contribution": float(weights[index]),
                    "fraction_of_mass": float(weights[index]) / max(total, 1e-30),
                    "depth": depth,
                    "radius_px": int(p["radii"][index]),
                    "opacity": float(model.get_opacity[index, 0]),
                    "scales": model.get_scaling[index].cpu().tolist(),
                    "xyz": xyz.tolist(),
                }
            )
        removals = []
        for count in suite()["counterfactual_top_k"]:
            with torch.no_grad():
                model._opacity[indices[:count]] = -30
                changed = to_image(render(view, model, pipe, bg)["render"])
                model._opacity.copy_(original_opacity)
                restored = to_image(render(view, model, pipe, bg)["render"])
            if (
                not np.array_equal(restored, baseline_image)
                or tensor_hash(model._opacity) != opacity_sha
            ):
                raise ValueError("Counterfactual restoration differs")
            Image.fromarray(changed).save(
                destination / f"{Path(name).stem}-remove-{count}.png"
            )
            removals.append(
                {
                    "top_k": count,
                    "psnr_db": psnr(changed),
                    "delta_psnr_db": psnr(changed) - psnr(baseline_image),
                    "restored_exact": True,
                }
            )
        Image.fromarray(baseline_image).save(
            destination / f"{Path(name).stem}-baseline.png"
        )
        result.append(
            {
                "view": name,
                "step": step,
                "baseline_psnr_db": psnr(baseline_image),
                "total_mean_alpha_mass": total,
                "top_contributors": top,
                "removals": removals,
            }
        )
        del colors, p, weights, baseline
    return result


def analyze(artifact_root):
    guard()
    result = {"status": "starting"}
    start = time.monotonic()
    try:
        result["trajectory_comparison"] = compare()
        result["numerical_probes"] = {}
        result["contributions"] = {}
        with gpu_lock():
            for eid in suite()["arms"]:
                run = ROOT / arm_config(eid)["training"]["output"]
                result["numerical_probes"][eid] = [
                    repeated_backward(run, step, artifact_root)
                    for step in suite()["repeated_backward_steps"]
                ]
                result["contributions"][eid] = []
                for step in suite()["contribution_steps"]:
                    result["contributions"][eid].extend(
                        contributions(run, step, artifact_root)
                    )
                    print(
                        json.dumps({"arm": eid, "contribution_step": step}), flush=True
                    )
        guard()
        result["status"] = "completed"
    except BaseException as e:
        result.update(status="failed_preserved", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        result["seconds"] = time.monotonic() - start
        write_new_json(output() / "analysis-result.json", result)
    comparison = result["trajectory_comparison"]
    print(
        json.dumps(
            {
                "first_differences": comparison["first_differences"],
                "first_topology": comparison["first_topology_selection_difference"],
                "screen_pruning": comparison["screen_pruning"],
            }
        ),
        flush=True,
    )
