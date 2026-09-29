#!/usr/bin/env python3
"""SEM368: fixed scale trial, original native samples, no evaluation labels in training."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_mask2former_training import amp_update, poly_factor, sampled_crop, save_checkpoint, stability_passes
from src.quality_expert_scale import CONFIG, CODE, Context, resize_pair, native_predict
from src.quality_protocol import sha256, verify_hashes, write_new_json


def require_stability(folder, context):
    audit = json.loads((folder / "result.json").read_text())
    if audit["status"] != "completed_mask2former_stability_acceptance" or not audit["two_fresh_exact"]:
        raise ValueError("Full run requires independent longer stability acceptance")
    run = ROOT / audit["run"]
    verify_hashes(run, {"result.json": audit["run_result_sha256"]})
    result = json.loads((run / "result.json").read_text())
    stability_passes(result, context.cfg)
    if dict(result["binding"], partition=context.binding["partition"]) != context.binding:
        raise ValueError("Stability source/config/runtime differs from full run")
    verify_hashes(run, result["artifacts"])
    for relative, digest in audit["replay_result_hashes"].items():
        verify_hashes(ROOT, {relative: digest})
    return sha256(folder / "result.json")


def train(a, ctx, rec):
    import numpy as np
    import torch
    from PIL import Image
    from detectron2.structures import Instances
    from src.semantic_cable_binary import read_rgb
    from src.quality_mask2former import load_official_model, author_optimizer, class_masks
    if a.mode == "full":
        rec["stability_audit_sha256"] = require_stability(a.stability_audit, ctx)
    t = ctx.cfg["training"]
    cfg = ctx.model_config()
    (a.output / "resolved_model.yaml").write_text(cfg.dump())
    model, loaded = load_official_model(cfg, ctx.runtime / "weights/model_final_6b4a3a.pkl")
    rec.update(pretrained_load=loaded, parameters=sum(v.numel() for v in model.parameters()),
               trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad))
    if rec["parameters"] != 215451514 or rec["trainable_parameters"] != rec["parameters"]:
        raise ValueError("Full registered Swin-L expert required")
    model.cuda()
    optimizer = author_optimizer(model, cfg)
    base_rates = [g["lr"] for g in optimizer.param_groups]
    scaler = torch.amp.GradScaler("cuda", init_scale=t["amp_initial_scale"])
    labels = ctx.labels()
    counts = labels.audit_counts()
    if any(n <= 0 for n in counts):
        raise ValueError("Each training fold needs all five classes")
    names = ctx.access["train"]
    images = {n: read_rgb(ctx.export_root / "images" / n) for n in names}
    targets = {n: labels.read(n) for n in names}
    rec.update(training_label_views=sorted(labels.accessed), class_counts=counts.tolist(),
               train_label_hashes={n: labels.records[n]["label_png_sha256"] for n in names},
               class_weights="author set loss/no-object weights; training counts not converted to inverse-frequency CE")
    ctx.guard()
    fold = int(a.partition[-1])
    rng = random.Random(ctx.cfg["seed"] + fold*100003)
    steps = t["steps"] if a.mode == "full" else t["stability_steps"]
    rec["requested_steps"] = steps
    seen, class_counts = set(), np.zeros(5, np.int64)
    sampling = hashlib.sha256()
    rec["amp_skips"], rec["checkpoints"] = 0, {}
    model.train()
    torch.cuda.reset_peak_memory_stats()
    start = previous = time.monotonic()
    with (a.output / "training.jsonl").open("x") as log:
        for step in range(1, steps+1):
            x, y, trace = sampled_crop(rng, names, images, targets, t["crop"])
            sampling.update(json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()+b"\n")
            seen.add(trace["name"]); class_counts += np.bincount(y.ravel(), minlength=5)
            if trace != ctx.baseline_trace[step-1]:
                raise ValueError("Native sample differs from completed SEM341 baseline")
            image, scaled_y = resize_pair(x, y)
            ids, masks = class_masks(scaled_y)
            instances = Instances((864, 864))
            instances.gt_classes = torch.from_numpy(ids)
            instances.gt_masks = torch.from_numpy(masks)
            factor = poly_factor(step-1, t["steps"], t["poly_power"])
            for group, rate in zip(optimizer.param_groups, base_rates):
                group["lr"] = rate * factor
            rec["stage"] = "train_update"
            row = amp_update(model, optimizer, scaler, image, instances,
                             max_attempts=t["amp_max_attempts_per_step"], clip=t["clip_global_norm"])
            rec["trained_steps"] = step
            rec["amp_skips"] += len(row["skips"])
            row.update(step=step, sample=trace, lr_factor=factor, elapsed_seconds=time.monotonic()-start)
            log.write(json.dumps(row)+"\n")
            if a.mode == "full" and step == 3000:
                ctx.guard()
                rec["checkpoints"][str(step)] = save_checkpoint(model, ctx, step, a.output / "model-3000.pt")
            if step == 1 or step % ctx.cfg["resource"]["logging_every_steps"] == 0 or step == steps:
                torch.cuda.synchronize(); now = time.monotonic()
                progress = {"step": step, "total": steps, "partition": a.partition, "loss": row["loss"],
                    "amp_skips": rec["amp_skips"], "scale": scaler.get_scale(), "elapsed_seconds": now-start,
                    "interval_seconds": now-previous, "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
                    "sampling_sha256": sampling.hexdigest(), "free_gib": shutil.disk_usage(ROOT).free/1024**3, "network_input": 864, "unix": time.time()}
                write_new_json(a.output / f"progress-{step:05d}.json", progress)
                log.flush(); print(json.dumps(progress), flush=True); previous = now
            if step % ctx.cfg["resource"]["guard_every_steps"] == 0:
                ctx.guard()
    torch.cuda.synchronize()
    rec.update(training_seconds=time.monotonic()-start, peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
               peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2, sampling_sha256=sampling.hexdigest(),
               unique_sampled_views=len(seen), sampled_crop_class_counts=class_counts.tolist())
    rec["stage"] = "save_final"
    ctx.guard()
    rec["checkpoints"][str(steps)] = save_checkpoint(model, ctx, steps, a.output / "model.pt")
    optimizer.zero_grad(set_to_none=True)
    del optimizer, scaler, images, targets
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    views = ctx.access["prediction"] if a.mode == "full" else [names[0], names[len(names)//2], names[-1]]
    rec["prediction_views"] = views
    (a.output / "probabilities").mkdir(); (a.output / "predictions").mkdir()
    rec["stage"] = "native_inference"
    infer_start = time.monotonic()
    for i, name in enumerate(views):
        ctx.guard()
        prob = native_predict(model, read_rgb(ctx.export_root / "images" / name))
        np.savez_compressed(a.output / "probabilities" / (Path(name).stem+".npz"), probabilities=prob)
        Image.fromarray(prob.argmax(0).astype(np.uint8)).save(a.output / "predictions" / name)
        if i % 16 == 0:
            print(json.dumps({"inference_views": i+1, "total": len(views), "partition": a.partition}), flush=True)
    rec.update(inference_seconds=time.monotonic()-infer_start,
               inference_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
               status="completed_mask2former_fit_pending_replay" if a.mode == "full" else "completed_mask2former_stability_pending_replay")
    if a.mode == "stability":
        stability_passes(rec, ctx.cfg)


def replay(a, ctx, rec):
    import numpy as np
    import torch
    from PIL import Image
    from src.quality_mask2former import load_official_model
    from src.quality_semantic_audit import check_replay
    from src.semantic_cable_binary import read_rgb
    result = json.loads((a.run / "result.json").read_text())
    if (result["status"] not in ("completed_mask2former_fit_pending_replay", "completed_mask2former_stability_pending_replay")
            or result["binding"] != ctx.binding or result["access"] != ctx.access
            or result["runtime"] != rec["runtime"]):
        raise ValueError("Incomplete or mismatched fitted model")
    verify_hashes(a.run, result["artifacts"])
    expected = ctx.access["prediction"] if result["mode"] == "full" else [ctx.access["train"][i] for i in [0, len(ctx.access["train"])//2, -1]]
    if result["prediction_views"] != expected:
        raise ValueError("Replay view role coverage differs")
    model, _ = load_official_model(ctx.model_config(), ctx.runtime / "weights/model_final_6b4a3a.pkl")
    payload = torch.load(a.run / "model.pt", map_location="cpu", weights_only=True)
    if payload["binding"] != ctx.binding or payload["access"] != ctx.access or payload["steps"] != result["trained_steps"]:
        raise ValueError("Checkpoint provenance mismatch")
    model.load_state_dict(payload["model"], strict=True)
    del payload
    model.cuda(); torch.cuda.reset_peak_memory_stats()
    rec.update(run_result_sha256=sha256(a.run / "result.json"), prediction_views=expected, compared_views=0)
    for i, name in enumerate(expected):
        ctx.guard()
        prob = native_predict(model, read_rgb(ctx.export_root / "images" / name))
        with np.load(a.run / "probabilities" / (Path(name).stem+".npz"), allow_pickle=False) as z:
            stored = z["probabilities"]
        with Image.open(a.run / "predictions" / name) as im:
            png = np.array(im)
        check_replay(prob, stored, png)
        rec["compared_views"] += 1
        if i % 16 == 0:
            print(json.dumps({"replay_views": i+1, "total": len(expected)}), flush=True)
    rec.update(status="completed_mask2former_fresh_replay", probabilities_exact=True,
               peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--partition", choices=("oof-0", "oof-1", "oof-2"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("stability", "full", "replay"), required=True)
    p.add_argument("--run", type=Path)
    p.add_argument("--stability-audit", type=Path)
    a = p.parse_args()
    if (a.mode == "replay") != (a.run is not None) or (a.mode == "full") != (a.stability_audit is not None):
        p.error("Replay requires run; full requires stability audit; no other combinations")
    if a.mode == "stability" and a.partition != "oof-0":
        p.error("Longer stability uses registered fold0 only")
    a.output = a.output.resolve()
    if not a.output.is_relative_to(ROOT / "outputs/quality/expert_scale"):
        p.error("Use a new isolated quality output")
    with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a.output.mkdir(parents=True, exist_ok=False)
        start = time.monotonic()
        rec = {"experiment_id": "SEM-368-EXPERT-INPUT-SCALE", "status": "initializing", "stage": "input_audit",
               "mode": a.mode, "partition": a.partition, "trained_steps": 0, "pid": os.getpid(),
               "target_labels_decoded": False, "rgb_model_imported": False, "quality_metrics": None,
               "automatic_promotion": False, "artifacts": {}}
        ctx = None
        try:
            ctx = Context(ROOT, a.artifact_root, a.partition)
            rec.update(binding=ctx.binding, access=ctx.access, git_commit=ctx.cfg["git_commit"], seed=ctx.cfg["seed"])
            for n in (*CODE, CONFIG):
                dest = a.output / "code" / n
                dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/n, dest)
            write_new_json(a.output / "started.json", rec)
            rec["runtime"] = ctx.setup_runtime()
            rec["stage"] = "model_execution"
            (replay if a.mode == "replay" else train)(a, ctx, rec)
            ctx.finish()
            rec.update(stage="completed", frozen_artifacts_verified=len(ctx.frozen), rgb_exports_unchanged=len(ctx.rgb_hashes),
                       artifacts={str(v.relative_to(a.output)): sha256(v) for v in a.output.rglob("*")
                                  if v.is_file() and "code" not in v.relative_to(a.output).parts})
        except BaseException as exc:
            rec.update(status="failed_mask2former_stage_not_method_rejection", error=f"{type(exc).__name__}: {exc}")
            (a.output / "failure.txt").write_text(traceback.format_exc())
            if ctx is not None:
                try:
                    ctx.finish(); rec.update(frozen_artifacts_verified=len(ctx.frozen), rgb_exports_unchanged=len(ctx.rgb_hashes))
                except Exception as immutable_error:
                    rec["immutable_error"] = repr(immutable_error)
            raise
        finally:
            rec.update(seconds=time.monotonic()-start, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            write_new_json(a.output / "result.json", rec)
            print(json.dumps({"status": rec["status"], "output": str(a.output)}), flush=True)


if __name__ == "__main__":
    main()
