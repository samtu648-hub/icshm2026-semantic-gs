#!/usr/bin/env python3
"""CPU acceptance of SEM341 training journal, roles, probability files and two fresh replays."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_mask2former_training import Context, stability_passes
from src.quality_protocol import sha256, verify_hashes, write_new_json


def audit_journal(run, result, cfg):
    names = result["access"]["train"]
    rng = random.Random(cfg["seed"] + int(result["partition"][-1])*100003)
    digest, seen, skips, steps = hashlib.sha256(), set(), 0, 0
    with (run / "training.jsonl").open() as stream:
        for steps, line in enumerate(stream, 1):
            row = json.loads(line)
            trace = {"name": names[rng.randrange(len(names))], "top": rng.randint(0, 989-768),
                     "left": rng.randint(0, 1320-768), "flip": rng.random() < .5}
            if row["step"] != steps or row["sample"] != trace:
                raise ValueError("Training sample/crop sequence not exact registered B2 order")
            digest.update(json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()+b"\n")
            seen.add(trace["name"])
            expected_factor = ((6000-(steps-1))/6000)**.9
            if abs(row["lr_factor"]-expected_factor) > 2e-15:
                raise ValueError("Schedule advanced on attempted rather than successful steps")
            if not np.isfinite([row["loss"], row["grad_norm"], row["scale"], *row["losses"].values()]).all():
                raise ValueError("Nonfinite effective training update")
            for attempt, skip in enumerate(row["skips"], 1):
                if (skip["attempt"] != attempt or skip["scale_after"] != skip["scale_before"]*.5
                        or skip["bad_gradient_count"] <= 0):
                    raise ValueError("Overflow backoff record differs")
            if len(row["skips"]) >= cfg["training"]["amp_max_attempts_per_step"]:
                raise ValueError("Unbounded/unsuccessful AMP update")
            skips += len(row["skips"])
    if (steps != result["trained_steps"] or skips != result["amp_skips"]
            or digest.hexdigest() != result["sampling_sha256"] or len(seen) != result["unique_sampled_views"]):
        raise ValueError("Training journal/summary mismatch")
    if sum(result["sampled_crop_class_counts"]) != steps*768*768:
        raise ValueError("Effective-update crop class accounting differs")
    return {"successful_steps": steps, "skips": skips, "unique_views": len(seen), "sampling_sha256": digest.hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("artifact-root", "run", "replay-1", "replay-2", "output"):
        p.add_argument("--"+n, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rec = {"status": "auditing", "quality_metrics": None, "target_labels_decoded": False,
           "new_training": False, "script_sha256": sha256(Path(__file__))}
    try:
        result = json.loads((a.run / "result.json").read_text())
        if result["mode"] not in ("stability", "full"):
            raise ValueError("Expected original training run")
        ctx = Context(ROOT, a.artifact_root, result["partition"])
        expect = "completed_mask2former_fit_pending_replay" if result["mode"] == "full" else "completed_mask2former_stability_pending_replay"
        steps = 6000 if result["mode"] == "full" else 128
        if (result["status"] != expect or result["trained_steps"] != steps or result["binding"] != ctx.binding
                or result["access"] != ctx.access or result["runtime"] != ctx.probe["runtime"]
                or result["training_label_views"] != ctx.access["train"]
                or result["target_labels_decoded"] or result["rgb_model_imported"]):
            raise ValueError("Unaccepted or leaking complete training record")
        verify_hashes(a.run, result["artifacts"])
        if result["mode"] == "stability":
            stability_passes(result, ctx.cfg)
        if (result["pretrained_load"]["loaded_tensors"] != 647 or not result["pretrained_load"]["all_loaded_tensors_exact"]
                or result["parameters"] != 215451514 or result["trainable_parameters"] != result["parameters"]):
            raise ValueError("Model/pretraining capacity mismatch")
        expected_checkpoints = {"3000", "6000"} if result["mode"] == "full" else {"128"}
        if set(result["checkpoints"]) != expected_checkpoints or any(not v["rng_preserved"] for v in result["checkpoints"].values()):
            raise ValueError("Stage checkpoints not retained without RNG change")
        journal = audit_journal(a.run, result, ctx.cfg)
        labels = ctx.labels()
        expected_label_hashes = {n: labels.records[n]["label_png_sha256"] for n in ctx.access["train"]}
        if result["train_label_hashes"] != expected_label_hashes:
            raise ValueError("Actual training label sources differ")
        verify_hashes(ctx.artifact_root / "r2/sem_001/labels", expected_label_hashes)
        expected_views = ctx.access["prediction"] if result["mode"] == "full" else [ctx.access["train"][i] for i in [0, len(ctx.access["train"])//2, -1]]
        if result["prediction_views"] != expected_views or len(set(expected_views)) != len(expected_views):
            raise ValueError("Predicted view role/coverage differs")
        repeat_hashes, pids = {}, set()
        for folder in (a.replay_1, a.replay_2):
            r = json.loads((folder / "result.json").read_text())
            if (r["status"] != "completed_mask2former_fresh_replay" or r["trained_steps"] != 0
                    or r["run_result_sha256"] != sha256(a.run/"result.json") or r["binding"] != result["binding"]
                    or r["runtime"] != result["runtime"] or r["prediction_views"] != expected_views
                    or r["compared_views"] != len(expected_views) or not r["probabilities_exact"] or r["target_labels_decoded"]):
                raise ValueError("Incomplete/different fresh replay")
            pids.add(r["pid"])
            repeat_hashes[str((folder/"result.json").resolve().relative_to(ROOT))] = sha256(folder/"result.json")
        if len(pids) != 2 or result["pid"] in pids:
            raise ValueError("Fresh independent processes required")
        for name in expected_views:
            with np.load(a.run / "probabilities" / (Path(name).stem+".npz"), allow_pickle=False) as z:
                p = z["probabilities"]
            with Image.open(a.run / "predictions" / name) as im:
                png = np.asarray(im)
            if (p.dtype != np.float32 or p.shape != (5, 989, 1320) or not np.isfinite(p).all()
                    or p.min() < 0 or p.max() > 1 or np.max(np.abs(p.astype(np.float64).sum(0)-1)) > 2e-6
                    or png.dtype != np.uint8 or not np.array_equal(p.argmax(0), png)):
                raise ValueError("Independent native probability/PNG validation failed")
        ctx.finish()
        rec.update(status="completed_mask2former_stability_acceptance" if result["mode"] == "stability" else "completed_mask2former_fold_acceptance",
                   experiment_id=ctx.cfg["experiment_id"], partition=result["partition"], binding=ctx.binding,
                   run=str(a.run.resolve().relative_to(ROOT)), run_result_sha256=sha256(a.run/"result.json"),
                   replay_result_hashes=repeat_hashes, two_fresh_exact=True, prediction_views=expected_views,
                   journal=journal, frozen_artifacts_verified=len(ctx.frozen), rgb_exports_unchanged=len(ctx.rgb_hashes))
    except Exception as exc:
        rec.update(status="failed_mask2former_acceptance", error=repr(exc))
        raise
    finally:
        rec["seconds"] = time.monotonic()-started
        write_new_json(a.output/"result.json", rec)


if __name__ == "__main__":
    main()
