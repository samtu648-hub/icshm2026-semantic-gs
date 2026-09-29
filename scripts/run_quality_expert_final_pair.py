#!/usr/bin/env python3
"""SEM362: final219 paired B2 endpoints; fixed SEM336 optimization."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import fcntl
from importlib.metadata import version
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
from src.quality_expert_final_pair import open_control, check_binding, save_endpoint, enforce_transfer_budget
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.quality_static_training import TrainingLabels, semantic_access
from src.quality_experts import verify_export, expert_counts_weights, target_for_expert, load_expert, predict

CONFIG = "experiments/runs/SEM-362-B2-PAIRED-FINAL/config.yaml"
CODE = ("scripts/run_quality_expert_final_pair.py", "src/quality_expert_final_pair.py", "src/quality_experts.py", "src/quality_protocol.py",
        "src/quality_static_training.py", "src/semantic_cable_binary.py", "src/semantic2d.py")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("rgb-run", "rendered-root", "artifact-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--target-root", type=Path)
    p.add_argument("--expert", choices=("fiveclass",), required=True)
    p.add_argument("--partition", choices=("development",), required=True)
    p.add_argument("--mode", choices=("preflight", "full"), required=True)
    p.add_argument("--preflight", type=Path)
    p.add_argument("--deadline",type=float,required=True)
    a = p.parse_args()
    if (a.mode == "full") != (a.preflight is not None):
        p.error("Full fitting requires a matching CPU preflight")
    if (a.partition == "development") != (a.target_root is not None):
        p.error("Only development fitting requires the separate rendered developer dataset")
    output = a.output.resolve()
    with ((ROOT / "outputs/quality/gpu.lock").open("a") if a.mode != "preflight" else nullcontext()) as lock:
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        record = {"experiment_id": "SEM-362-B2-PAIRED-FINAL", "status": "initializing", "mode": a.mode,
                  "trained_steps": 0, "target_labels_decoded": False, "real_rgb_decoded": False,
                  "rgb_imported_or_optimized": False, "quality_metrics": None, "artifacts": {},
                  "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "code_sha256": {n: sha256(ROOT / n) for n in CODE}}
        try:
            import numpy as np
            config = json.loads((ROOT / CONFIG).read_text())
            control, control_path = open_control(ROOT, config, a.partition, a.expert)
            if a.rgb_run.resolve() != (ROOT / config["study"]["rgb_run"]).resolve() or a.rendered_root.resolve() != (ROOT / config["study"]["rendered_root"]).resolve():
                raise ValueError("Duration study requires its exact registered RGB export")
            if a.target_root.resolve() != (ROOT/config["study"]["target_root"]).resolve():
                raise ValueError("Final target export differs from registered path")
            record["historical_reference_result_sha256"] = sha256(control_path / "result.json")
            protocol = json.loads((ROOT / config["protocol"]).read_text())
            verify_hashes(ROOT, protocol["input_hashes"])
            parent = json.loads((a.rgb_run / "result.json").read_text())
            if parent["experiment_id"] not in config["allowed_rgb_parents"] or parent["mode"] != "full" or parent["seed"] != 0:
                raise ValueError("Unregistered RGB parent")
            verify_hashes(a.rgb_run, parent["artifacts"])
            split = json.loads((ROOT / "experiments/splits/sem_001_scaffold_static_v1.json").read_text())
            oof = json.loads((ROOT / "experiments/splits/sem_005_oof_spatial_v1.json").read_text())
            access = semantic_access(split, oof, json.loads((a.rgb_run / "access.json").read_text()), a.partition)
            export = json.loads((a.rendered_root / "result.json").read_text())
            exported = verify_export(a.rendered_root, split["semantic_train_labeled"], sha256(a.rgb_run / "result.json"), export["identity"], "semantic-training")
            if a.target_root:
                target_export = verify_export(a.target_root, split["semantic_validation_labeled"], sha256(a.rgb_run / "result.json"), export["identity"], "development")
                if target_export["frozen_rgb_signature"] != export["frozen_rgb_signature"]:
                    raise ValueError("Train/developer renders differ in RGB state")
            frozen = json.loads((ROOT / protocol["frozen_manifest"]).read_text())
            frozen_hashes = {i["path"]: i["sha256"] for i in frozen["artifacts"]}
            verify_hashes(a.artifact_root, frozen_hashes)
            manifest = json.loads((ROOT / config["label_manifest"]).read_text())
            if sha256(ROOT / config["label_manifest"]) != oof["label_manifest_sha256"]:
                raise ValueError("Label manifest differs from OOF protocol")
            labels = TrainingLabels(a.artifact_root / config["labels"], access["train"], manifest, config["resolution"])
            expert = config["experts"][a.expert]
            positive = expert.get("positive_class")
            counts, weights = expert_counts_weights(labels.audit_counts(), positive)
            model_dir = a.artifact_root / config["model"]["relative_weights"]
            model_hashes = {n: sha256(model_dir / n) for n in ("config.json", "pytorch_model.bin")}
            binding = {"config_sha256": sha256(ROOT / CONFIG), "input_hashes": protocol["input_hashes"],
                "parent_result_sha256": sha256(a.rgb_run / "result.json"), "identity": export["identity"],
                "frozen_rgb_signature": exported["frozen_rgb_signature"], "parent_seed": parent["seed"],
                "train_export_sha256": sha256(a.rendered_root / "result.json"),
                "target_export_sha256": sha256(a.target_root / "result.json") if a.target_root else None,
                "model_hashes": model_hashes, "expert": a.expert, "seed": config["seed"], "pixel_space": "PINHOLE"}
            record.update(binding=binding, access=access, class_counts=counts.tolist(), class_weights=weights.tolist(),
                train_label_hashes={n: labels.records[n]["label_png_sha256"] for n in access["train"]},
                runtime={k: version(k) for k in ("torch", "transformers", "numpy")}, frozen_verified=len(frozen_hashes))
            check_binding(record, control)
            for name in (*CODE, CONFIG):
                dest = output / "code" / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, dest)
            write_new_json(output / "started.json", record)
            if a.mode == "preflight":
                record["status"] = "passed_expert_cpu_preflight"
                return
            if a.mode == "full":
                smoke = json.loads((a.preflight / "result.json").read_text())
                if (smoke.get("status") != "passed_expert_cpu_preflight" or smoke["binding"] != binding
                        or smoke["access"] != access or smoke["code_sha256"] != record["code_sha256"]):
                    raise ValueError("Smoke differs in expert, parent, partition, code or recipe")
                verify_hashes(a.preflight, smoke["artifacts"])
                record["smoke_result_sha256"] = sha256(a.preflight / "result.json")
            enforce_transfer_budget(ROOT,config,pending_bytes=256*2**20)
            if time.time()>a.deadline: raise TimeoutError('Registered final GPU-stage deadline')
            import torch
            import torch.nn.functional as F
            from PIL import Image
            from src.semantic_cable_binary import read_rgb, random_native_pair, normalize_rgb
            seed = config["seed"]
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            record["amp"] = str(amp)
            model = load_expert(model_dir, seed, positive).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
            scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)
            weight = torch.as_tensor(weights, device="cuda")
            fold = 0 if a.partition == "development" else int(a.partition[-1])
            rng = random.Random(seed + fold * 100003)
            names = access["train"]
            # CPU caching changes I/O only, not crop selection or image resolution.
            images = {n: read_rgb(a.rendered_root / "images" / n) for n in names}
            targets = {n: target_for_expert(labels.read(n), positive) for n in names}
            steps = config["training"]["smoke_steps" if a.mode == "smoke" else "steps"]
            crop = expert["crop"]
            losses = []
            torch.cuda.reset_peak_memory_stats()
            fit_started = time.monotonic()
            model.train()
            journal=(output/"training_samples.jsonl").open("x")
            for step in range(1, steps + 1):
                if time.time()>a.deadline: raise TimeoutError("Registered final GPU-stage deadline")
                name = names[rng.randrange(len(names))]
                probe=random.Random();probe.setstate(rng.getstate())
                trace=dict(name=name,top=probe.randint(0,989-crop),left=probe.randint(0,1320-crop),flip=probe.random()<.5)
                rgb, target = random_native_pair(images[name], targets[name], crop, rng)
                if rng.getstate()!=probe.getstate():raise ValueError("Sampling journal changed RNG calls")
                journal.write(json.dumps(dict(step=step,sample=trace),sort_keys=True)+"\n")
                x, y = normalize_rgb(rgb)[None].cuda(), torch.from_numpy(target.astype(np.int64))[None].cuda()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=amp):
                    logits = F.interpolate(model(pixel_values=x).logits, size=(crop, crop), mode="bilinear", align_corners=False)
                    loss = F.cross_entropy(logits.float(), y, weight=weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at step {step}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError(f"Nonfinite gradient at step {step}; no update")
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach()))
                record["trained_steps"] = step
                if step == 3000 and a.mode == "full":
                    enforce_transfer_budget(ROOT,config,pending_bytes=256*2**20)
                    record["paired_control_3000"] = save_endpoint(model, binding, access, 3000, output / "expert-3000.pt")
                    write_new_json(output / "endpoint-3000.json", record["paired_control_3000"])
                    print(json.dumps({"paired_control_saved": record["paired_control_3000"], "partition": a.partition}), flush=True)
                if step == 1 or step % 100 == 0 or step == steps:
                    print(json.dumps({"step": step, "steps": steps, "expert": a.expert, "partition": a.partition,
                                      "loss": losses[-1], "elapsed_seconds": time.monotonic() - fit_started}), flush=True)
            journal.close()
            torch.cuda.synchronize()
            record.update(training_seconds=time.monotonic() - fit_started,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
                loss={"first": losses[0], "last": losses[-1], "mean": float(np.mean(losses))})
            enforce_transfer_budget(ROOT,config,pending_bytes=256*2**20)
            record["final_endpoint"] = save_endpoint(model, binding, access, steps, output / f"expert-{steps}.pt")
            del optimizer, model, scaler, images, targets
            torch.cuda.empty_cache()
            views = access["prediction"] if a.mode == "full" else access["prediction"][:1]
            endpoints = [3000, 6000] if a.mode == "full" else [steps]
            record["endpoints"] = endpoints
            for endpoint in endpoints:
                reloaded = load_expert(model_dir, seed, positive).cuda()
                reloaded.load_state_dict(torch.load(output / f"expert-{endpoint}.pt", map_location="cpu", weights_only=True)["state_dict"])
                dest = output / f"step-{endpoint}"
                (dest / "probabilities").mkdir(parents=True)
                (dest / "predictions").mkdir()
                for name in views:
                    if time.time()>a.deadline:raise TimeoutError("Registered final GPU-stage deadline")
                    enforce_transfer_budget(ROOT,config,pending_bytes=32*2**20)
                    probability = predict(reloaded, read_rgb(a.target_root / "images" / name), crop=crop, stride=expert["stride"], amp_dtype=amp)
                    np.savez_compressed(dest / "probabilities" / (Path(name).stem + ".npz"), probabilities=probability)
                    Image.fromarray(probability.argmax(0).astype(np.uint8)).save(dest / "predictions" / name)
                del reloaded
                torch.cuda.empty_cache()
            verify_hashes(a.rgb_run, parent["artifacts"])
            verify_hashes(a.artifact_root, frozen_hashes)
            verify_hashes(model_dir, model_hashes)
            verify_hashes(ROOT, record["code_sha256"])
            verify_hashes(ROOT, protocol["input_hashes"])
            if sha256(ROOT / CONFIG) != binding["config_sha256"]:
                raise ValueError("Experiment config changed during fitting")
            verify_export(a.rendered_root, split["semantic_train_labeled"], binding["parent_result_sha256"], binding["identity"], "semantic-training")
            if a.target_root:
                verify_export(a.target_root, split["semantic_validation_labeled"], binding["parent_result_sha256"], binding["identity"], "development")
            enforce_transfer_budget(ROOT,config)
            record.update(status="completed_expert_fit_pending_independent_audit",
                prediction_views=views, rgb_artifacts_unchanged=True,
                artifacts={str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file() and "code" not in p.relative_to(output).parts and p.name != "started.json"})
        except BaseException as exc:
            record.update(status="expert_fit_failed_not_method_rejection", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["seconds"] = time.monotonic() - started
            write_new_json(output / "result.json", record)
            print(json.dumps({"status": record["status"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
