#!/usr/bin/env python3
"""Two-step native SEM340 probe or independent checkpoint replay, never quality fit."""
from __future__ import annotations
import argparse
from collections import OrderedDict
import fcntl
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json

CONFIG = "experiments/runs/SEM-340-MASK2FORMER-PREFLIGHT/config.yaml"
CODE = ["scripts/run_quality_mask2former_preflight.py", "src/quality_mask2former.py",
        "src/quality_experts.py", "src/quality_protocol.py", "src/quality_static_training.py",
        "src/semantic_cable_binary.py", "src/semantic2d.py"]


def source_manifest(runtime, config):
    result = {}
    for folder, key in [("Mask2Former", "mask2former"), ("detectron2", "detectron2")]:
        path = runtime / "vendor" / folder
        if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip() != config["upstream"][key + "_commit"]:
            raise ValueError("Upstream version changed")
        names = subprocess.check_output(["git", "ls-files"], cwd=path, text=True).splitlines()
        for name in names:
            item = path / name
            if item.is_file():
                result[str(item.relative_to(runtime))] = sha256(item)
    for path in (runtime / "runtime").rglob("*"):
        if path.is_file() and (path.suffix in (".py", ".so") or path.name in ("METADATA", "RECORD")):
            result[str(path.relative_to(runtime))] = sha256(path)
    for name in ["vendor/detectron2/detectron2", "vendor/Mask2Former/mask2former/modeling/pixel_decoder/ops"]:
        binaries = list((runtime / name).glob("*.so"))
        if len(binaries) != 1:
            raise ValueError("Expected one newly built native extension")
        result[str(binaries[0].relative_to(runtime))] = sha256(binaries[0])
    return result


def operator_test():
    """Native CUDA values/gradients against author's independent PyTorch core."""
    import torch
    from mask2former.modeling.pixel_decoder.ops.functions.ms_deform_attn_func import (
        MSDeformAttnFunction, ms_deform_attn_core_pytorch)
    torch.manual_seed(3)
    shapes = torch.tensor([[6, 4], [3, 2]], dtype=torch.long, device="cuda")
    start = torch.tensor([0, 24], dtype=torch.long, device="cuda")
    value = torch.rand(1, 30, 2, 4, device="cuda", requires_grad=True)
    loc = torch.rand(1, 3, 2, 2, 2, 2, device="cuda", requires_grad=True)
    weight = torch.rand(1, 3, 2, 2, 2, device="cuda", requires_grad=True)
    native = MSDeformAttnFunction.apply(value, shapes, start, loc, weight, 1)
    ref = ms_deform_attn_core_pytorch(value, shapes, loc, weight)
    torch.testing.assert_close(native, ref, atol=1e-5, rtol=1e-5)
    ng = torch.autograd.grad(native.sum(), (value, loc, weight), retain_graph=True)
    rg = torch.autograd.grad(ref.sum(), (value, loc, weight))
    for x, y in zip(ng, rg):
        torch.testing.assert_close(x, y, atol=1e-4, rtol=1e-4)
    return {"forward_max_abs": float((native-ref).detach().abs().max()),
            "gradient_max_abs": [float((x-y).abs().max()) for x, y in zip(ng, rg)], "passed": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime-root", type=Path, required=True)
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--activation-checkpoint", action="store_true")
    p.add_argument("--replay-of", type=Path)
    a = p.parse_args()
    a.runtime_root = a.runtime_root.resolve()
    a.output = a.output.resolve()
    if not a.output.is_relative_to(a.runtime_root):
        p.error("Outputs must stay in the isolated experiment root")
    with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a.output.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        record = {"experiment_id": "SEM-340-MASK2FORMER-PREFLIGHT", "status": "initializing",
                  "stage": "input_audit", "trained_steps": 0, "target_labels_decoded": False,
                  "quality_metrics": None, "full_training": False, "rgb_model_imported": False,
                  "activation_checkpoint": a.activation_checkpoint, "artifacts": {}}
        frozen_hashes, rgb_hashes = {}, {}
        torch = None
        try:
            import numpy as np
            from src.quality_static_training import semantic_access, TrainingLabels
            from src.quality_experts import verify_export
            from src.semantic_cable_binary import read_rgb, random_native_pair
            from src.quality_mask2former import config_for_model, load_official_model, author_optimizer, native_predict, class_masks
            cfg = json.loads((ROOT / CONFIG).read_text())
            for name in ["dataset_manifest", "semantic_split", "oof_split", "label_manifest", "rgb_result"]:
                verify_hashes(ROOT, {cfg[name]: cfg[name + "_sha256"]})
            if shutil.disk_usage(a.runtime_root).free < cfg["resource"]["minimum_free_disk_gib"] * 1024**3:
                raise ValueError("Disk reserve under100GiB")
            split = json.loads((ROOT / cfg["semantic_split"]).read_text())
            oof = json.loads((ROOT / cfg["oof_split"]).read_text())
            rgb_result = ROOT / cfg["rgb_result"]
            parent = json.loads(rgb_result.read_text())
            verify_hashes(rgb_result.parent, parent["artifacts"])
            access = semantic_access(split, oof, json.loads((rgb_result.parent / "access.json").read_text()), "oof-0")
            name = access["train"][0]
            export_root = ROOT / cfg["rgb_export"]
            verify_hashes(export_root, {"result.json": cfg["rgb_export_result_sha256"]})
            export = json.loads((export_root / "result.json").read_text())
            export = verify_export(export_root, split["semantic_train_labeled"], cfg["rgb_result_sha256"], export["identity"], "semantic-training")
            rgb_hashes = {e["name"]: e["sha256"] for e in export["rendered_images"]}
            frozen = json.loads((ROOT / "experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json").read_text())
            frozen_hashes = {e["path"]: e["sha256"] for e in frozen["artifacts"]}
            verify_hashes(a.artifact_root, frozen_hashes)
            binding = {"config_sha256": sha256(ROOT / CONFIG), "code_sha256": {n: sha256(ROOT / n) for n in CODE},
                       "upstream_runtime_sha256": source_manifest(a.runtime_root, cfg),
                       "rgb_identity": export["identity"], "frozen_rgb_signature": export["frozen_rgb_signature"],
                       "export_sha256": cfg["rgb_export_result_sha256"], "pixel_space": "PINHOLE",
                       "view": name, "rgb_sha256": rgb_hashes[name], "seed": cfg["seed"]}
            record.update(binding=binding, access=access, training_label_views=[], git_commit=cfg["git_commit"])
            snapshot = a.output / "code"
            for n in CODE + [CONFIG]:
                dest = snapshot / n
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / n, dest)
            sys.path[:0] = [str(a.runtime_root / n) for n in ["runtime", "vendor/detectron2", "vendor/Mask2Former",
                                            "vendor/Mask2Former/mask2former/modeling/pixel_decoder/ops"]]
            import torch
            from detectron2.structures import Instances
            if not torch.cuda.is_available():
                raise RuntimeError("GPU inaccessible in this process; not a hardware failure verdict")
            record["stage"] = "operator_validation"
            record["operator"] = operator_test()
            random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
            torch.cuda.manual_seed_all(cfg["seed"])
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            record["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                                 "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
                                 "versions": {k: importlib.metadata.version(k) for k in ["torchvision", "timm", "fvcore", "iopath", "yacs", "omegaconf", "hydra-core"]}}
            model_cfg = config_for_model(a.runtime_root / "vendor/Mask2Former", cfg["upstream"]["config"], a.activation_checkpoint)
            (a.output / "resolved_model.yaml").write_text(model_cfg.dump())
            record["stage"] = "pretrained_load"
            model, load = load_official_model(model_cfg, a.runtime_root / "weights/model_final_6b4a3a.pkl")
            record["pretrained_load"] = load
            record["parameters"] = sum(x.numel() for x in model.parameters())
            record["trainable_parameters"] = sum(x.numel() for x in model.parameters() if x.requires_grad)
            if record["parameters"] != record["trainable_parameters"]:
                raise ValueError("Expert backbone unexpectedly frozen")
            if a.replay_of:
                previous = json.loads((a.replay_of / "result.json").read_text())
                if (previous["status"] != "completed_two_step_probe_pending_fresh" or previous["binding"] != binding
                        or previous["activation_checkpoint"] != a.activation_checkpoint):
                    raise ValueError("Incompatible probe/replay binding")
                verify_hashes(a.replay_of, previous["artifacts"])
                saved = torch.load(a.replay_of / "model.pt", map_location="cpu", weights_only=True)
                if saved["binding"] != binding:
                    raise ValueError("Checkpoint bound to different RGB/code/space")
                model.load_state_dict(saved["model"], strict=True)
                del saved
            model.cuda()
            torch.cuda.reset_peak_memory_stats()
            rgb = read_rgb(export_root / "images" / name)
            if not a.replay_of:
                labels = TrainingLabels(a.artifact_root / "r2/sem_001/labels", [name],
                                        json.loads((ROOT / cfg["label_manifest"]).read_text()), cfg["resolution"])
                label = labels.read(name)
                record["training_label_views"] = sorted(labels.accessed)
                record["training_label_sha256"] = labels.records[name]["label_png_sha256"]
                optimizer = author_optimizer(model, model_cfg)
                scaler = torch.amp.GradScaler("cuda", init_scale=cfg["preflight"]["amp_initial_scale"])
                rng = random.Random(cfg["seed"])
                record["steps"] = []
                record["amp_skips"] = []
                for step in range(cfg["preflight"]["steps"]):
                    model.train(); optimizer.zero_grad(set_to_none=True)
                    x, y = random_native_pair(rgb, label, 768, rng)
                    ids, masks = class_masks(y)
                    instances = Instances((768, 768))
                    instances.gt_classes = torch.from_numpy(ids)
                    instances.gt_masks = torch.from_numpy(masks)
                    image = torch.from_numpy(x.transpose(2, 0, 1).copy()).float()
                    torch.cuda.synchronize(); step_started = time.monotonic()
                    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
                    for attempt in range(cfg["preflight"]["amp_max_attempts_per_step"]):
                        optimizer.zero_grad(set_to_none=True)
                        torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng)
                        record["stage"] = "train_forward"
                        with torch.autocast("cuda", dtype=torch.float16):
                            losses = model([{"image": image, "instances": instances}])
                            loss = sum(losses.values())
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite native loss")
                        record["stage"] = "train_backward"
                        scaler.scale(loss).backward()
                        record["stage"] = "optimizer"
                        scaler.unscale_(optimizer)
                        bad = [n for n, v in model.named_parameters() if v.grad is not None and not torch.isfinite(v.grad).all()]
                        if bad:
                            before = scaler.get_scale()
                            # unscale_ recorded found_inf; native scaler skips this
                            # step. Do not clip infinities or change model params.
                            scaler.step(optimizer); scaler.update()
                            record["amp_skips"].append({"step": step+1, "attempt": attempt+1,
                                "scale_before": before, "scale_after": scaler.get_scale(),
                                "nonfinite_gradient_parameters": bad, "loss": float(loss.detach())})
                            if scaler.get_scale() >= before:
                                raise FloatingPointError("AMP did not back off invalid gradients")
                            del losses, loss
                            continue
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .01, error_if_nonfinite=True)
                        scaler.step(optimizer); scaler.update()
                        break
                    else:
                        raise FloatingPointError("Bounded AMP backoff exhausted; no successful update")
                    torch.cuda.synchronize()
                    record["trained_steps"] += 1
                    record["steps"].append({"step": step+1, "seconds": time.monotonic()-step_started,
                        "loss": float(loss.detach()), "grad_norm": float(norm), "scale": scaler.get_scale(),
                        "components": {k: float(v.detach()) for k, v in losses.items()}})
                    print(json.dumps({"step": step+1, "seconds": record["steps"][-1]["seconds"],
                                      "peak_mib": torch.cuda.max_memory_allocated()/1024**2}), flush=True)
                    del losses, loss
                record["training_peak_allocated_mib"] = torch.cuda.max_memory_allocated()/1024**2
                record["training_peak_reserved_mib"] = torch.cuda.max_memory_reserved()/1024**2
                optimizer.zero_grad(set_to_none=True)
                del optimizer, scaler
                record["stage"] = "save_checkpoint"
                original_state = model.state_dict()
                state = OrderedDict((k, v.detach().cpu()) for k, v in original_state.items())
                state._metadata = original_state._metadata
                if any(not torch.isfinite(v).all() for v in state.values()):
                    raise FloatingPointError("Nonfinite saved state")
                torch.save({"model": state, "binding": binding}, a.output / "model.pt")
                del state, original_state
                record["artifacts"]["model.pt"] = sha256(a.output / "model.pt")
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            record["stage"] = "native_inference"
            pred_started = time.monotonic()
            prob = native_predict(model, rgb)
            record["inference_seconds"] = time.monotonic()-pred_started
            record["inference_peak_allocated_mib"] = torch.cuda.max_memory_allocated()/1024**2
            np.save(a.output / "probability.npy", prob, allow_pickle=False)
            record["artifacts"]["probability.npy"] = sha256(a.output / "probability.npy")
            if a.replay_of:
                ref = np.load(a.replay_of / "probability.npy", allow_pickle=False)
                record["replay_max_abs"] = float(np.max(np.abs(prob-ref)))
                if not np.array_equal(prob, ref):
                    raise ValueError("Fresh native probability replay not exact")
                record["status"] = "completed_independent_native_replay"
            else:
                record["status"] = "completed_two_step_probe_pending_fresh"
            record["stage"] = "completed"
        except Exception as exc:
            oom = torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError)
            record.update(status="failed_cuda_oom" if oom else "failed_engineering", error=repr(exc))
            (a.output / "failure.txt").write_text(traceback.format_exc())
            raise
        finally:
            record["seconds"] = time.monotonic()-started
            if torch is not None and torch.cuda.is_initialized():
                record["last_peak_allocated_mib"] = torch.cuda.max_memory_allocated()/1024**2
                record["last_peak_reserved_mib"] = torch.cuda.max_memory_reserved()/1024**2
            try:
                if frozen_hashes:
                    verify_hashes(a.artifact_root, frozen_hashes)
                    record["frozen_artifacts_verified"] = len(frozen_hashes)
                if rgb_hashes:
                    verify_hashes(export_root / "images", rgb_hashes)
                    record["rgb_exports_unchanged"] = len(rgb_hashes)
            except Exception as exc:
                record.update(status="failed_immutability", immutable_error=repr(exc))
            write_new_json(a.output / "result.json", record)


if __name__ == "__main__":
    main()
