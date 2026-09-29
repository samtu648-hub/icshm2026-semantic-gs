"""SEM341 training contracts, immutable sources, native sampling and safe AMP."""
from collections import OrderedDict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import sys

import numpy as np
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.quality_static_training import semantic_access, TrainingLabels
from src.quality_experts import verify_export
from src.quality_mask2former import config_for_model

CONFIG = "experiments/runs/SEM-341-MASK2FORMER-OOF/config.yaml"
CODE = ("src/quality_mask2former_training.py", "src/quality_mask2former.py", "src/quality_protocol.py",
        "src/quality_static_training.py", "src/quality_experts.py", "src/quality_semantic_audit.py",
        "src/semantic_cable_binary.py", "src/semantic2d.py", "scripts/run_quality_mask2former_preflight.py",
        "scripts/run_quality_mask2former_oof.py")


def poly_factor(completed, total=6000, power=.9):
    if not isinstance(completed, int) or not 0 <= completed <= total or total <= 0 or power <= 0:
        raise ValueError("Invalid successful-update schedule position")
    return (1 - completed / total) ** power


def sampled_crop(rng, names, images, targets, crop=768):
    """Same RNG calls/order as SEM336; coordinates journaled, no quality selection."""
    from src.semantic_cable_binary import random_native_pair
    name = names[rng.randrange(len(names))]
    probe = random.Random()
    probe.setstate(rng.getstate())
    trace = {"name": name, "top": probe.randint(0, 989-crop),
             "left": probe.randint(0, 1320-crop), "flip": probe.random() < .5}
    image, label = random_native_pair(images[name], targets[name], crop, rng)
    if probe.getstate() != rng.getstate():
        raise ValueError("Sampling differs from SEM336 crop implementation")
    return image, label, trace


def validate_training_config(c):
    t = c["training"]
    if (c["experiment_id"] != "SEM-341-MASK2FORMER-OOF" or c["resolution"] != [1320, 989]
            or c["pixel_space"] != "PINHOLE" or c["partitions"] != ["oof-0", "oof-1", "oof-2"]
            or [t[k] for k in ["steps", "stability_steps", "batch_size", "gradient_accumulation", "crop", "stride"]]
            != [6000, 128, 1, 1, 768, 512] or t["activation_checkpoint"] or not t["rgb_frozen"]):
        raise ValueError("Unregistered training capacity, budget or pixel protocol")
    if ([t[k] for k in ["learning_rate", "backbone_lr_multiplier", "weight_decay", "clip_global_norm", "poly_power"]]
            != [.0001, .1, .05, .01, .9] or t["stage_checkpoints"] != [3000, 6000]):
        raise ValueError("Unregistered optimizer/schedule/checkpoint choice")


def stability_passes(r, c):
    gate = c["stability"]
    if (r.get("status") != "completed_mask2former_stability_pending_replay"
            or r["trained_steps"] != gate["required_updates"] or r["partition"] != gate["partition"]
            or r["unique_sampled_views"] < gate["minimum_unique_sampled_views"]
            or any(n <= 0 for n in r["sampled_crop_class_counts"])
            or r["amp_skips"] / r["trained_steps"] > gate["maximum_skips_per_successful_update"]):
        raise ValueError("Longer numerical/coverage preflight failed; not a method quality result")


class Context:
    def __init__(self, root, artifact_root, partition):
        self.root, self.artifact_root = Path(root), Path(artifact_root)
        self.cfg = c = json.loads((self.root / CONFIG).read_text())
        validate_training_config(c)
        if partition not in c["partitions"]:
            raise ValueError("Only registered training-side folds permitted")
        self.hashes = {CONFIG: sha256(self.root / CONFIG), **{n: sha256(self.root/n) for n in CODE}}
        for name in ["preflight_config", "preflight_result", "preflight_audit", "pair_config", "pair_result", "grid_config"]:
            self.hashes[c[name]] = c[name+"_sha256"]
        verify_hashes(self.root, self.hashes)
        self.pc = pc = json.loads((self.root / c["preflight_config"]).read_text())
        self.probe = probe = json.loads((self.root / c["preflight_result"]).read_text())
        audit = json.loads((self.root / c["preflight_audit"]).read_text())
        if (audit["status"] != "completed_native_preflight_not_quality_validation"
                or audit["probe_result_sha256"] != c["preflight_result_sha256"]
                or probe["binding"]["config_sha256"] != c["preflight_config_sha256"]):
            raise ValueError("Unaccepted SEM340 provenance")
        verify_hashes(self.root, probe["binding"]["code_sha256"])
        self.runtime = self.root / c["runtime_root"]
        self.runtime_hashes = probe["binding"]["upstream_runtime_sha256"]
        verify_hashes(self.runtime, self.runtime_hashes)
        for name in ["dataset_manifest", "semantic_split", "oof_split", "label_manifest", "rgb_result"]:
            self.hashes[pc[name]] = pc[name+"_sha256"]
        verify_hashes(self.root, self.hashes)
        self.rgb_run = self.root / Path(pc["rgb_result"]).parent
        self.parent = json.loads((self.rgb_run / "result.json").read_text())
        verify_hashes(self.rgb_run, self.parent["artifacts"])
        self.split = json.loads((self.root / pc["semantic_split"]).read_text())
        self.oof = json.loads((self.root / pc["oof_split"]).read_text())
        self.access = semantic_access(self.split, self.oof,
                                     json.loads((self.rgb_run / "access.json").read_text()), partition)
        self.export_root = self.root / pc["rgb_export"]
        verify_hashes(self.export_root, {"result.json": pc["rgb_export_result_sha256"]})
        export = json.loads((self.export_root / "result.json").read_text())
        verify_export(self.export_root, self.split["semantic_train_labeled"], pc["rgb_result_sha256"], export["identity"], "semantic-training")
        self.rgb_hashes = {e["name"]: e["sha256"] for e in export["rendered_images"]}
        self.frozen = {e["path"]: e["sha256"] for e in json.loads((self.root /
            "experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json").read_text())["artifacts"]}
        self.label_manifest = json.loads((self.root / pc["label_manifest"]).read_text())
        self.binding = {"config_sha256": self.hashes[CONFIG], "code_sha256": {n: self.hashes[n] for n in CODE},
                        "input_hashes": self.hashes, "runtime_sha256": self.runtime_hashes,
                        "rgb_identity": export["identity"], "frozen_rgb_signature": export["frozen_rgb_signature"],
                        "export_sha256": pc["rgb_export_result_sha256"], "seed": c["seed"], "pixel_space": "PINHOLE",
                        "resolution": c["resolution"], "partition": partition,
                        "weight_sha256": pc["upstream"]["weights_sha256"]}
        self.finish()

    def labels(self):
        return TrainingLabels(self.artifact_root / "r2/sem_001/labels", self.access["train"],
                              self.label_manifest, self.cfg["resolution"])

    def guard(self):
        verify_hashes(self.root, self.hashes)
        verify_hashes(self.runtime, self.runtime_hashes)
        if shutil.disk_usage(self.root).free < self.cfg["resource"]["minimum_free_gib"]*1024**3:
            raise OSError("100GiB disk reserve reached; do not silently discard outputs")

    def finish(self):
        self.guard()
        verify_hashes(self.artifact_root, self.frozen)
        verify_hashes(self.rgb_run, self.parent["artifacts"])
        verify_hashes(self.export_root / "images", self.rgb_hashes)

    def setup_runtime(self):
        sys.path[:0] = [str(self.runtime/n) for n in ["runtime", "vendor/detectron2", "vendor/Mask2Former",
                                                   "vendor/Mask2Former/mask2former/modeling/pixel_decoder/ops"]]
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("GPU inaccessible to process, not a hardware-fault diagnosis")
        identity = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "versions": {n: importlib.metadata.version(n) for n in ["torchvision", "timm", "fvcore", "iopath", "yacs", "omegaconf", "hydra-core"]}}
        if identity != self.probe["runtime"]:
            raise ValueError("Runtime differs from accepted native preflight")
        random.seed(self.cfg["seed"]); np.random.seed(self.cfg["seed"]); torch.manual_seed(self.cfg["seed"])
        torch.cuda.manual_seed_all(self.cfg["seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return identity

    def model_config(self):
        c = config_for_model(self.runtime / "vendor/Mask2Former", self.pc["upstream"]["config"], False)
        c.defrost()
        c.DATASETS.TRAIN = ("sem341_rendered_training",)
        c.SOLVER.MAX_ITER = self.cfg["training"]["steps"]
        c.SOLVER.POLY_LR_POWER = self.cfg["training"]["poly_power"]
        c.freeze()
        if (c.SOLVER.BASE_LR != .0001 or c.SOLVER.BACKBONE_MULTIPLIER != .1 or c.SOLVER.WEIGHT_DECAY != .05
                or c.SOLVER.WARMUP_ITERS != 0 or c.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES != 100
                or c.MODEL.MASK_FORMER.TRAIN_NUM_POINTS != 12544):
            raise ValueError("Author optimizer/loss graph no longer matches registered adaptation")
        return c


def save_checkpoint(model, context, steps, path):
    import torch
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    state = model.state_dict()
    copied = OrderedDict((k, v.detach().cpu()) for k, v in state.items())
    copied._metadata = state._metadata
    if any(not torch.isfinite(v).all() for v in copied.values()):
        raise FloatingPointError("Nonfinite checkpoint state")
    with Path(path).open("xb") as stream:
        torch.save({"model": copied, "binding": context.binding, "access": context.access, "steps": steps}, stream)
    if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(cuda_rng, torch.cuda.get_rng_state()):
        raise ValueError("Checkpoint serialization changed training RNG")
    return {"steps": steps, "sha256": sha256(path), "rng_preserved": True,
            "cpu_rng_sha256": hashlib.sha256(cpu_rng.numpy().tobytes()).hexdigest(),
            "cuda_rng_sha256": hashlib.sha256(cuda_rng.cpu().numpy().tobytes()).hexdigest()}


def amp_update(model, optimizer, scaler, image, instances, *, max_attempts=12, clip=.01):
    """Bounded standard overflow recovery, same stochastic forward on each retry."""
    import torch
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    skips = []
    for attempt in range(max_attempts):
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng)
        with torch.autocast("cuda", dtype=torch.float16):
            losses = model([{"image": image, "instances": instances}])
            loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite unscaled native loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        bad = [n for n, v in model.named_parameters() if v.grad is not None and not torch.isfinite(v.grad).all()]
        if bad:
            before = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            skips.append({"attempt": attempt+1, "scale_before": before, "scale_after": scaler.get_scale(),
                          "bad_gradient_count": len(bad), "first_bad_parameters": bad[:8]})
            if scaler.get_scale() >= before:
                raise FloatingPointError("AMP did not skip/back off nonfinite gradient")
            del losses, loss
            continue
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
        scaler.step(optimizer); scaler.update()
        return {"loss": float(loss.detach()), "grad_norm": float(norm), "scale": scaler.get_scale(),
                "losses": {k: float(v.detach()) for k, v in losses.items()}, "skips": skips}
    raise FloatingPointError("AMP exhausted12 attempts without finite update; preserve failed run")
