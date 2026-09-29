#!/usr/bin/env python3
"""SEM370 full released-label fitting using unchanged native semantic recipes.

Each stage runs in a fresh process. Predictions are fixed training diagnostics,
not a held-out evaluation. No thresholds, experts or geometric losses are added.
"""

import argparse
from dataclasses import asdict
import fcntl
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
from src.quality_protocol import sha256, write_new_json, verify_hashes


def renderer(a):
    from src.quality_scaffold import ScaffoldQualityRenderer

    return ScaffoldQualityRenderer(
        a.artifact_root / "r1/third_party/scaffold-gs",
        a.rgb_run / "model",
        30000,
        train_camera_count=400,
    )


def save_probability(out, name, p):
    import numpy as np
    from PIL import Image
    from src.quality_experts import validate_probability

    validate_probability(p, p.shape[0], (989, 1320))
    (out / "probabilities").mkdir(exist_ok=True, parents=True)
    (out / "predictions").mkdir(exist_ok=True)
    np.savez_compressed(
        out / "probabilities" / (Path(name).stem + ".npz"), probabilities=p
    )
    Image.fromarray(p.argmax(0).astype(np.uint8)).save(out / "predictions" / name)


def export(a, out, r, cameras, split):
    import numpy as np
    from PIL import Image

    m = renderer(a)
    signature = m.rgb_signature()
    (out / "images").mkdir()
    r["identity"] = asdict(m.identity)
    r["frozen_rgb_signature"] = signature
    for i, n in enumerate(split["semantic_train"]):
        rgb = m.render_pinhole(cameras[n]).rgb
        if rgb.shape != (989, 1320, 3) or not np.isfinite(rgb).all():
            raise ValueError("Invalid native RGB")
        Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(out / "images" / n)
        if i % 50 == 0:
            print(json.dumps(dict(exported=i + 1, total=300)), flush=True)
    first = split["semantic_train"][0]
    assert np.array_equal(
        np.asarray(Image.open(out / "images" / first)),
        np.rint(m.render_pinhole(cameras[first]).rgb * 255).astype(np.uint8),
    )
    r.update(
        image_hashes={n: sha256(out / "images" / n) for n in split["semantic_train"]},
        repeated_rgb_exact=True,
    )


def static(a, out, r, cameras, labels, cfg):
    import numpy as np
    import torch
    from src.quality_semantic_fields import PrimitiveSemanticField
    from src.quality_static_training import fit_static, median_frequency_weights

    m = renderer(a)
    sig = m.rgb_signature()
    before = {
        n: m.render_pinhole(cameras[n]).rgb.copy() for n in cfg["diagnostic_views"][:3]
    }
    field = PrimitiveSemanticField(m)
    counts = labels.audit_counts()

    def progress(step, name, loss):
        r["trained_steps"] = step
        if step == 1 or step % 100 == 0:
            print(json.dumps(dict(stage="static", step=step, loss=loss)), flush=True)

    r["training"] = fit_static(
        m,
        field,
        {n: cameras[n] for n in sorted(labels.allowed)},
        labels,
        median_frequency_weights(counts),
        steps=cfg["static_steps"],
        seed=cfg["seed"],
        learning_rate=0.05,
        weight_decay=0.0,
        on_step=progress,
    )
    torch.save(
        dict(
            field=field.state_dict(),
            binding=r["binding"],
            access=r["access"],
            trained_steps=r["trained_steps"],
        ),
        out / "semantic_field.pt",
    )
    loaded = PrimitiveSemanticField(m)
    loaded.load_state_dict(
        torch.load(out / "semantic_field.pt", map_location="cpu", weights_only=True)[
            "field"
        ],
        m,
    )
    assert torch.equal(field.logits, loaded.logits)
    with torch.no_grad():
        for n in cfg["diagnostic_views"]:
            save_probability(
                out, n, loaded.render_probabilities(m, cameras[n]).cpu().numpy()
            )
    assert m.rgb_signature() == sig and all(
        np.array_equal(v, m.render_pinhole(cameras[n]).rgb) for n, v in before.items()
    )
    r.update(
        rgb_signature=sig,
        frozen_rgb_pixels_exact=list(before),
        checkpoint_reload_exact=True,
        class_counts=counts.tolist(),
    )


def b2(a, out, r, labels, cfg):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from src.quality_experts import (
        load_expert,
        expert_counts_weights,
        target_for_expert,
        predict,
    )
    from src.semantic_cable_binary import read_rgb, normalize_rgb
    from src.quality_mask2former_training import sampled_crop

    positive = {"b2": None, "deck": 1, "stay_cable": 2}[a.stage]
    crop = 768 if positive is None else 960
    stride = 512 if positive is None else 480
    steps = 6000 if positive is None else 3000
    weights_dir = a.artifact_root / "r2/sem_005/third_party/nvidia_mit_b2_3bb39e8"
    r["pretrained_hashes"] = {
        n: sha256(weights_dir / n) for n in ["config.json", "pytorch_model.bin"]
    }
    model = load_expert(weights_dir, cfg["seed"], positive).cuda()
    counts, weights = expert_counts_weights(labels.audit_counts(), positive)
    weight = torch.as_tensor(weights, device="cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)
    names = sorted(labels.allowed)
    images = {n: read_rgb(a.batch / "export/images" / n) for n in names}
    targets = {n: target_for_expert(labels.read(n), positive) for n in names}
    rng = random.Random(cfg["seed"])
    model.train()
    start = time.monotonic()
    with (out / "training.jsonl").open("x") as journal:
        for step in range(1, steps + 1):
            rgb, y, trace = sampled_crop(rng, names, images, targets, crop)
            x = normalize_rgb(rgb)[None].cuda()
            target = torch.from_numpy(y.astype(np.int64))[None].cuda()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp):
                logits = F.interpolate(
                    model(pixel_values=x).logits,
                    size=(crop, crop),
                    mode="bilinear",
                    align_corners=False,
                )
                loss = F.cross_entropy(logits.float(), target, weight=weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters()
            ):
                raise FloatingPointError("Nonfinite gradient")
            scaler.step(optimizer)
            scaler.update()
            r["trained_steps"] = step
            journal.write(
                json.dumps(dict(step=step, sample=trace, loss=float(loss.detach())))
                + "\n"
            )
            if step % 100 == 0 or step == 1:
                journal.flush()
                print(
                    json.dumps(
                        dict(
                            stage=a.stage,
                            step=step,
                            seconds=time.monotonic() - start,
                            loss=float(loss.detach()),
                        )
                    ),
                    flush=True,
                )
            if step in [3000, steps]:
                torch.save(
                    dict(
                        state_dict={
                            k: v.detach().cpu() for k, v in model.state_dict().items()
                        },
                        binding=r["binding"],
                        access=r["access"],
                        steps=step,
                    ),
                    out / f"expert-{step}.pt",
                )
    r["training_seconds"] = time.monotonic() - start
    del optimizer, scaler, model, images, targets
    torch.cuda.empty_cache()
    for endpoint in [3000, 6000] if positive is None else [3000]:
        model = load_expert(weights_dir, cfg["seed"], positive).cuda()
        model.load_state_dict(
            torch.load(
                out / f"expert-{endpoint}.pt", map_location="cpu", weights_only=True
            )["state_dict"]
        )
        dest = out / f"step-{endpoint}"
        for n in cfg["diagnostic_views"]:
            save_probability(
                dest,
                n,
                predict(
                    model,
                    read_rgb(a.batch / "export/images" / n),
                    crop=crop,
                    stride=stride,
                    amp_dtype=amp,
                ),
            )
        del model
        torch.cuda.empty_cache()
    r.update(
        class_counts=counts.tolist(),
        class_weights=weights.tolist(),
        amp=str(amp),
        endpoints=[3000, 6000] if positive is None else [3000],
    )


def mask2former(a, out, r, labels, cfg):
    import numpy as np
    import torch
    from src.quality_mask2former import (
        config_for_model,
        load_official_model,
        author_optimizer,
        class_masks,
        native_predict,
    )
    from src.quality_mask2former_training import (
        sampled_crop,
        amp_update,
        poly_factor,
        save_checkpoint,
    )
    from types import SimpleNamespace
    from src.semantic_cable_binary import read_rgb
    from detectron2.structures import Instances

    runtime = ROOT / "outputs/quality/mask2former/SEM-340"
    c = config_for_model(
        runtime / "vendor/Mask2Former",
        "configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml",
        False,
    )
    c.defrost()
    c.DATASETS.TRAIN = ("sem341_rendered_training",)
    c.SOLVER.MAX_ITER = 6000
    c.SOLVER.POLY_LR_POWER = 0.9
    c.freeze()
    (out / "resolved_model.yaml").write_text(c.dump())
    model, loaded = load_official_model(c, runtime / "weights/model_final_6b4a3a.pkl")
    r["pretrained_load"] = loaded
    assert sum(p.numel() for p in model.parameters()) == 215451514
    model.cuda()
    optimizer = author_optimizer(model, c)
    rates = [g["lr"] for g in optimizer.param_groups]
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    names = sorted(labels.allowed)
    images = {n: read_rgb(a.batch / "export/images" / n) for n in names}
    targets = {n: labels.read(n) for n in names}
    rng = random.Random(cfg["seed"])
    model.train()
    start = time.monotonic()
    r["amp_skips"] = 0
    with (out / "training.jsonl").open("x") as journal:
        for step in range(1, 6001):
            x, y, trace = sampled_crop(rng, names, images, targets, 768)
            ids, masks = class_masks(y)
            inst = Instances(tuple(y.shape))
            inst.gt_classes = torch.from_numpy(ids)
            inst.gt_masks = torch.from_numpy(masks)
            image = torch.from_numpy(x.transpose(2, 0, 1).copy()).float()
            factor = poly_factor(step - 1)
            for g, rate in zip(optimizer.param_groups, rates):
                g["lr"] = rate * factor
            row = amp_update(
                model, optimizer, scaler, image, inst, max_attempts=12, clip=0.01
            )
            row.update(
                step=step,
                sample=trace,
                lr_factor=factor,
                elapsed_seconds=time.monotonic() - start,
            )
            journal.write(json.dumps(row) + "\n")
            r["trained_steps"] = step
            r["amp_skips"] += len(row["skips"])
            if step % 100 == 0 or step == 1:
                journal.flush()
                print(
                    json.dumps(
                        dict(
                            stage="mask2former",
                            step=step,
                            amp_skips=r["amp_skips"],
                            seconds=time.monotonic() - start,
                        )
                    ),
                    flush=True,
                )
            if step in (3000, 6000):
                save_checkpoint(
                    model,
                    SimpleNamespace(binding=r["binding"], access=r["access"]),
                    step,
                    out / f"model-{step}.pt",
                )
    r["training_seconds"] = time.monotonic() - start
    del optimizer, scaler, images, targets, model
    torch.cuda.empty_cache()
    model, _ = load_official_model(c, runtime / "weights/model_final_6b4a3a.pkl")
    model.load_state_dict(
        torch.load(out / "model-6000.pt", map_location="cpu", weights_only=True)[
            "model"
        ]
    )
    model.cuda()
    for n in cfg["diagnostic_views"]:
        save_probability(
            out, n, native_predict(model, read_rgb(a.batch / "export/images" / n))
        )


def replay(a, out, r, cameras, cfg):
    import numpy as np
    import torch
    from PIL import Image
    from src.semantic_cable_binary import read_rgb
    from src.quality_experts import load_expert, predict

    original = a.batch / a.stage
    record = json.loads((original / "result.json").read_text())
    verify_hashes(original, record["artifacts"])
    r["source_result_sha256"] = sha256(original / "result.json")
    names = cfg["diagnostic_views"][:3]
    checked = []
    if a.stage in ["export", "static"]:
        m = renderer(a)
        if a.stage == "static":
            from src.quality_semantic_fields import PrimitiveSemanticField

            field = PrimitiveSemanticField(m)
            field.load_state_dict(
                torch.load(
                    original / "semantic_field.pt",
                    map_location="cpu",
                    weights_only=True,
                )["field"],
                m,
            )
        for n in names:
            if a.stage == "export":
                actual = np.rint(m.render_pinhole(cameras[n]).rgb * 255).astype(
                    np.uint8
                )
                expected = np.asarray(Image.open(original / "images" / n))
            else:
                with torch.no_grad():
                    actual = field.render_probabilities(m, cameras[n]).cpu().numpy()
                with np.load(original / "probabilities" / (Path(n).stem + ".npz")) as z:
                    expected = z["probabilities"]
            if not np.array_equal(actual, expected):
                raise ValueError("Fresh native replay differs: " + n)
            checked.append(n)
    elif a.stage == "mask2former":
        from src.quality_mask2former import (
            config_for_model,
            load_official_model,
            native_predict,
        )

        runtime = ROOT / "outputs/quality/mask2former/SEM-340"
        c = config_for_model(
            runtime / "vendor/Mask2Former",
            "configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml",
            False,
        )
        m, _ = load_official_model(c, runtime / "weights/model_final_6b4a3a.pkl")
        m.load_state_dict(
            torch.load(
                original / "model-6000.pt", map_location="cpu", weights_only=True
            )["model"]
        )
        m.cuda()
        for n in names:
            actual = native_predict(m, read_rgb(a.batch / "export/images" / n))
            with np.load(original / "probabilities" / (Path(n).stem + ".npz")) as z:
                expected = z["probabilities"]
            if not np.array_equal(actual, expected):
                raise ValueError("Fresh M2F replay differs: " + n)
            checked.append(n)
    else:
        positive = {"b2": None, "deck": 1, "stay_cable": 2}[a.stage]
        crop = 768 if positive is None else 960
        stride = 512 if positive is None else 480
        for endpoint in [3000, 6000] if positive is None else [3000]:
            m = load_expert(
                a.artifact_root / "r2/sem_005/third_party/nvidia_mit_b2_3bb39e8",
                cfg["seed"],
                positive,
            ).cuda()
            m.load_state_dict(
                torch.load(
                    original / f"expert-{endpoint}.pt",
                    map_location="cpu",
                    weights_only=True,
                )["state_dict"]
            )
            for n in names:
                actual = predict(
                    m,
                    read_rgb(a.batch / "export/images" / n),
                    crop=crop,
                    stride=stride,
                    amp_dtype=torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16,
                )
                with np.load(
                    original
                    / f"step-{endpoint}/probabilities"
                    / (Path(n).stem + ".npz")
                ) as z:
                    expected = z["probabilities"]
                if not np.array_equal(actual, expected):
                    raise ValueError("Fresh B2 replay differs: " + n)
                checked.append(f"{endpoint}/{n}")
            del m
            torch.cuda.empty_cache()
    r.update(fresh_probability_or_rgb_exact=True, checked=checked)


def main():
    p = argparse.ArgumentParser()
    for n in ["artifact-root", "rgb-run", "batch"]:
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument(
        "--stage",
        choices=["export", "static", "b2", "deck", "stay_cable", "mask2former"],
        required=True,
    )
    p.add_argument("--replay", action="store_true")
    a = p.parse_args()
    a.artifact_root = a.artifact_root.resolve()
    a.rgb_run = a.rgb_run.resolve()
    a.batch = a.batch.resolve()
    source = a.artifact_root / "r1/third_party/scaffold-gs"
    runtime = ROOT / "outputs/quality/mask2former/SEM-340"
    sys.path[:0] = [
        str(p)
        for p in [
            source,
            source / "submodules/diff-gaussian-rasterization",
            source / "submodules/simple-knn",
            runtime / "runtime",
            runtime / "vendor/detectron2",
            runtime / "vendor/Mask2Former",
            runtime / "vendor/Mask2Former/mask2former/modeling/pixel_decoder/ops",
        ]
    ]
    import numpy as np
    import torch
    from src.quality_static_training import TrainingLabels
    from src.quality_cameras import read_camera_requests

    cfgpath = ROOT / "experiments/runs/SEM-370-ALL300/config.yaml"
    cfg = json.loads(cfgpath.read_text())
    split = json.loads((a.rgb_run / "split.json").read_text())
    parent = json.loads((a.rgb_run / "result.json").read_text())
    if parent["status"] != "completed_all400_rgb_pending_matched_semantics":
        raise ValueError("RGB endpoint incomplete")
    verify_hashes(a.rgb_run, parent["artifacts"])
    assert len(split["rgb_train"]) == 400 and len(split["semantic_train"]) == 300
    cameras = read_camera_requests(a.rgb_run / "scene/sparse/0")
    assert set(cameras) == set(split["rgb_train"])
    assert all(
        c.model == "PINHOLE" and (c.width, c.height) == (1320, 989)
        for c in cameras.values()
    )
    labels = TrainingLabels(
        a.batch / "labels",
        split["semantic_train"],
        json.loads((a.batch / "label_manifest.json").read_text()),
        [1320, 989],
    )
    with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = a.batch / (a.stage + ("-replay" if a.replay else ""))
        out.mkdir(exist_ok=False)
        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        code = [
            "scripts/run_quality_all300.py",
            "src/quality_static_training.py",
            "src/quality_semantic_fields.py",
            "src/quality_scaffold.py",
            "src/quality_experts.py",
            "src/quality_mask2former.py",
            "src/quality_mask2former_training.py",
            "src/semantic_cable_binary.py",
        ]
        r = dict(
            experiment_id=cfg["experiment_id"],
            stage=a.stage,
            status="running",
            trained_steps=0,
            binding=dict(
                config_sha256=sha256(cfgpath),
                parent_result_sha256=sha256(a.rgb_run / "result.json"),
                split_sha256=sha256(a.rgb_run / "split.json"),
                label_manifest_sha256=sha256(a.batch / "label_manifest.json"),
                seed=cfg["seed"],
            ),
            access=dict(
                train=split["semantic_train"],
                prediction=cfg["diagnostic_views"],
                partition="all300_training_diagnostic",
                independent_holdout=False,
            ),
            git_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            code_sha256={n: sha256(ROOT / n) for n in code},
            quality_metrics=None,
        )
        for n in code:
            dest = out / "code" / n
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / n, dest)
        write_new_json(out / "started.json", r)
        try:
            if a.stage != "export":
                export_result = json.loads((a.batch / "export/result.json").read_text())
                verify_hashes(a.batch / "export/images", export_result["image_hashes"])
                r["binding"]["export_result_sha256"] = sha256(
                    a.batch / "export/result.json"
                )
            if a.replay:
                replay(a, out, r, cameras, cfg)
            elif a.stage == "export":
                export(a, out, r, cameras, split)
            elif a.stage == "static":
                static(a, out, r, cameras, labels, cfg)
            elif a.stage == "mask2former":
                mask2former(a, out, r, labels, cfg)
            else:
                b2(a, out, r, labels, cfg)
            verify_hashes(a.rgb_run, parent["artifacts"])
            verify_hashes(ROOT, r["code_sha256"])
            r.update(
                status="completed_training_diagnostic_only",
                rgb_artifacts_unchanged=True,
                labels_accessed=sorted(labels.accessed),
                artifacts={
                    str(p.relative_to(out)): sha256(p)
                    for p in out.rglob("*")
                    if p.is_file() and p.suffix in (".pt", ".npz", ".png")
                },
            )
        except BaseException as e:
            r.update(status="failed_preserved", error=f"{type(e).__name__}: {e}")
            raise
        finally:
            r.update(
                seconds=time.monotonic() - start,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            )
            write_new_json(out / "result.json", r)


if __name__ == "__main__":
    main()
