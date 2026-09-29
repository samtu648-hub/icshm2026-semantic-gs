#!/usr/bin/env python3
"""Fixed two-model downstream probe; all predictions sealed before labels."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.spatial_seed_replication import confusion_metrics
from run_component_drop_extensions import gpu_lock, seed_everything
import run_sem396_spatial as base

CONFIG = ROOT / "experiments/runs/QUAL-016-DROPOUT-FROZEN-SEMANTIC-PROBE/config.yaml"


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-root", type=Path, required=True)
    a = p.parse_args()
    while subprocess.check_output(
        [
            "systemctl",
            "--user",
            "show",
            "icshm-component-exploratory-pair.service",
            "-p",
            "ActiveState",
            "--value",
        ],
        text=True,
    ).strip() in ("active", "activating"):
        time.sleep(15)
    c = read(CONFIG)
    o = ROOT / c["output_path"]
    o.mkdir(parents=True, exist_ok=False)
    split = read(ROOT / c["split_path"])
    names = sorted(split["primary_fold"]["semantic_validation_labeled"])
    groups = {n: g["group_id"] for g in split["groups"] for n in g["images"]}
    result = {
        "status": "starting",
        "experiment_id": c["experiment_id"],
        "scope": c["scope"],
        "bindings": {},
    }
    start = time.monotonic()
    try:
        import numpy as np
        from PIL import Image

        for seed, relative in c["semantic_models"].items():
            modeldir = ROOT / relative
            r = read(modeldir / "result.json")
            verify_hashes(modeldir, r["artifacts"])
            if set(r["access"]["train"]) & set(names):
                raise ValueError("Frozen semantic model saw confirmation labels")
            result["bindings"][seed] = sha256(modeldir / "model-6000.pt")
        for arm, relative in c["rgb_inputs"].items():
            r = read(ROOT / relative / "result.json")
            verify_hashes(
                ROOT / relative / "images", {n: r["image_hashes"][n] for n in names}
            )
            result["bindings"][arm] = sha256(ROOT / relative / "result.json")
        write_new_json(
            o / "binding.json",
            {
                "config_sha256": sha256(CONFIG),
                "script_sha256": sha256(Path(__file__)),
                "inputs": result["bindings"],
                "protocol_sha256": sha256(ROOT / c["protocol"]),
            },
        )
        with gpu_lock():
            runtime, torch, np = base.setup_runtime(a.artifact_root.resolve())
            from src.quality_mask2former import (
                config_for_model,
                load_official_model,
                native_predict,
            )

            seed_everything(0, torch, np)
            torch.cuda.reset_peak_memory_stats()
            hashes = {}
            for seed, relative in c["semantic_models"].items():
                mc = config_for_model(
                    runtime / "vendor/Mask2Former",
                    "configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml",
                    False,
                )
                model, _ = load_official_model(
                    mc, runtime / "weights/model_final_6b4a3a.pkl"
                )
                model.load_state_dict(
                    torch.load(
                        ROOT / relative / "model-6000.pt",
                        map_location="cpu",
                        weights_only=True,
                    )["model"]
                )
                model.cuda().eval()
                for arm, rgbdir in c["rgb_inputs"].items():
                    dest = o / seed / arm
                    dest.mkdir(parents=True)
                    for name in names:
                        image = np.asarray(
                            Image.open(ROOT / rgbdir / "images" / name).convert("RGB")
                        )
                        pred = native_predict(model, image).argmax(0).astype(np.uint8)
                        Image.fromarray(pred).save(dest / name)
                        hashes[str((dest / name).relative_to(o))] = sha256(dest / name)
                    print(
                        json.dumps(
                            {"seed": seed, "arm": arm, "predictions": len(names)}
                        ),
                        flush=True,
                    )
                del model
                torch.cuda.empty_cache()
            result["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
            write_new_json(
                o / "prediction-seal.json",
                {"predictions": hashes, "sealed_unix": time.time()},
            )
        manifest = {
            r["image"]: r["label_png_sha256"]
            for r in read(ROOT / c["label_manifest"])["records"]
        }
        verify_hashes(ROOT / c["label_root"], {n: manifest[n] for n in names})
        scores = {}
        per_view = []
        for seed in c["semantic_models"]:
            scores[seed] = {}
            for arm in c["rgb_inputs"]:
                totals = {
                    r: np.zeros((5, 5), np.int64)
                    for r in ["holdout", "region-1", "region-2", "region-3"]
                }
                for name in names:
                    target = np.asarray(Image.open(ROOT / c["label_root"] / name))
                    pred = np.asarray(Image.open(o / seed / arm / name))
                    m = np.zeros((5, 5), np.int64)
                    np.add.at(m, (target.ravel(), pred.ravel()), 1)
                    totals["holdout"] += m
                    totals[groups[name]] += m
                    per_view.append(
                        {
                            "seed": seed,
                            "arm": arm,
                            "view": name,
                            "group": groups[name],
                            **confusion_metrics(m.tolist()),
                        }
                    )
                scores[seed][arm] = {
                    r: confusion_metrics(m.tolist()) for r, m in totals.items()
                }
        deltas = {}
        for seed, s in scores.items():
            deltas[seed] = {}
            for role in s["control"]:
                old, new = s["control"][role], s["dropout"][role]
                deltas[seed][role] = {
                    "foreground_miou": new["mIoU_without_background"]
                    - old["mIoU_without_background"],
                    "five_class_miou": new["mIoU_with_background"]
                    - old["mIoU_with_background"],
                    "stay_cable_iou": new["per_class"]["stay_cable"]["IoU"]
                    - old["per_class"]["stay_cable"]["IoU"],
                }
        supported = all(
            d["holdout"]["foreground_miou"] > 0 and d["holdout"]["stay_cable_iou"] >= 0
            for d in deltas.values()
        )
        for seed, relative in c["semantic_models"].items():
            if sha256(ROOT / relative / "model-6000.pt") != result["bindings"][seed]:
                raise ValueError("Frozen checkpoint changed")
        verify_hashes(o, hashes)
        result.update(
            status="completed",
            scores=scores,
            deltas=deltas,
            matched_retraining_supported=supported,
            prediction_count=120,
            independent_confusion_count=120,
            per_view=per_view,
        )
        print(
            json.dumps({"deltas": deltas, "matched_retraining_supported": supported}),
            flush=True,
        )
    except BaseException as e:
        result.update(status="failed_preserved", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        result["seconds"] = time.monotonic() - start
        write_new_json(o / "result.json", result)


if __name__ == "__main__":
    main()
