#!/usr/bin/env python3
"""Independent CPU replay of QUAL018 lineage, metrics and artifact evidence."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from run_absgs_divergence import arm_config, guard, output, suite
from src.quality_protocol import sha256, verify_hashes, write_new_json


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    if not Path(path).exists():
        return []
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.endswith("}")
    ]


def progress():
    result = {}
    for eid in suite()["arms"]:
        run = ROOT / arm_config(eid)["training"]["output"]
        grads = rows(run / "trace/gradients.jsonl")
        probes = rows(run / "trace/probes.jsonl")
        resources = rows(run / "resources.jsonl")
        result[eid] = {
            "status": read(run / "result.json")["status"]
            if (run / "result.json").exists()
            else "running"
            if run.exists()
            else "queued",
            "gradient_step": grads[-1]["step"] if grads else None,
            "probe_step": probes[-1]["step"] if probes else None,
            "resources": resources[-1] if resources else None,
        }
    for name in ("execution-result.json", "analysis-result.json"):
        path = output() / name
        if path.exists():
            result[name] = {
                k: v
                for k, v in read(path).items()
                if k in ("status", "error", "seconds")
            }
    print(json.dumps(result, indent=2), flush=True)


def replay_lineage(run):
    import numpy as np

    trace = run / "trace"
    states = {r["step"]: r for r in rows(trace / "states.jsonl")}
    n0 = states[0]["count"]
    extent = states[0]["spatial_lr_scale"]
    percent_dense = read(trace / "runtime.json")["training_args"]["percent_dense"]
    ids = np.arange(n0, dtype=np.int64)
    next_id = n0
    appended, selected, deleted = 0, 0, 0
    birth_columns = [[], [], [], []]
    expected_parents = {}
    screens = []
    for event in rows(trace / "events.jsonl"):
        step, kind = event["step"], event["type"]
        if kind == "densify_begin":
            with np.load(trace / event["file"]) as z:
                assert np.array_equal(z["ids"], ids), (step, "score IDs")
                assert all(
                    np.isfinite(z[k]).all()
                    for k in ("signed", "absolute", "scale", "max_radii")
                )
                small = z["scale"] <= percent_dense * extent
                expected_parents = {
                    "clone": ids[(z["signed"] >= 0.0002) & small],
                    "split": np.tile(ids[(z["absolute"] >= 0.0004) & ~small], 2),
                }
                if event["screen_threshold"]:
                    count = int(
                        np.count_nonzero(z["max_radii"] > event["screen_threshold"])
                    )
                    assert count == event["screen_big_before_append"]
                    screens.append(count)
        elif kind == "append":
            with np.load(trace / event["file"]) as z:
                child_ids, parents = z["ids"], z["parents"]
                assert np.array_equal(parents, expected_parents[event["kind"]]), (
                    step,
                    "selection",
                )
                assert np.array_equal(
                    child_ids, np.arange(next_id, next_id + len(parents))
                )
                assert (
                    z["xyz"].shape == (len(parents), 3) and np.isfinite(z["xyz"]).all()
                )
                ids = np.concatenate((ids, child_ids))
                for column, values in zip(
                    birth_columns,
                    (
                        child_ids,
                        parents,
                        np.full(len(parents), step),
                        np.full(len(parents), 1 if event["kind"] == "clone" else 2),
                    ),
                ):
                    column.append(values.copy())
                next_id += len(parents)
                appended += 1
                selected += len(parents)
                assert event["max_radii_nonzero_after_append"] == 0
        elif kind == "prune":
            with np.load(trace / event["file"]) as z:
                removed = z["removed_ids"]
                mask = np.isin(ids, removed)
                assert np.array_equal(ids[mask], removed), (step, "prune IDs")
                ids = ids[~mask]
                deleted += len(removed)
                assert (
                    len(ids) == event["remaining"] and len(removed) == event["removed"]
                )
                if event["reasons"] is not None:
                    r = event["reasons"]
                    assert r["screen_big_at_actual_prune"] == 0
                    assert max(r.values()) <= len(removed) <= sum(r.values())
        elif kind == "densify_end":
            assert len(ids) == event["count"]
            snapshot = trace / f"snapshots/step-{step:05d}/ids.npy"
            if snapshot.exists():
                assert np.array_equal(ids, np.load(snapshot)), (step, "snapshot IDs")
    with np.load(trace / "lineage.npz") as z:
        assert np.array_equal(ids, z["final_ids"])
        for key, column in zip(("ids", "parents", "birth_step", "kind"), birth_columns):
            assert np.array_equal(np.concatenate(column), z[key]), key
    assert len(ids) == states[30000]["count"] == n0 + selected - deleted
    return {
        "initial_count": n0,
        "final_count": len(ids),
        "append_calls": appended,
        "created_count": selected,
        "removed_count": deleted,
        "active_screen_events": len(screens),
        "oversized_point_event_incidences": sum(screens),
        "lineage_and_threshold_replay_exact": True,
    }


def audit_pixels(run):
    import numpy as np
    from PIL import Image

    trace = run / "trace"
    photos = {}
    max_error = 0.0
    records = rows(trace / "probes.jsonl")
    expected = {
        (step, name)
        for step in suite()["probe_steps"]
        for name in suite()["probe_views"]
    }
    assert {(r["step"], r["view"]) for r in records} == expected and len(
        records
    ) == len(expected)
    for record in records:
        name, step = record["view"], record["step"]
        if name not in photos:
            photos[name] = np.asarray(
                Image.open(ROOT / suite()["scene"] / "images" / name).convert("RGB"),
                dtype=np.float64,
            )
        rgb = np.asarray(
            Image.open(trace / f"snapshots/step-{step:05d}" / name).convert("RGB"),
            dtype=np.float64,
        )
        squared_error = np.square(rgb - photos[name]).sum()
        psnr = float(10 * np.log10(rgb.size * 255**2 / max(squared_error, 1e-30)))
        error = abs(psnr - record["psnr_db"])
        max_error = max(max_error, error)
        assert error < 1e-10
    return {"probe_count": len(records), "independent_psnr_max_error_db": max_error}


def audit_parameters(roots):
    import numpy as np
    from plyfile import PlyData

    result = {}
    for step in (0, 1):
        data = [
            PlyData.read(p / f"trace/snapshots/step-{step:05d}/point_cloud.ply")[
                "vertex"
            ].data
            for p in roots
        ]
        fields = {}
        for name in data[0].dtype.names:
            a, b = data[0][name], data[1][name]
            assert np.isfinite(a).all() and np.isfinite(b).all()
            fields[name] = {
                "different_elements": int(np.count_nonzero(a != b)),
                "max_abs_difference": float(
                    np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))
                ),
            }
        result[str(step)] = {k: v for k, v in fields.items() if v["different_elements"]}
    for p in roots:
        v = PlyData.read(p / "model/point_cloud/iteration_30000/point_cloud.ply")[
            "vertex"
        ].data
        assert all(np.isfinite(v[k]).all() for k in v.dtype.names)
        result[p.parent.name] = {"final_ply_finite": True, "count": len(v)}
    return result


def audit_evaluation(run):
    import numpy as np
    from PIL import Image

    evaluation = run / "evaluation"
    result = read(evaluation / "result.json")
    assert result["status"] == "completed"
    verify_hashes(evaluation / "images", result["image_hashes"])
    fold = read(ROOT / suite()["split_path"])["primary_fold"]
    summary = {}
    max_error = 0.0
    for role, key in (("train", "train_rgb_400"), ("holdout", "validation_labeled")):
        with (evaluation / f"{role}.csv").open() as stream:
            records = list(csv.DictReader(stream))
        assert len(records) == len(fold[key]) and {r["view"] for r in records} == set(
            fold[key]
        )
        summary[role] = {}
        for metric in ("PSNR_dB", "SSIM", "LPIPS_Alex"):
            mean = float(np.array([float(r[metric]) for r in records]).mean())
            assert abs(mean - result["metrics"][role][metric]["mean"]) < 1e-12
            summary[role][metric] = mean
        for r in records:
            name = r["view"]
            target = (
                np.asarray(
                    Image.open(ROOT / suite()["scene"] / "images" / name).convert(
                        "RGB"
                    ),
                    dtype=np.float32,
                )
                / 255
            )
            actual = (
                np.asarray(
                    Image.open(evaluation / "images" / name).convert("RGB"),
                    dtype=np.float32,
                )
                / 255
            )
            squared = np.square(target - actual).astype(np.float64)
            value = float(-10 * np.log10(squared.sum() / squared.size))
            error = abs(value - float(r["PSNR_dB"]))
            max_error = max(error, max_error)
            assert error < 1e-10
    summary["independent_383_image_psnr_max_error_db"] = max_error
    summary["group_metrics"] = result["group_metrics"]
    return summary


def topology_firsts(roots):
    import numpy as np

    events = [rows(p / "trace/events.jsonl") for p in roots]
    result = {}
    for event_type, fields in (
        ("split_random", ("rng_before", "rng_after", "samples_sha256")),
        ("densify_end", ("count",)),
    ):
        aligned = [{r["step"]: r for r in e if r["type"] == event_type} for e in events]
        for field in fields:
            result[event_type + "_" + field] = next(
                (
                    s
                    for s in sorted(aligned[0])
                    if aligned[0][s][field] != aligned[1][s][field]
                ),
                None,
            )
    result["first_child_position_difference"] = None
    for step in range(600, 15000, 100):
        for kind in ("clone", "split"):
            with (
                np.load(roots[0] / f"trace/events/{step:05d}-{kind}.npz") as a,
                np.load(roots[1] / f"trace/events/{step:05d}-{kind}.npz") as b,
            ):
                if not np.array_equal(a["xyz"], b["xyz"]):
                    same = np.array_equal(a["parents"], b["parents"])
                    result["first_child_position_difference"] = {
                        "step": step,
                        "kind": kind,
                        "parents_exact": same,
                        "max_abs_difference": float(np.max(np.abs(a["xyz"] - b["xyz"])))
                        if same
                        else None,
                    }
                    return result
    return result


def audit(destination):
    guard()
    q = suite()
    roots = [ROOT / arm_config(e)["training"]["output"] for e in q["arms"]]
    result = {
        "status": "running",
        "code_sha256": sha256(Path(__file__)),
        "experiment_id": q["experiment_id"],
        "git_commit": q["git_commit"],
        "config_sha256": sha256(
            ROOT / f"experiments/runs/{q['experiment_id']}/config.yaml"
        ),
        "seed": q["seed"],
        "split_id": q["split_id"],
        "arms": {},
    }
    start = time.monotonic()
    try:
        for eid, run in zip(q["arms"], roots):
            record = read(run / "result.json")
            assert (
                record["status"] == "completed_spatial_training"
                and record["exit_status"] == 0
            )
            verify_hashes(run, record["artifacts"])
            verify_hashes(run, record["trace_hashes"])
            cameras = read(run / "trace/camera-trace.json")
            allowed = set(read(ROOT / q["split_path"])["primary_fold"]["train_rgb_400"])
            assert len(cameras) == 30000 and [r["step"] for r in cameras] == list(
                range(1, 30001)
            )
            assert {
                r["camera"] + ".png" if not Path(r["camera"]).suffix else r["camera"]
                for r in cameras
            } == allowed
            result["arms"][eid] = {
                **replay_lineage(run),
                **audit_pixels(run),
                "train_seconds": record["seconds"],
                "artifacts_and_trace_hashes_exact": True,
                "train_camera_roles_exact": True,
                "evaluation": audit_evaluation(run),
            }
        result["parameter_differences"] = audit_parameters(roots)
        result["topology_firsts"] = topology_firsts(roots)
        result["status"] = "passed"
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - start
        write_new_json(destination, result)
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--progress", action="store_true")
    p.add_argument("--wait-minutes", type=float, default=0)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.wait_minutes:
        deadline = time.monotonic() + a.wait_minutes * 60
        while time.monotonic() < deadline:
            state = subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "is-active",
                    "icshm-absgs-divergence-diagnostic.service",
                ],
                capture_output=True,
                text=True,
            )
            if state.stdout.strip() != "active":
                break
            time.sleep(min(30, max(0, deadline - time.monotonic())))
    if a.progress:
        progress()
    elif a.output:
        audit(a.output)
    else:
        p.error("Specify --progress or --output")


if __name__ == "__main__":
    main()
