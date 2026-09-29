#!/usr/bin/env python3
"""One fixed exploratory quota pair, preserving the negative SEM416 diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import run_sem396_spatial as base
import run_component_drop_extensions as ext
from src.quality_protocol import sha256, verify_hashes, write_new_json

EID = "QUAL-015-COMPONENT-EXPLORATORY"


def read(path):
    return json.loads(Path(path).read_text())


def cfg():
    return ext.config(EID)


def output():
    return ROOT / cfg()["output_path"]


def guard():
    ext.guard()
    verify_hashes(ROOT, read(output() / "binding.json")["files"])


def prepare():
    c = cfg()
    output().mkdir(parents=True, exist_ok=False)
    paths = [
        str(Path(__file__).relative_to(ROOT)),
        f"experiments/runs/{EID}/config.yaml",
        c["protocol"],
    ]
    paths += [f"experiments/runs/{e}/config.yaml" for e in c["pair"]]
    paths += [
        "outputs/quality/SEM-416/r1/result.json",
        "outputs/quality/SEM-412/r1/inference/result.json",
    ]
    for relative in paths:
        target = output() / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    write_new_json(
        output() / "binding.json",
        {
            "files": {p: sha256(ROOT / p) for p in paths},
            "upstream_binding_sha256": sha256(ext.out() / "binding.json"),
            "created_unix": time.time(),
            "diagnostic_support": False,
            "authorization": "User authorized a fixed exploratory intervention without prevalence prerequisite",
        },
    )


def rgb_check(artifact_root, phase):
    guard()
    c = ext.config(cfg()["pair"][0])
    record = {"status": "starting", "phase": phase, "views": {}}
    start = time.monotonic()
    try:
        with ext.gpu_lock():
            _, torch, np = base.setup_runtime(artifact_root)
            from gaussian_renderer import GaussianModel, render
            from arguments import PipelineParams
            from PIL import Image
            from src.quality_cameras import read_camera_requests
            from src.quality_native_gaussian import make_native_view

            ext.seed_everything(0, torch, np)
            model = GaussianModel(3)
            model.load_ply(
                str(
                    ROOT
                    / c["rgb_run"]
                    / "model/point_cloud/iteration_30000/point_cloud.ply"
                )
            )
            p = argparse.ArgumentParser()
            pipe = PipelineParams(p).extract(p.parse_args([]))
            cams = read_camera_requests(ROOT / "outputs/quality/RGB-154/scene/sparse/0")
            with torch.no_grad():
                for n in ("001.png", "120.png", "300.png"):
                    image = render(
                        make_native_view(cams[n]),
                        model,
                        pipe,
                        torch.zeros(3, device="cuda"),
                    )["render"]
                    got = np.rint(
                        image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
                    ).astype(np.uint8)
                    expected = np.asarray(
                        Image.open(ROOT / c["rgb_evaluation"] / "images" / n).convert(
                            "RGB"
                        )
                    )
                    if not np.array_equal(got, expected):
                        raise ValueError("Frozen RGB differs")
                    record["views"][n] = "pixel_exact"
        verify_hashes(
            ROOT / c["rgb_run"], read(ROOT / c["rgb_run"] / "result.json")["artifacts"]
        )
        record["status"] = "passed"
    except BaseException as e:
        record.update(status="failed_preserved", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        record["seconds"] = time.monotonic() - start
        write_new_json(output() / f"rgb-{phase}.json", record)


def seal():
    guard()
    records = []
    traces = []
    checkpoints = {}
    for eid in cfg()["pair"]:
        c = ext.config(eid)
        path = ROOT / c["output_path"] / "mask2former"
        r = read(path / "result.json")
        if r["status"] != "completed_sealed_6000" or r["trained_steps"] != 6000:
            raise ValueError("Incomplete paired endpoint")
        verify_hashes(path, r["artifacts"])
        records.append(r)
        traces.append(
            [
                json.loads(l)["sample"]
                for l in (path / "training.jsonl").read_text().splitlines()
            ]
        )
        checkpoints[eid] = sha256(path / "model-6000.pt")
    if (
        records[0]["initial_model_state_sha256"]
        != records[1]["initial_model_state_sha256"]
        or traces[0] != traces[1]
    ):
        raise ValueError("Unmatched initial states or sampling")
    write_new_json(
        output() / "pair-seal.json",
        {
            "status": "sealed",
            "checkpoint_hashes": checkpoints,
            "paired_samples": 6000,
            "initial_state_sha256": records[0]["initial_model_state_sha256"],
            "sealed_unix": time.time(),
        },
    )


def semantic(eid, stage, artifact_root):
    guard()
    c = ext.config(eid)
    base.EID, base.CONFIG = eid, ROOT / f"experiments/runs/{eid}/config.yaml"
    base.CODE_PATHS = tuple(
        sorted(
            set(base.CODE_PATHS)
            | set(ext.NEW_CODE)
            | {str(Path(__file__).relative_to(ROOT))}
        )
    )
    if stage == "prepare":
        base.prepare(c)
    elif stage == "train":
        if (
            read(output() / "semantic-preflight/result.json")["status"]
            != "passed_eight_update_pair"
        ):
            raise ValueError("Preflight failed")
        original = base.setup_runtime
        hook = None

        def setup(root):
            nonlocal hook
            value = original(root)
            from src.component_point_sampling import ComponentSampler

            hook = ComponentSampler(c["component_quota"])
            hook.install()
            return value

        base.setup_runtime = setup
        try:
            base.train(c, artifact_root)
        finally:
            base.setup_runtime = original
            if hook:
                hook.restore()
                write_new_json(
                    ROOT / c["output_path"] / "quota-statistics.json", hook.stats
                )
    else:
        sealed = read(output() / "pair-seal.json")
        for name, expected in sealed["checkpoint_hashes"].items():
            if (
                sha256(
                    ROOT / ext.config(name)["output_path"] / "mask2former/model-6000.pt"
                )
                != expected
            ):
                raise ValueError("Endpoint changed")
        base.inference(c, artifact_root)
    guard()


def audit():
    guard()
    from audit_component_drop_extensions import recount_extension
    from src.spatial_seed_replication import flatten

    raw = []
    audits = []
    costs = []
    for eid in cfg()["pair"]:
        c = ext.config(eid)
        path = ROOT / c["output_path"]
        r = read(path / "inference/result.json")
        audits.append(recount_extension(c, read(ROOT / c["split_path"]), r))
        raw.append(r)
        train = read(path / "mask2former/result.json")
        costs.append(
            {
                "training_seconds": train["training_seconds"],
                "allocated_mib": train["peak_allocated_mib"],
                "reserved_mib": train["peak_reserved_mib"],
                "quota": read(path / "quota-statistics.json"),
            }
        )
    a, b = raw
    delta = {
        role: {k: flatten(b, role)[k] - flatten(a, role)[k] for k in flatten(a, role)}
        for role in a["metrics"]
    }
    cd = {
        n: b["metrics"]["holdout"]["per_class"][n]["IoU"]
        - a["metrics"]["holdout"]["per_class"][n]["IoU"]
        for n in ("deck", "stay_cable", "tower", "foundation")
    }
    d = delta["holdout"]
    passed = (
        d["foreground_miou"] > 0
        and d["boundary_f1"] >= 0
        and d["stay_cable_precision"] >= -0.001
        and min(cd.values()) >= -0.005
    )
    guard()
    result = {
        "status": "completed",
        "experiment_id": EID,
        "retention_gate_passed": passed,
        "independent_audits": audits,
        "control": {r: flatten(a, r) for r in a["metrics"]},
        "candidate": {r: flatten(b, r) for r in b["metrics"]},
        "delta": delta,
        "class_iou_delta": cd,
        "costs": costs,
        "diagnostic_prevalence_gate_still_failed": True,
    }
    write_new_json(output() / "audit-result.json", result)
    print(
        json.dumps(
            {"retention_gate_passed": passed, "delta": d, "class_iou_delta": cd}
        ),
        flush=True,
    )


def run(artifact_root):
    prepare()
    while subprocess.check_output(
        [
            "systemctl",
            "--user",
            "show",
            "icshm-component-drop-extensions.service",
            "-p",
            "ActiveState",
            "--value",
        ],
        text=True,
    ).strip() in ("active", "activating"):
        time.sleep(15)
    records = []
    start = time.monotonic()

    def execute(name, stage, eid=None):
        command = [
            sys.executable,
            str(Path(__file__)),
            "--artifact-root",
            str(artifact_root),
            "--stage",
            stage,
        ]
        if eid:
            command += ["--experiment", eid]
        begin = time.monotonic()
        with (output() / f"{name}.log").open("x") as log:
            p = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )
        row = {
            "name": name,
            "command": command,
            "exit_status": p.returncode,
            "seconds": time.monotonic() - begin,
        }
        write_new_json(output() / f"{name}-execution.json", row)
        records.append(row)
        print(json.dumps(row), flush=True)
        if p.returncode:
            raise RuntimeError(f"Stage failed: {name}")

    result = {"status": "running"}
    try:
        execute("preflight", "preflight")
        execute("rgb-before", "rgb-before")
        for stage in ("prepare", "train"):
            for eid in cfg()["pair"]:
                execute(f"{eid}-{stage}", stage, eid)
        execute("seal", "seal")
        for eid in cfg()["pair"]:
            execute(f"{eid}-inference", "inference", eid)
        execute("rgb-after", "rgb-after")
        execute("audit", "audit")
        result["status"] = "completed"
    except BaseException as e:
        result.update(status="failed_preserved", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        result.update(seconds=time.monotonic() - start, stages=records)
        write_new_json(output() / "execution-result.json", result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--experiment")
    a = p.parse_args()
    root = a.artifact_root.resolve()
    if a.stage == "run":
        run(root)
    elif a.stage == "preflight":
        import audit_component_drop_extensions as worker

        worker.out = output
        worker.preflight_semantic(root)
    elif a.stage.startswith("rgb-"):
        rgb_check(root, a.stage[4:])
    elif a.stage == "seal":
        seal()
    elif a.stage == "audit":
        audit()
    else:
        semantic(a.experiment, a.stage, root)


if __name__ == "__main__":
    main()
