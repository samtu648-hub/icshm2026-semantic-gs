#!/usr/bin/env python3
"""Reuse the frozen quota implementation for exactly one second-seed pair."""

import argparse
import json
from pathlib import Path
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import run_component_exploratory_pair as parent
import run_component_drop_extensions as ext
from src.quality_protocol import sha256, verify_hashes, write_new_json

parent.EID = "QUAL-017-COMPONENT-SEED-CONFIRMATION"


def run(artifact_root):
    parent.prepare()
    extra = [
        str(Path(__file__).relative_to(ROOT)),
        "scripts/run_component_exploratory_pair.py",
        parent.cfg()["first_seed_result"],
    ]
    write_new_json(
        parent.output() / "confirmation-binding.json",
        {"files": {p: sha256(ROOT / p) for p in extra}},
    )
    while subprocess.check_output(
        [
            "systemctl",
            "--user",
            "show",
            "icshm-native-absgs-replay.service",
            "-p",
            "ActiveState",
            "--value",
        ],
        text=True,
    ).strip() in ("active", "activating"):
        time.sleep(15)
    start = time.monotonic()
    records = []
    result = {"status": "running"}

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
        t = time.monotonic()
        with (parent.output() / f"{name}.log").open("x") as log:
            proc = subprocess.run(
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
        r = {
            "name": name,
            "command": command,
            "exit_status": proc.returncode,
            "seconds": time.monotonic() - t,
        }
        write_new_json(parent.output() / f"{name}-execution.json", r)
        records.append(r)
        print(json.dumps(r), flush=True)
        if proc.returncode:
            raise RuntimeError(f"Failed: {name}")

    try:
        for eid in parent.cfg()["pair"]:
            execute(eid + "-prepare", "prepare", eid)
        execute("preflight", "preflight")
        execute("rgb-before", "rgb-before")
        for eid in parent.cfg()["pair"]:
            execute(eid + "-train", "train", eid)
        execute("seal", "seal")
        for eid in parent.cfg()["pair"]:
            execute(eid + "-inference", "inference", eid)
        execute("rgb-after", "rgb-after")
        execute("audit", "audit")
        result["status"] = "completed"
    except BaseException as e:
        result.update(status="failed_preserved", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        result.update(seconds=time.monotonic() - start, stages=records)
        write_new_json(parent.output() / "execution-result.json", result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--stage", default="run")
    p.add_argument("--experiment")
    a = p.parse_args()
    root = a.artifact_root.resolve()
    if a.stage == "run":
        run(root)
        return
    verify_hashes(
        ROOT, parent.read(parent.output() / "confirmation-binding.json")["files"]
    )
    if a.stage == "preflight":
        import audit_component_drop_extensions as worker

        original = worker.config

        def config(eid):
            c = original(eid)
            if eid == ext.SUITE:
                c = {**c, "semantic_pair": parent.cfg()["pair"]}
            if eid == ext.DIAG:
                c = {
                    **c,
                    "sampling_plan": ext.config(parent.cfg()["pair"][0])["output_path"]
                    + "/sampling-plan.json",
                }
            return c

        worker.config = config
        worker.out = parent.output
        worker.preflight_semantic(root)
    elif a.stage.startswith("rgb-"):
        parent.rgb_check(root, a.stage[4:])
    elif a.stage == "seal":
        parent.seal()
    elif a.stage == "audit":
        parent.audit()
        old = parent.read(ROOT / parent.cfg()["first_seed_result"])
        new = parent.read(parent.output() / "audit-result.json")
        mean = {
            role: {
                k: (old["delta"][role][k] + new["delta"][role][k]) / 2
                for k in old["delta"][role]
            }
            for role in old["delta"]
        }
        write_new_json(
            parent.output() / "two-seed-result.json",
            {
                "status": "completed",
                "first_seed_passed": old["retention_gate_passed"],
                "second_seed_passed": new["retention_gate_passed"],
                "replication_passed": old["retention_gate_passed"]
                and new["retention_gate_passed"],
                "mean_delta": mean,
                "first_seed_delta": old["delta"]["holdout"],
                "second_seed_delta": new["delta"]["holdout"],
            },
        )
    else:
        parent.semantic(a.experiment, a.stage, root)


if __name__ == "__main__":
    main()
