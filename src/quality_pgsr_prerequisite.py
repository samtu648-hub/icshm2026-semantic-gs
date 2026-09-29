"""Fail-closed evidence checks for the independently registered PGSR repair."""
import json
from src.quality_protocol import sha256, verify_hashes


def validate_diagnostic(record, expected, current_source, helper_hash):
    backward = record.get("backward_diagnostic", {})
    if (record.get("status") != "completed_numeric_diagnostic_not_quality_experiment"
            or backward.get("status") != "diagnostic_limit_reached_no_nonfinite_observed"
            or backward.get("completed_optimizer_steps", 0) < expected["minimum_completed_optimizer_steps"]
            or record.get("source_manifest") != current_source
            or record.get("source_delta_patch_sha256") != expected["patch_sha256"]
            or record.get("code_sha256", {}).get(expected["helper"]) != helper_hash
            or helper_hash != expected["helper_sha256"]):
        raise ValueError("PGSR repair differs from its passed full diagnostic window")


def validate_smoke(record, *, experiment, config_hash, protocol_hash, code_hashes, source_hash, extra_sources):
    if (record.get("experiment_id") != experiment or record.get("mode") != "smoke"
            or record.get("status") != "completed_native_training_pending_independent_reload_and_evaluation"
            or record.get("config_sha256") != config_hash or record.get("protocol_sha256") != protocol_hash
            or record.get("code_sha256") != code_hashes or record.get("source_manifest_sha256") != source_hash
            or record.get("extra_sources") != extra_sources
            or not record.get("numeric_inventory") or any(not v.get("finite") for v in record["numeric_inventory"].values())):
        raise ValueError("Full PGSR repair requires the same accepted smoke configuration, code and source")


def check_repair(root, config, source_files):
    path = root / "experiments/runs/RGB-115-PGSR-SAFE-CYCLE/execution_config.yaml"
    expected = json.loads(path.read_text())
    if expected["experiment_id"] != config["experiment_id"] or config["project_dependency"] != expected["helper"] or config["source_delta_patch"] != expected["patch"]:
        raise ValueError("Unregistered PGSR repair dependencies")
    verify_hashes(root, {expected["diagnostic_result"]: expected["diagnostic_result_sha256"],
                         expected["patch"]: expected["patch_sha256"], expected["helper"]: expected["helper_sha256"]})
    diagnostic = json.loads((root / expected["diagnostic_result"]).read_text())
    validate_diagnostic(diagnostic, expected, source_files, sha256(root / expected["helper"]))
    return {"execution_config_sha256": sha256(path), **expected}
