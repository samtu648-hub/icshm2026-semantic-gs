#!/usr/bin/env python3
"""Independent CPU recount of QUAL003 evidence, not a second geometry estimator."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "artifact-root", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--semantic-run", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    parent = json.loads((a.run / "result.json").read_text())
    record = {"experiment_id": "QUAL-003-COVERAGE-INIT-DIAG", "status": "running_independent_recount",
        "run_result_sha256": sha256(a.run / "result.json"), "audit_code_sha256": sha256(Path(__file__)),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "independent_renderer": False}
    try:
        import numpy as np
        from PIL import Image
        cfg_path = ROOT / "experiments/runs/QUAL-003-COVERAGE-INIT-DIAG/config.yaml"
        if sha256(cfg_path) != parent["config_sha256"]:
            raise ValueError("Diagnostic config changed")
        cfg = json.loads(cfg_path.read_text())
        protocol = json.loads((ROOT / cfg["protocol"]).read_text())
        verify_hashes(a.run, parent["artifacts"])
        verify_hashes(ROOT, parent["input_hashes"])
        if parent["stage"] == "actual_initialization_intersection":
            if parent["status"] != "completed_actual_initialization_diagnosis" or a.semantic_run:
                raise ValueError("Not a completed initialization diagnostic")
            scene = a.artifact_root / protocol["rgb"]["scene"]
            verify_hashes(scene, parent["scene_input_hashes"])
            ids = set()
            for line in (scene / "sparse/0/points3D.txt").read_text().splitlines():
                if line and not line.startswith("#"):
                    ids.add(int(line.split()[0]))
            with np.load(a.run / "retained_conditioning.npz", allow_pickle=False) as z:
                values = {n: z[n] for n in z.files}
            s = dict(initial_points=len(ids), cross_predicted_tracks=len(values["point_id"]),
                cross_predicted_retained=0, large_error_all=0, large_error_retained=0,
                weak_all=0, weak_retained=0, weak_large_retained=0, well_conditioned_large_retained=0)
            for pid, error, angle, stored_keep in zip(values["point_id"], values["error_px"], values["source_max_ray_angle_deg"], values["retained"]):
                keep, large, weak = int(pid) in ids, float(error) > 5., float(angle) < 1.
                if keep != bool(stored_keep):
                    raise ValueError("Independent scalar retained membership differs")
                for key, yes in (("cross_predicted_retained", keep), ("large_error_all", large), ("large_error_retained", large and keep),
                                 ("weak_all", weak), ("weak_retained", weak and keep), ("weak_large_retained", keep and large and weak),
                                 ("well_conditioned_large_retained", keep and large and not weak)):
                    s[key] += int(yes)
            s["initial_without_successful_cross_prediction"] = len(ids)-s["cross_predicted_retained"]
            if s != parent["summary"]:
                raise ValueError("Independent scalar track counts disagree")
            record.update(summary=s, labels_decoded=False, stage="initialization_join_recount")
        elif parent["stage"] == "native_coverage":
            if parent["status"] != "completed_native_coverage_diagnosis" or not a.semantic_run or sha256(a.semantic_run / "result.json") != parent["semantic_run_sha256"]:
                raise ValueError("Requires matching completed static parent")
            sem = json.loads((a.semantic_run / "result.json").read_text())
            verify_hashes(a.semantic_run, sem["artifacts"])
            sc = json.loads((ROOT / "experiments/runs/SEM-330-QUALITY-STATIC/config.yaml").read_text())
            per_view = json.loads((a.run / "per_view.json").read_text())
            pooled = {str(c): {k: 0 for k in parent["classes"][str(c)]} for c in range(5)}
            for name in parent["views"]:
                path = a.artifact_root / sc["labels"] / name
                if sha256(path) != parent["target_label_hashes"][name]:
                    raise ValueError("Diagnostic label changed")
                with Image.open(path) as im:
                    gt = np.asarray(im).ravel()
                with np.load(a.run / "coverage" / (Path(name).stem+".npz"), allow_pickle=False) as z:
                    alpha = z["alpha"].ravel()
                    if not np.array_equal(z["low_coverage"].ravel(), alpha < .5) or not np.array_equal(z["residual_background"].ravel(), np.maximum(1-alpha, 0)):
                        raise ValueError("Coverage stored masks disagree with raw alpha")
                with np.load(a.semantic_run / "probabilities" / (Path(name).stem+".npz"), allow_pickle=False) as z:
                    pred = z["probabilities"].argmax(0).ravel()
                low = alpha < .5
                # Single 5x5x2 histogram is independent of the diagnostic's class mask sums.
                hist = np.bincount(gt.astype(np.int64)*10+pred*2+low.astype(np.int64), minlength=50).reshape(5, 5, 2)
                for c in range(5):
                    h = hist[c]
                    counts = {"gt": int(h.sum()), "errors": int(h.sum()-h[c].sum()),
                        "low_coverage_gt": int(h[:, 1].sum()), "low_coverage_errors": int(h[:, 1].sum()-h[c, 1]),
                        "high_coverage_errors": int(h[:, 0].sum()-h[c, 0]),
                        "to_background": int(h[0].sum()), "low_coverage_to_background": int(h[0, 1])}
                    if counts != per_view[name]["classes"][str(c)]:
                        raise ValueError("Independent coverage histogram differs")
                    for key, value in counts.items():
                        pooled[str(c)][key] += value
            if pooled != parent["classes"]:
                raise ValueError("Pooled independent counts disagree")
            record.update(classes=pooled, views=parent["views"], labels_decoded=True, stage="coverage_histogram_recount")
        else:
            raise ValueError("Unknown diagnostic stage")
        frozen = json.loads((ROOT / protocol["frozen_manifest"]).read_text())
        verify_hashes(a.artifact_root, {r["path"]: r["sha256"] for r in frozen["artifacts"]})
        verify_hashes(a.run, parent["artifacts"])
        record.update(status="completed_independent_diagnostic_recount", frozen_verified=len(frozen["artifacts"]))
    except BaseException as exc:
        record.update(status="diagnostic_recount_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["seconds"] = time.monotonic()-started
        write_new_json(a.output / "result.json", record)
        print(json.dumps({"status": record["status"], "output": str(a.output)}), flush=True)


if __name__ == "__main__":
    main()
