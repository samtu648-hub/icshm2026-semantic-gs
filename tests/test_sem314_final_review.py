import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from src.semantic_foundation_review_final import validate_final_split,segment_groups,final_gates,require_engineering

ROOT=Path(__file__).resolve().parents[1]


class FinalReviewContracts(unittest.TestCase):
    def test_registered_split_and_existing_segment_coverage(self):
        split=json.loads((ROOT/"experiments/splits/sem_001_scaffold_static_v1.json").read_text())
        train,dev=validate_final_split(split)
        groups=segment_groups(json.loads((ROOT/"experiments/trajectory_segments.json").read_text()),dev)
        self.assertEqual(len(train),219);self.assertEqual(sum(map(len,groups.values())),74)
        self.assertTrue(all(groups.values()))
        split["semantic_train_labeled"][0]=dev[0]
        with self.assertRaisesRegex(ValueError,"overlap"):validate_final_split(split)

    def test_duplicate_segment_coverage_rejected(self):
        with self.assertRaises(ValueError):segment_groups({"segments":[{"segment_id":"a","images":["x"]},{"segment_id":"b","images":["x"]}]},["x"])

    def metrics(self,miou=.8,iou=.7,recall=.92):
        return {"mIoU_with_background":miou,"per_class":{"foundation":{"iou":iou,"recall":recall}}}

    def test_final_gates_reject_recall_or_local_collapse(self):
        cfg=json.loads((ROOT/"experiments/runs/SEM-314-DINOV2-FOUNDATION-REVIEW-FINAL/config.yaml").read_text())["developer_gates"]
        m={"baseline":self.metrics(),"candidate":self.metrics(.81,.73,.91)}
        g={"seg":copy.deepcopy(m)}; sub={"challenge":{"baseline_mean_miou":.8,"candidate_mean_miou":.81}}
        self.assertTrue(all(final_gates(m,g,sub,cfg).values()))
        bad=copy.deepcopy(m);bad["candidate"]["per_class"]["foundation"]["recall"]=.89
        self.assertFalse(final_gates(bad,g,sub,cfg)["recall_floor"])
        g["seg"]["candidate"]["per_class"]["foundation"]["iou"]=.68
        self.assertFalse(final_gates(m,g,sub,cfg)["seg_foundation_safety"])
        sub["challenge"]["candidate_mean_miou"]=.79
        self.assertFalse(final_gates(m,g,sub,cfg)["challenge_miou_safety"])

    def test_absent_foundation_group_is_not_invented_zero(self):
        cfg=json.loads((ROOT/"experiments/runs/SEM-314-DINOV2-FOUNDATION-REVIEW-FINAL/config.yaml").read_text())["developer_gates"]
        m={"baseline":self.metrics(),"candidate":self.metrics(.81,.73,.91)}
        g={"empty":{"baseline":self.metrics(iou=None),"candidate":self.metrics(iou=None)}}
        self.assertNotIn("empty_foundation_safety",final_gates(m,g,{},cfg))

    def test_engineering_identity_and_actual_final_probes_required(self):
        a={"status":"passed_final_review_engineering","prediction_result_sha256":"p","training_result_sha256":"t",
           "predictions_checked":74,"native_replays":[{},{}],"external_probes":[{},{},{}]}
        require_engineering(a,"p","t")
        for changed in ({"status":"failed"},{"native_replays":[]},{"training_result_sha256":"wrong"}):
            with self.assertRaises(ValueError):require_engineering({**a,**changed},"p","t")


try:
    from scripts import run_sem314_final_review as runner
except ImportError:
    runner=None


@unittest.skipIf(runner is None,"optional model runtime unavailable")
class FinalReviewAccessTest(unittest.TestCase):
    def test_failed_engineering_opens_no_developer_labels(self):
        args=SimpleNamespace(audit_dir=Path("audit"),train_dir=Path("train"),prediction_dir=Path("pred"))
        with patch.object(runner,"prediction_context",return_value=({},[],{},{},{},{},{},{})), \
             patch.object(runner,"read_json",return_value={"status":"failed"}), \
             patch.object(runner,"sha256",return_value="hash"),patch.object(runner,"png") as decode:
            with self.assertRaisesRegex(ValueError,"engineering gate"):runner.score(args,0)
            decode.assert_not_called()
