import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

try:
    from scripts import run_sem313_foundation_review as runner
except ImportError:
    runner = None


@unittest.skipIf(runner is None,"optional model/audit runtime unavailable")
class ReviewProtocolTest(unittest.TestCase):
    def test_failed_calibration_never_decodes_check_labels(self):
        args=SimpleNamespace(config=Path("config"),run_dir=Path("build"),audit_dir=Path("audit"),
                             calibration_dir=Path("calibration"),mode="check")
        audit={"status":"passed_independent_review_engineering","build_result_sha256":"hash",
               "predictions_checked":438,"native_replays":[{}]*6,"review_external_probes":[{}]*4}
        cal={"scope":"calibration","passed":False}
        with patch.object(runner,"context",return_value=({}, {}, {}, [], {}, {}, {}, {})), \
             patch.object(runner,"verify_build",return_value=({}, {})), \
             patch.object(runner,"sha256",return_value="hash"), \
             patch.object(runner,"read_json",side_effect=[audit,cal]), \
             patch.object(runner,"png") as decode:
            with self.assertRaisesRegex(ValueError,"Check cannot open"):
                runner.score(args)
            decode.assert_not_called()

    def test_missing_engineering_never_decodes_labels(self):
        args=SimpleNamespace(config=Path("config"),run_dir=Path("build"),audit_dir=Path("audit"),mode="calibration")
        with patch.object(runner,"context",return_value=({}, {}, {}, [], {}, {}, {}, {})), \
             patch.object(runner,"verify_build",return_value=({}, {})), \
             patch.object(runner,"read_json",return_value={"status":"failed"}), \
             patch.object(runner,"png") as decode:
            with self.assertRaisesRegex(ValueError,"engineering gate"):
                runner.score(args)
            decode.assert_not_called()

    def test_output_authority_rejects_addition_or_protected_edit(self):
        base=np.array([[0,1,2,3,4]],np.uint8)
        for at,value in ((0,4),(1,0),(2,0),(3,0),(4,1)):
            bad=base.copy(); bad[0,at]=value
            with self.assertRaises(ValueError): runner.check_review_contract(base,bad)
        valid=base.copy(); valid[0,4]=0
        runner.check_review_contract(base,valid)
