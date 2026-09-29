import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


class FinalVetoContracts(unittest.TestCase):
    def test_external_reader_accepts_camera_sized_id_map(self):
        try:
            import sys
            import numpy as np
            from PIL import Image
            sys.path.insert(0,str(ROOT/"scripts"))
            from run_sem329_external import external_label_array
        except ImportError:
            self.skipTest("requires optional native runtime")
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"small.png"
            expected=(np.arange(240*320).reshape(240,320)%5).astype(np.uint8)
            Image.fromarray(expected).save(path)
            np.testing.assert_array_equal(external_label_array(path),expected)
            Image.fromarray(np.full((240,320),5,np.uint8)).save(path)
            with self.assertRaises(ValueError):external_label_array(path)

    def test_native_predict_and_audit_are_unchanged_oof_implementation(self):
        def functions(path):
            tree=ast.parse(path.read_text())
            return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
        old=functions(ROOT/"scripts/run_sem328_second_source.py")
        new=functions(ROOT/"scripts/run_sem329_final.py")
        for name in ("predict","verify_prediction","invariant","summarize"):
            self.assertEqual(old[name],new[name])

    def test_current_rule_external_adapter_and_no_legacy_modification(self):
        import json,hashlib
        cfg=json.loads((ROOT/"experiments/runs/SEM-329-SECOND-SOURCE-VETO-FINAL/config.yaml").read_text())
        self.assertEqual(hashlib.sha256((ROOT/cfg["veto_core"]).read_bytes()).hexdigest(),cfg["veto_core_sha256"])
        self.assertEqual(cfg["split_id"].split(';')[0],"sem_001_scaffold_static_v1")
        code=(ROOT/"scripts/run_sem329_external.py").read_text()
        self.assertIn('original=legacy.probe(old)',code)
        self.assertIn('"target_is_distorted":True',code)
        self.assertIn('review_second_source(baseline,dense,**kwargs,independent=True)',code)

    def test_failed_external_does_not_decode_target_labels(self):
        try:
            import sys
            sys.path.insert(0,str(ROOT/"scripts"))
            import run_sem329_final as runner
        except ImportError:
            self.skipTest("requires optional Torch runtime")
        from unittest.mock import patch
        from types import SimpleNamespace
        args=SimpleNamespace(prediction_dir=Path("prediction"),audit_dir=Path("audit"),external_dir=Path("external"),config=Path("config"))
        audit={"status":"passed_independent_second_source_audit","prediction_result_sha256":"h","config_sha256":"h"}
        external={"status":"failed","config_sha256":"h","probes":[]}
        with patch.object(runner,"read_json",side_effect=[{},audit,external]),patch.object(runner,"sha256",return_value="h"),patch.object(runner,"label_array") as decode:
            with self.assertRaises(ValueError):runner.require_engineering(args)
            decode.assert_not_called()


if __name__=="__main__":unittest.main()
