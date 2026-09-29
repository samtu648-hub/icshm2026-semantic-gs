import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from src.quality_expert_final_pair import enforce_transfer_budget,validate_recipe


class FinalPairTests(unittest.TestCase):
    def test_space_has_no_fixed_reserve_but_covers_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            c={'study':{'budget':{'aggregate_root':'.','maximum_new_gib':4.5}}}
            with patch('shutil.disk_usage',return_value=SimpleNamespace(free=5*2**30)):
                self.assertEqual(enforce_transfer_budget(temp,c),0)
            with patch('shutil.disk_usage',return_value=SimpleNamespace(free=4*2**30)):
                with self.assertRaises(OSError):enforce_transfer_budget(temp,c)
            with patch('shutil.disk_usage',return_value=SimpleNamespace(free=100*2**30)):
                with self.assertRaises(OSError):enforce_transfer_budget(temp,c,pending_bytes=5*2**30)

    def test_final_role_cannot_change_optimizer(self):
        root=Path(__file__).resolve().parents[1]
        c=json.loads((root/'experiments/runs/SEM-362-B2-PAIRED-FINAL/config.yaml').read_text())
        validate_recipe(c,{})
        c['training']['learning_rate']*=2
        with self.assertRaises(ValueError):validate_recipe(c,{})
