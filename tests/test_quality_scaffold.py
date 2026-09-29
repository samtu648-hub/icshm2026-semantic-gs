from pathlib import Path
import tempfile
import unittest
from src.quality_scaffold import saved_namespace


class QualityScaffoldTests(unittest.TestCase):
    def test_literal_checkpoint_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cfg_args"
            path.write_text("Namespace(n_offsets=10, use_feat_bank=False, source_path='relative/scene')")
            self.assertEqual(saved_namespace(path)["n_offsets"], 10)
            for text in ("Namespace(n_offsets=__import__('os'))", "object()", "Namespace(**{'x': 1})"):
                path.write_text(text)
                with self.assertRaises((ValueError, TypeError)):
                    saved_namespace(path)
