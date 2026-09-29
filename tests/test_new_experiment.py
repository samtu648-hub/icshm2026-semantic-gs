from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from new_experiment import create_experiment  # noqa: E402


class NewExperimentTest(unittest.TestCase):
    def make_repo(self, root: Path) -> None:
        (root / "experiments" / "templates").mkdir(parents=True)
        (root / "experiments" / "runs").mkdir()
        (root / "experiments" / "registry.yaml").write_text(
            json.dumps({"schema_version": 1, "experiments": []}),
            encoding="utf-8",
        )
        (root / "experiments" / "templates" / "config.yaml").write_text(
            '{"experiment_id": "__EXPERIMENT_ID__", "title": "__TITLE__"}\n',
            encoding="utf-8",
        )
        (root / "experiments" / "templates" / "report.md").write_text(
            "# __EXPERIMENT_ID__ — __TITLE__\n",
            encoding="utf-8",
        )

    def test_creates_files_and_registry_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_repo(root)
            run_dir = create_experiment(
                root=root,
                experiment_id="RGB-001",
                title="Baseline",
                phase="R1",
            )
            self.assertTrue((run_dir / "config.yaml").is_file())
            registry = json.loads(
                (root / "experiments" / "registry.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(registry["experiments"][0]["experiment_id"], "RGB-001")

    def test_rejects_duplicate_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_repo(root)
            create_experiment(root, "SEM-001", "First", "S0")
            with self.assertRaises(ValueError):
                create_experiment(root, "SEM-001", "Duplicate", "S0")

    def test_rejects_unsafe_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_repo(root)
            with self.assertRaises(ValueError):
                create_experiment(root, "../bad", "Bad", "R1")

    def test_rejects_unknown_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_repo(root)
            with self.assertRaises(ValueError):
                create_experiment(root, "SEM-002", "Child", "S1", parent="SEM-001")


if __name__ == "__main__":
    unittest.main()
