import unittest
from pathlib import Path
from scripts.queue_quality_mask2former_oof import stages
from scripts.queue_quality_mask2former_comparison import require_completed_batch


class QueueTests(unittest.TestCase):
    def test_bounded_fold_order_and_separate_replays(self):
        plan = list(stages(Path("batch"), Path("artifacts"), ["oof-0", "oof-1", "oof-2"], "python"))
        self.assertEqual(len(plan), 12)
        for i in range(3):
            part = f"oof-{i}"
            rows = plan[i*4:i*4+4]
            self.assertEqual([r[0] for r in rows], [f"full-{part}", f"fresh-{part}-1", f"fresh-{part}-2", f"audit-{part}"])
            full = list(map(str, rows[0][1]))
            self.assertNotIn("--run", full)
            self.assertNotIn("--resume", full)
            self.assertEqual(full[full.index("--mode")+1], "full")
            for row in rows[1:3]:
                cmd = list(map(str, row[1]))
                self.assertEqual(cmd[cmd.index("--run")+1], f"batch/full-{part}")
            self.assertEqual(rows[-1][2], "completed_mask2former_fold_acceptance")

    def test_quality_release_requires_every_fresh_and_fold_acceptance(self):
        plan = list(stages(Path("batch"), Path("artifacts"), ["oof-0", "oof-1", "oof-2"], "python"))
        result = {"status": "completed_mask2former_threefold_acceptance_pending_comparison",
            "automatic_promotion": False, "target_labels_decoded": False,
            "events": [{"stage": n, "state": "completed"} for n, _, _ in plan]}
        require_completed_batch(result)
        for changed in (dict(result, events=result["events"][:-1]), dict(result, status="running"),
                        dict(result, target_labels_decoded=True), dict(result, automatic_promotion=True)):
            with self.assertRaises(ValueError): require_completed_batch(changed)


if __name__ == "__main__":
    unittest.main()
