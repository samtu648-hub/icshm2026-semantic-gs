import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_r1_run.py"
SPEC = importlib.util.spec_from_file_location("summarize_r1_run", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SummarizeR1RunTest(unittest.TestCase):
    def test_parses_gnu_time_and_gpu_peak(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / "run.log"
            samples = root / "gpu.csv"
            log.write_text(
                "\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:02.50\n"
                "\tMaximum resident set size (kbytes): 1234\n"
                "\tExit status: 0\n",
                encoding="utf-8",
            )
            samples.write_text(
                "timestamp,utilization_gpu_percent,memory_used_mib,memory_total_mib\n"
                "a,1,100,1000\n"
                "b,1,321,1000\n",
                encoding="utf-8",
            )
            summary = MODULE.parse_time_log(log)
            peak = MODULE.peak_gpu_memory(samples)
        self.assertEqual(summary["wall_seconds"], 62.5)
        self.assertEqual(summary["host_max_rss_kib"], 1234)
        self.assertEqual(summary["wrapper_exit_status"], 0)
        self.assertEqual(peak, 321)


if __name__ == "__main__":
    unittest.main()
