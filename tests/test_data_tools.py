from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_data import image_dimensions  # noqa: E402


class DataToolsTest(unittest.TestCase):
    def test_reads_png_dimensions_without_external_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.png"
            path.write_bytes(
                b"\x89PNG\r\n\x1a\n"
                + struct.pack(">I", 13)
                + b"IHDR"
                + struct.pack(">II", 1320, 989)
            )
            self.assertEqual(image_dimensions(path), (1320, 989))


if __name__ == "__main__":
    unittest.main()

