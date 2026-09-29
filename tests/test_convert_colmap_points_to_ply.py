import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.convert_colmap_points_to_ply import read_points, write_ply


class ColmapPointsToPlyTest(unittest.TestCase):
    def test_text_values_and_binary_vertex_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "points3D.txt"
            target = root / "points3D.ply"
            source.write_text("# points\n7 1 2 3 4 5 6 0.1\n", encoding="utf-8")
            xyz, rgb = read_points(source)
            self.assertTrue(np.array_equal(xyz, np.asarray([[1, 2, 3]], np.float32)))
            self.assertTrue(np.array_equal(rgb, np.asarray([[4, 5, 6]], np.uint8)))
            write_ply(target, xyz, rgb)
            payload = target.read_bytes()
            header, binary = payload.split(b"end_header\n", 1)
            self.assertIn(b"element vertex 1", header)
            self.assertEqual(len(binary), 27)


if __name__ == "__main__":
    unittest.main()
