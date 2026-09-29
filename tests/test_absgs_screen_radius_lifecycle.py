import unittest


class ScreenRadiusLifecycleTest(unittest.TestCase):
    def test_extension_preserves_old_values_and_zeros_only_new_slots(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional torch runtime unavailable")
        from src.absgs_screen_radius_lifecycle import extend_radius_history

        old = torch.tensor([0.0, 21.0, 7.5])
        result = extend_radius_history(old, 5)
        self.assertTrue(torch.equal(result[:3], old))
        self.assertTrue(torch.equal(result[3:], torch.zeros(2)))

    def test_extension_rejects_point_count_reduction(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional torch runtime unavailable")
        from src.absgs_screen_radius_lifecycle import extend_radius_history

        with self.assertRaises(ValueError):
            extend_radius_history(torch.zeros(3), 2)


if __name__ == "__main__":
    unittest.main()
