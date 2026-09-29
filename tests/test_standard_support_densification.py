import unittest
from src.standard_support_densification import support_weight


class SupportWeightTests(unittest.TestCase):
    def test_unseen_is_neutral(self):
        self.assertEqual(support_weight(0., 0.), 1.)

    def test_low_exposure_shrinks_both_directions(self):
        self.assertLess(abs(support_weight(1., 1.)-1), abs(support_weight(100.,100.)-1))
        self.assertGreater(support_weight(-1.,1.), .9-1e-12)

    def test_opposing_views_cancel_without_rewarding_count(self):
        self.assertEqual(support_weight(0.,100.),1.)

    def test_bounds_and_direction(self):
        for count in [1,4,100,400,1e6]:
            self.assertGreater(support_weight(-count,count),.5)
            self.assertLess(support_weight(count,count),1.5)
            self.assertGreater(support_weight(count,count),1.)

if __name__=='__main__':unittest.main()
