import unittest
import numpy as np
from src.semantic_foundation_review import review_foundation


class FoundationReviewTest(unittest.TestCase):
    def test_no_additions_and_exact_protected(self):
        b=np.array([[0,1,2,3,4,4,4]],np.uint8)
        p=np.array([[1,0,0,0,.49,.5,1.]])
        expected=np.array([[0,1,2,3,0,4,4]],np.uint8)
        self.assertTrue(np.array_equal(review_foundation(b,p),expected))
        self.assertTrue(np.array_equal(b,[[0,1,2,3,4,4,4]]))

    def test_disabled_and_idempotent(self):
        b=np.array([[4,4,0]],np.uint8); p=np.array([[.1,.9,.9]])
        self.assertTrue(np.array_equal(review_foundation(b,p,enabled=False),b))
        out=review_foundation(b,p)
        self.assertTrue(np.array_equal(review_foundation(out,p),out))

    def test_invalid_inputs(self):
        b=np.array([[4]],np.uint8)
        for p in (np.array([[np.nan]]),np.array([[1.1]]),np.zeros((2,2))):
            with self.assertRaises(ValueError): review_foundation(b,p)
        with self.assertRaises(ValueError): review_foundation(b.astype(float),np.ones((1,1)))
        with self.assertRaises(ValueError): review_foundation(b,np.ones((1,1)),enabled=.5)
