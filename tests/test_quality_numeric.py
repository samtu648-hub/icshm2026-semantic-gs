import unittest
import numpy as np
from src.quality_numeric import parameter_inventory, require_finite_inventory
from src.quality_backbone import array_signature


class QualityNumericTests(unittest.TestCase):
    def test_finite_does_not_alter_parameter_bytes(self):
        value = np.array([-40., 0., 1.], dtype=np.float32)
        before = value.tobytes()
        result = parameter_inventory({"scale": value})
        require_finite_inventory({"model.ply": result})
        self.assertEqual(before, value.tobytes())

    def test_nan_and_inf_saved_model_rejected_with_evidence(self):
        result = parameter_inventory({"xyz": np.array([np.nan, np.inf, -np.inf, 0.])})
        self.assertEqual(result["fields"]["xyz"]["nonfinite"], 3)
        with self.assertRaisesRegex(FloatingPointError, "xyz"):
            require_finite_inventory({"model.ply": result})

    def test_invalid_or_absent_parameters_rejected(self):
        for value in ({}, {"x": np.array([])}, {"x": np.array([object()])}):
            with self.assertRaises(ValueError):
                parameter_inventory(value)
        with self.assertRaises(ValueError):
            require_finite_inventory({})

    def test_adapter_identifies_bad_tensor_without_sanitizing(self):
        value = np.array([1., np.nan])
        with self.assertRaisesRegex(ValueError, "_xyz: 1/2"):
            array_signature({"_xyz": value})
        self.assertTrue(np.isnan(value[1]))
