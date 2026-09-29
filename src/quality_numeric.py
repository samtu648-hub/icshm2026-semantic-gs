"""Reject nonfinite learned parameters before RGB scores become eligible evidence."""
from __future__ import annotations
import numpy as np


def parameter_inventory(named_arrays):
    if not named_arrays:
        raise ValueError("Empty model parameter inventory")
    fields = {}
    for name, value in named_arrays.items():
        array = np.asarray(value)
        if array.dtype.kind not in "fiub" or array.size == 0:
            raise ValueError(f"Invalid model parameter field {name}")
        fields[name] = {"elements": int(array.size), "nonfinite": int((~np.isfinite(array)).sum())}
    return {"finite": all(v["nonfinite"] == 0 for v in fields.values()), "fields": fields}


def ply_inventory(path):
    from plyfile import PlyData
    data = PlyData.read(path)["vertex"].data
    return parameter_inventory({name: data[name] for name in data.dtype.names})


def require_finite_inventory(inventory):
    failures = {path: {name: value["nonfinite"] for name, value in report["fields"].items() if value["nonfinite"]}
                for path, report in inventory.items() if not report["finite"]}
    if failures:
        raise FloatingPointError(f"Nonfinite saved model parameters: {failures}")
    if not inventory:
        raise ValueError("No saved model parameter inventory")
