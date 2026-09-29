"""A label-free, one-way background confirmation gate for SEM347."""
import numpy as np


def background_gate(raw, probabilities, threshold):
    if probabilities.shape != (5, *raw.shape):
        raise ValueError('Expected five native probability planes')
    result = raw.copy()
    if threshold is not None:
        if not 0 <= threshold <= 1:
            raise ValueError('Invalid threshold')
        mask = (raw == 2) & (probabilities.argmax(0) == 0)
        result[mask & (probabilities[0] >= threshold)] = 0
    return result
