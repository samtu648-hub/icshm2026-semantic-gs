"""SEM357 supervision only. Never used to access evaluation labels at prediction."""
from fractions import Fraction
import numpy as np


def validate_fit_confusion(matrix):
    m = np.asarray(matrix)
    if m.shape != (5, 5) or not np.issubdtype(m.dtype, np.integer) or np.any(m < 0):
        raise ValueError('Expected nonnegative integer 5x5 fit confusion')
    if np.any(m.sum(1) == 0):
        raise ValueError('All five fit GT classes must have support; otherwise raw fallback')
    return m


def marginal_utility_table(matrix):
    m = validate_fit_confusion(matrix)
    tp = np.diag(m).astype(np.float64)
    union = (m.sum(0)+m.sum(1)-np.diag(m)).astype(np.float64)
    scale = float(m.sum())/5
    table = np.full((5, 5, 5), np.nan)
    for t in range(5):
        for r in range(5):
            if m[t, r] == 0:
                continue
            for a in range(5):
                if a == r:
                    table[t, r, a] = 0
                    continue
                if t == r:
                    removed = -1/union[r]
                else:
                    if union[r] <= 1:
                        raise ValueError('Invalid removal union')
                    removed = tp[r]/(union[r]*(union[r]-1))
                added = 1/union[a] if t == a else -tp[a]/(union[a]*(union[a]+1))
                table[t, r, a] = scale*(removed+added)
    return table


def reference_utility_table(matrix):
    """Direct integer matrix edits and exact rational mIoU; no marginal formula."""
    matrix = validate_fit_confusion(matrix)
    def exact(cm):
        return sum(Fraction(int(cm[k,k]),int(cm[k].sum()+cm[:,k].sum()-cm[k,k])) for k in range(5))/5
    before = exact(matrix)
    table = np.full((5,5,5), np.nan)
    for t in range(5):
        for r in range(5):
            if matrix[t,r] == 0:
                continue
            for a in range(5):
                edited = matrix.copy()
                edited[t,r] -= 1
                edited[t,a] += 1
                table[t,r,a] = float((exact(edited)-before)*int(matrix.sum()))
    return table
