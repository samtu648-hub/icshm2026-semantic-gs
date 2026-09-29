"""SEM337 relative-recall calibration; no labels or fitting in inference."""
import numpy as np
from src.quality_fusion import choose_threshold


def select_recall_preserving(rows, reference):
    """Exact TP comparison avoids rounded recall gates; all targets must match."""
    ref = np.asarray(reference['confusion'], dtype=np.int64)
    if ref.shape != (5, 5) or (ref < 0).any() or ref[4].sum() == 0:
        raise ValueError('Undefined foundation recall reference')
    admitted = []
    for row in rows:
        matrix = np.asarray(row['metrics']['confusion'], dtype=np.int64)
        if matrix.shape != (5, 5) or (matrix < 0).any() or not np.array_equal(matrix.sum(1), ref.sum(1)):
            raise ValueError('Calibration candidates have different targets')
        if matrix[4, 4] >= ref[4, 4]:
            admitted.append(row)
    if not admitted or not any(r['threshold'] is None for r in admitted):
        raise ValueError('No admissible off fallback; provenance or reference error')
    return choose_threshold(admitted, 4)


def independent_map(static, p4, cable, deck, threshold):
    """Direct construction separate from pipeline/replace_foundation helpers."""
    if static.dtype != np.uint8 or static.ndim != 2 or not static.size or static.max() > 4:
        raise ValueError('Invalid native class map')
    for prob in (p4, cable, deck):
        if prob.shape != static.shape or not np.isfinite(prob).all() or prob.min() < 0 or prob.max() > 1:
            raise ValueError('Invalid aligned probability')
    if threshold is not None and (not np.isfinite(threshold) or not 0 < threshold < 1):
        raise ValueError('Invalid threshold')
    result = static.copy()
    lane = np.isin(static, [0, 2])
    result[lane] = 0
    result[lane & (cable >= .75)] = 2
    result[lane & (cable < .75) & (deck >= .98)] = 1
    if threshold is not None:
        result[(static == 4) & (p4 < threshold)] = 0
    return result


class LabelRelease:
    """A current-run immutable calibration seal is required before check labels."""
    def __init__(self, folder):
        self.folder = folder
        self.seal_hash = None

    def seal(self):
        from src.quality_protocol import sha256
        if self.seal_hash is not None:
            raise ValueError('Calibration already sealed')
        self.seal_hash = sha256(self.folder / 'calibration_sealed.json')

    def require(self, role):
        from src.quality_protocol import sha256
        if role not in ('calibration', 'check'):
            raise ValueError('Development/unknown labels prohibited')
        if role == 'check' and (self.seal_hash is None or
                sha256(self.folder / 'calibration_sealed.json') != self.seal_hash):
            raise ValueError('Check labels require unchanged calibration seal')
