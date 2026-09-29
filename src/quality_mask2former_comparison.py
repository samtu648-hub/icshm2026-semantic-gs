"""SEM341 independent four-stage construction and matched comparison contracts."""
import numpy as np
from src.quality_fusion import validate_inputs

ARMS = ("b2_3k", "b2_6k", "mask2former_6k")
STAGES = ("fiveclass_lane", "cable", "foundation", "deck")
VARIANTS = tuple(f"{arm}/{mode}" for arm in ARMS for mode in ("raw", "fixed", "calibrated"))
CONTRASTS = {
    **{f"mask2former_vs_b2_6k/{mode}": (f"b2_6k/{mode}", f"mask2former_6k/{mode}")
       for mode in ("raw", "fixed", "calibrated")},
    **{f"calibration_effect/{arm}": (f"{arm}/fixed", f"{arm}/calibrated") for arm in ARMS},
    "mask2former_vs_b2_3k/fixed": ("b2_3k/fixed", "mask2former_6k/fixed"),
    "mask2former_vs_b2_3k/calibrated": ("b2_3k/calibrated", "mask2former_6k/calibrated"),
}


def independent_map(static, five, cable, deck, thresholds):
    """Direct region algebra; does not call pipeline/apply_stage.

    Stage2 overwrites the entire0/2 lane only when enabled. A class4 deleted by
    the foundation gate may NOT be filled by deck, even though it is now0.
    """
    validate_inputs(static, five, cable, deck)
    if set(thresholds) - set(STAGES):
        raise ValueError("Unknown stage")
    for value in thresholds.values():
        if value is not None and (not np.isfinite(value) or not 0 < value < 1):
            raise ValueError("Invalid threshold")
    out = static.copy()
    lane = (static == 0) | (static == 2)
    cable_t, five_t = thresholds.get("cable"), thresholds.get("fiveclass_lane")
    if cable_t is not None:
        out[lane] = (cable[lane] >= cable_t).astype(np.uint8)*2
    elif five_t is not None:
        out[lane] = (five[2, lane] >= five_t).astype(np.uint8)*2
    # This mask is captured BEFORE foundation deletion, matching provenance.
    deck_lane = out == 0
    foundation_t = thresholds.get("foundation")
    if foundation_t is not None:
        out[(static == 4) & (five[4] < foundation_t)] = 0
    deck_t = thresholds.get("deck")
    if deck_t is not None:
        out[deck_lane & (deck >= deck_t)] = 1
    return out


def validate_arms(arms):
    if tuple(arms) != ARMS:
        raise ValueError("Require equally treated registered3k/6k B2 and6k Mask2Former arms")


def assert_baseline_counts(evaluation, paired):
    for arm, endpoint in (("b2_3k", "3k"), ("b2_6k", "6k")):
        for mode, old in (("raw", "raw_"), ("fixed", "full_")):
            if evaluation[f"{arm}/{mode}"]["confusion"] != paired[old+endpoint]["confusion"]:
                raise ValueError("Frozen paired B2 baseline does not reproduce")


def class_transitions(old, new, target):
    """Exact old-class/new-class/target-class cube; unchanged pixels excluded."""
    if old.shape != new.shape or old.shape != target.shape:
        raise ValueError("Unaligned transition evidence")
    changed = old != new
    ids = old[changed].astype(np.int64)*25 + new[changed].astype(np.int64)*5 + target[changed]
    return np.bincount(ids, minlength=125).reshape(5, 5, 5)
