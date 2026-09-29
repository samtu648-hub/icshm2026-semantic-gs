"""SEM354 truth-assisted diagnostics only; never an inference policy."""
import numpy as np

BRANCHES = ('static', 'b2_3k', 'b2_6k', 'deck_positive',
            'stay_cable_positive', 'fullframe', 'm2f_fused',
            'crossview_base', 'crossview_veto')
SCOPES = ('core', 'core_context', 'all', 'lock_deck_stay_cable',
          'background_foundation', 'foundation_proposal',
          'approved_foundation05', 'tower_proposal')


def positive_proposal(binary, class_id):
    if class_id not in (1, 2) or not np.isin(binary, [0, 1]).all():
        raise ValueError('Expected a named binary positive class')
    return np.where(binary == 1, class_id, 5).astype(np.uint8)


def eligible_sources(scope, base, raw, maps, five):
    """Eligibility depends solely on saved predictions/probabilities, never GT."""
    keys = BRANCHES[:5] if scope == 'core' else BRANCHES[:6] if scope == 'core_context' else BRANCHES
    if scope in ('foundation_proposal', 'approved_foundation05', 'tower_proposal'):
        if scope == 'tower_proposal':
            ok = (raw == 2) & (maps['static'] == 3)
        else:
            ok = (raw == 3) & (maps['static'] == 4) & (five[4] >= .1)
            if scope == 'approved_foundation05':
                ok &= (five[3] - five[4]) <= .05
        return [('static', maps['static'], ok & (maps['static'] != base))]
    out = []
    for key in keys:
        proposal = maps[key]
        ok = (proposal < 5) & (proposal != base)
        if scope == 'lock_deck_stay_cable':
            ok &= np.isin(base, [0, 3, 4]) & np.isin(proposal, [0, 3, 4])
        elif scope == 'background_foundation':
            ok &= np.isin(base, [0, 4]) & np.isin(proposal, [0, 4])
        out.append((key, proposal, ok))
    return out


def count_matrix(gt, pred):
    return np.bincount((gt.astype(np.int64)*5 + pred).ravel(), minlength=25).reshape(5, 5)


def diagnose(base, raw, maps, five, gt):
    if base.shape != gt.shape or not np.isin(gt, range(5)).all():
        raise ValueError('Invalid target grid or class')
    cm = count_matrix(gt, base)
    cubes = {}
    bits = np.zeros(base.shape, np.int16)
    for i, key in enumerate(BRANCHES):
        proposal = maps[key]
        if proposal.shape != gt.shape or not np.isin(proposal, range(6)).all():
            raise ValueError('Invalid proposal grid/class')
        ids = (gt.astype(np.int64)*5 + base)*6 + proposal
        cubes[key] = np.bincount(ids.ravel(), minlength=150).reshape(5, 5, 6).tolist()
        bits |= ((proposal == gt).astype(np.int16) << i)
    wrong = base != gt
    codes = ((gt[wrong].astype(np.int64)*5 + base[wrong])*512 + bits[wrong])
    values, counts = np.unique(codes, return_counts=True)
    joint = [[int(v//2560), int(v//512 % 5), int(v % 512), int(n)] for v, n in zip(values, counts)]
    scopes = {}
    for scope in SCOPES:
        recover = np.zeros(base.shape, bool)
        harmed = np.zeros(base.shape, bool)
        eligible = np.zeros(base.shape, bool)
        for _, proposal, ok in eligible_sources(scope, base, raw, maps, five):
            recover |= ok & (proposal == gt)
            harmed |= ok & (base == gt) & (proposal != gt)
            eligible |= ok
        repair = np.bincount((gt[recover].astype(np.int64)*5 + base[recover]), minlength=25).reshape(5, 5)
        oracle = base.copy()
        oracle[recover] = gt[recover]
        direct = count_matrix(gt, oracle)
        algebra = cm - repair + np.diag(repair.sum(axis=1))
        if not np.array_equal(direct, algebra) or np.any(recover & ~wrong):
            raise ValueError('Oracle repair algebra or zero-harm contract failed')
        scopes[scope] = {'recovered_confusion': repair.tolist(), 'oracle_confusion': direct.tolist(),
                         'eligible_pixels': int(eligible.sum()),
                         'correct_control_pixels_exposed': np.bincount(gt[harmed], minlength=5).tolist()}
    return {'confusion': cm.tolist(), 'source_cubes': cubes, 'wrong_correctness_joint': joint, 'scopes': scopes}


def independent_diagnose(base, raw, maps, five, gt):
    """Explicit class intersections and flat masks, independent of bincount encoding."""
    b, r, y = base.ravel(), raw.ravel(), gt.ravel()
    props = {k: v.ravel() for k, v in maps.items()}
    f = five.reshape(5, -1)
    cm = [[int(np.count_nonzero((y == t) & (b == p))) for p in range(5)] for t in range(5)]
    cubes = {k: [[[int(np.count_nonzero((y == t) & (b == p) & (v == q)))
                   for q in range(6)] for p in range(5)] for t in range(5)] for k, v in props.items()}
    bit = np.zeros(len(y), np.int16)
    for i, key in enumerate(BRANCHES):
        bit[props[key] == y] += 2**i
    joint = []
    for t in range(5):
        for p in range(5):
            if t == p:
                continue
            unique, counts = np.unique(bit[(y == t) & (b == p)], return_counts=True)
            joint.extend([[t, p, int(v), int(n)] for v, n in zip(unique, counts)])
    scopes = {}
    for scope in SCOPES:
        selected = np.zeros(len(y), bool)
        exposed = np.zeros(len(y), bool)
        eligible = np.zeros(len(y), bool)
        if scope in ('foundation_proposal', 'approved_foundation05', 'tower_proposal'):
            q = props['static']
            ok = ((r == 2) & (q == 3)) if scope == 'tower_proposal' else ((r == 3) & (q == 4) & (f[4] >= .1))
            if scope == 'approved_foundation05':
                ok = (r == 3) & (q == 4) & (f[4] >= .1) & (np.subtract(f[3], f[4], dtype=np.float32) <= .05)
            pairs = [(q, ok & (q != b))]
        else:
            count = 5 if scope == 'core' else 6 if scope == 'core_context' else 9
            pairs = []
            for key in BRANCHES[:count]:
                q = props[key]
                ok = (q != 5) & (q != b)
                if scope == 'lock_deck_stay_cable':
                    ok &= (b != 1) & (b != 2) & (q != 1) & (q != 2)
                if scope == 'background_foundation':
                    ok &= ((b == 0) | (b == 4)) & ((q == 0) | (q == 4))
                pairs.append((q, ok))
        for q, ok in pairs:
            selected[np.flatnonzero(ok & (q == y))] = True
            exposed[np.flatnonzero(ok & (b == y))] = True
            eligible |= ok
        repair = [[int(np.count_nonzero(selected & (y == t) & (b == p))) for p in range(5)] for t in range(5)]
        out = b.copy()
        out[selected] = y[selected]
        ocm = [[int(np.count_nonzero((y == t) & (out == p))) for p in range(5)] for t in range(5)]
        scopes[scope] = {'recovered_confusion': repair, 'oracle_confusion': ocm,
                         'eligible_pixels': int(np.count_nonzero(eligible)),
                         'correct_control_pixels_exposed': [int(np.count_nonzero(exposed & (y == t))) for t in range(5)]}
    return {'confusion': cm, 'source_cubes': cubes, 'wrong_correctness_joint': joint, 'scopes': scopes}
