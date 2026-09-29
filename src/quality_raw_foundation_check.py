"""Sealed SEM343 evaluation and descriptive component accounting."""
import numpy as np
from scipy import ndimage


def validate_selection(selection, parent, calibration):
    if (selection['experiment_id'] != 'SEM-343-RAW-FOUNDATION-GATE'
            or selection['selected_margin'] != 0.05
            or selection['roles'] != parent['roles']
            or selection['check_candidate_scores_seen_before_sealing']
            or selection['automatic_promotion']):
        raise ValueError('Selection differs from the authorized sealed0.05 candidate')
    roles = selection['roles']
    if (len(roles['calibration']) != 108 or len(roles['check']) != 111
            or len(set(roles['calibration']) | set(roles['check'])) != 219):
        raise ValueError('Changed historical108/111 access roles')
    row = next(r for r in calibration['curves'] if r['maximum_margin'] == 0.05)
    if not row['spatial_screen_pass'] or any(row[k] != v for k,v in selection['approved_calibration_counts'].items()):
        raise ValueError('Approved calibration evidence changed')
    return row


def components(mask):
    """All8-connected components; bounds are native x0,y0,x1,y1 exclusive."""
    if mask.ndim != 2 or mask.dtype != bool:
        raise ValueError('Expected2D Boolean mask')
    labels, count = ndimage.label(mask, structure=np.ones((3,3),np.uint8))
    sizes = np.bincount(labels.ravel())
    rows = []
    for idx, bounds in enumerate(ndimage.find_objects(labels), 1):
        ys, xs = bounds
        rows.append({'pixels':int(sizes[idx]), 'bbox_xyxy':[xs.start,ys.start,xs.stop,ys.stop]})
    if sum(v['pixels'] for v in rows) != np.count_nonzero(mask) or len(rows) != count:
        raise ValueError('Local component counts do not reconcile')
    return sorted(rows,key=lambda v:(-v['pixels'],v['bbox_xyxy'][1],v['bbox_xyxy'][0]))


def evidence_masks(raw, candidate, target):
    changed = raw != candidate
    if np.any(changed & ((raw != 3) | (candidate != 4))):
        raise ValueError('Unregistered prediction modification')
    masks = {'corrected':changed & (target == 4), 'harmed':changed & (target == 3),
             'wrong_to_wrong':changed & (target != 3) & (target != 4)}
    encoded = np.zeros(raw.shape,np.uint8)
    for i, mask in enumerate(masks.values(),1):
        encoded[mask] = i
    return encoded, {k:components(v) for k,v in masks.items()}
