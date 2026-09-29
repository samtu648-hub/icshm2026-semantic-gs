"""SEM345 descriptive partitions, not a prediction cleanup policy."""
import numpy as np
from scipy import ndimage


def detached_cable(raw):
    """8-connected foreground with no predicted deck/tower/foundation anchor."""
    labels,n=ndimage.label(raw>0,structure=np.ones((3,3),np.uint8))
    attached=np.zeros(n+1,bool)
    attached[np.unique(labels[np.isin(raw,[1,3,4])])]=True
    return (raw==2)&~attached[labels]


def evidence_masks(raw,static,binary):
    if raw.shape!=static.shape or raw.shape!=binary.shape:
        raise ValueError('Unaligned evidence')
    lane=raw==2;detached=detached_cable(raw)
    a=lane&(static==0);b=lane&(binary==0)
    return {'detached':detached,'static_background':a,'binary_background':b,
            'both_background':a&b,'detached_both_background':detached&a&b}


def truth_counts(mask,target):
    return np.bincount(target[mask],minlength=5).astype(np.int64)


def largest_component(mask):
    labels,n=ndimage.label(mask,structure=np.ones((3,3),np.uint8))
    if not n:return None
    counts=np.bincount(labels.ravel());counts[0]=0;ident=int(counts.argmax());sl=ndimage.find_objects(labels)[ident-1]
    return {'pixels':int(counts[ident]),'bbox_xyxy':[sl[1].start,sl[0].start,sl[1].stop,sl[0].stop]}


def joint_evidence(raw,static,binary,prob,target,bins):
    detached=detached_cable(raw)
    bucket=np.clip(np.searchsorted(bins,prob,side='right')-1,0,len(bins)-2)
    shape=(5,2,2,len(bins)-1,5)
    codes=((((static.astype(np.int64)*2+binary)*2+detached)*shape[3]+bucket)*5+target)
    return np.bincount(codes[raw==2],minlength=np.prod(shape)).reshape(shape)
