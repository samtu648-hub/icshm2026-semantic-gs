"""SEM358 count-only descriptive statistics, no prediction policy."""
from collections import Counter
from fractions import Fraction
import math
import numpy as np
from src.quality_semantic_audit import metrics


def describe(values):
    values=np.asarray(values,dtype=np.float64)
    if not values.size or not np.isfinite(values).all():raise ValueError('Nonfinite/empty statistic')
    return dict(mean=float(values.mean()),median=float(np.median(values)),q25=float(np.quantile(values,.25)),q75=float(np.quantile(values,.75)),minimum=float(values.min()),maximum=float(values.max()))


def window_stats(rows):
    report={k:describe([r[k] for r in rows]) for k in ('loss','grad_norm','lr_factor','scale')}
    keys=sorted(rows[0]['losses'])
    if any(sorted(r['losses'])!=keys for r in rows):raise ValueError('Loss schema changes')
    report['loss_components']={k:describe([r['losses'][k] for r in rows]) for k in keys}
    report['loss_groups']={kind:describe([sum(v for k,v in r['losses'].items() if k.startswith('loss_'+kind)) for r in rows]) for kind in ('ce','mask','dice')}
    report['amp_skips']=sum(len(r['skips']) for r in rows)
    report['views']=dict(sorted(Counter(r['sample']['name'] for r in rows).items()))
    report['successful_updates']=len(rows)
    # A second accumulation method audits all means rather than repeating np.mean.
    for key in ('loss','grad_norm','lr_factor','scale'):
        if not math.isclose(report[key]['mean'],math.fsum(r[key] for r in rows)/len(rows),rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Independent mean')
    for key in keys:
        if not math.isclose(report['loss_components'][key]['mean'],math.fsum(r['losses'][key] for r in rows)/len(rows),rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Independent loss mean')
    return report


def distribution(cm):
    cm=np.asarray(cm,np.int64)
    if cm.shape!=(5,5) or np.any(cm<0) or cm.sum()==0:raise ValueError('Invalid confusion')
    total=int(cm.sum());gt=cm.sum(1);tp=np.diag(cm);union=gt+cm.sum(0)-tp;error=gt-tp
    ratio=lambda v:[float(Fraction(int(x),total)) for x in v]
    result=dict(pixels=total,gt=gt.tolist(),union=union.tolist(),errors=error.tolist(),gt_fraction=ratio(gt),union_fraction=ratio(union),error_fraction=ratio(error),error_rate_within_class=[float(Fraction(int(e),int(g))) if g else None for e,g in zip(error,gt)],metrics=metrics(cm))
    valid=[Fraction(int(tp[i]),int(union[i])) for i in range(5) if union[i]]
    if abs(float(sum(valid)/len(valid))-result['metrics']['mIoU_with_background'])>1e-14:raise ValueError('Rational mIoU differs')
    return result


def role_shift(fit, evaluation):
    f,e=distribution(fit),distribution(evaluation)
    out=dict(fit=f,evaluation=e)
    for key in ('gt_fraction','union_fraction','error_fraction'):
        out[key+'_evaluation_over_fit']=[b/a if a else None for a,b in zip(f[key],e[key])]
    out['gt_total_variation']=.5*sum(abs(a-b) for a,b in zip(f['gt_fraction'],e['gt_fraction']))
    return out


def core_counts(rows):
    raw=sum(np.array(r['raw']['confusion'],np.int64) for r in rows)
    repair=sum(np.array(r['raw']['scopes']['core']['recovered_confusion'],np.int64) for r in rows)
    joint=np.zeros((5,5),np.int64);allwrong=np.zeros((5,5),np.int64)
    for row in rows:
        for t,p,bits,n in row['raw']['wrong_correctness_joint']:
            if t==p:raise ValueError('Correct pixel in wrong joint')
            allwrong[t,p]+=n
            if bits & 31:joint[t,p]+=n
    wrong=raw-np.diag(np.diag(raw))
    if not np.array_equal(joint,repair) or not np.array_equal(allwrong,wrong):raise ValueError('Independent core bitset recount')
    common=wrong-repair
    oracle=raw-repair+np.diag(repair.sum(1))
    return dict(raw=distribution(raw),core_recovered=repair.sum(1).tolist(),core_common_errors=common.sum(1).tolist(),core_recovered_confusion=repair.tolist(),core_common_confusion=common.tolist(),recoverable_fraction_of_error=[int(a)/int(b) if b else None for a,b in zip(repair.sum(1),wrong.sum(1))],core_oracle_truth_assisted_only=distribution(oracle))
