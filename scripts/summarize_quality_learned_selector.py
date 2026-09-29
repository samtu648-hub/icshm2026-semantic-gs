#!/usr/bin/env python3
"""Saved-count SEM355 closeout; no images, labels, fitting, or selection."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,write_new_json
from src.quality_semantic_audit import metrics

def read(p):return json.loads(Path(p).read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=read(a.batch/'result.json');rows=read(a.batch/'per_view.json');old={r['name']:r for r in read(ROOT/'outputs/quality/complementarity/SEM-354-r3/per_view.json')};comparison=read(ROOT/'outputs/quality/mask2former/SEM-341-r1/comparison/calibration/per_view.json')
    checks=0
    for r in rows:
        if r['confusions']['raw']!=old[r['name']]['raw']['confusion'] or r['confusions']['foundation005']!=old[r['name']]['selected']['confusion']:raise ValueError('Historical raw or selected comparator drift')
        if r['confusions']['historical_fused']!=comparison[r['name']]['metrics']['mask2former_6k/calibrated']['confusion']:raise ValueError('Historical fused drift')
        checks+=3
    groups=[result['pooled'],*result['folds'].values(),*result['heads'].values()]
    fractions=0
    for group in groups:
        for m in group['metrics'].values():
            cm=np.array(m['confusion'],dtype=object);iou=[]
            for i in range(5):
                t=int(cm[i,i]);fp=int(sum(cm[:,i])-t);fn=int(sum(cm[i,:])-t)
                if t+fp+fn:iou.append(Fraction(t,t+fp+fn))
            v=float(sum(iou)/len(iou))
            if abs(v-m['mIoU_with_background'])>1e-14:raise ValueError('Rational metric audit')
            fractions+=1
    cube=np.array(result['pooled']['gt_raw_selector_cube'],np.int64)
    corrected=sum(result['pooled']['corrected_by_class']);harmed=sum(result['pooled']['harmed_by_class'])
    wrongwrong=sum(int(cube[t,r,q]) for t in range(5) for r in range(5) for q in range(5) if t!=r and t!=q and r!=q)
    core=read(ROOT/'outputs/quality/complementarity/SEM-354-r3/result.json')['pooled']['raw']['scopes']['core']
    heads=[read(a.batch/r['head']/'result.json') for r in read(a.batch/'audit.json')['heads']]
    fits=[read(a.batch/r['head']/'fit.json') for r in read(a.batch/'audit.json')['heads']]
    if corrected>core['recovered_pixels']:raise ValueError('Candidate oracle containment')
    out=dict(experiment_id=result['experiment_id'],result_sha256=sha256(a.batch/'result.json'),summary_code_sha256=sha256(Path(__file__)),historical_comparator_integer_checks=checks,rational_pooled_fold_head_checks=fractions,corrected=corrected,harmed=harmed,wrong_to_other_wrong=wrongwrong,net_correct_pixels=corrected-harmed,core_oracle_recoverable=core['recovered_pixels'],fraction_core_recoverable_corrected=corrected/core['recovered_pixels'],core_oracle_delta_miou_pp=core['delta_miou_pp'],actual_delta_miou_pp=result['pooled']['delta_miou_pp'],fit_seconds=sum(f['fit_seconds'] for f in fits),head_seconds=sum(h['seconds'] for h in heads),peak_head_rss_mib=max(h['peak_rss_mib'] for h in heads),fallback_heads=[r['role']['id'] for r in fits if r['fallback']],real_fitted_heads=sum(not r['fallback'] for r in fits))
    write_new_json(a.output,out);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
