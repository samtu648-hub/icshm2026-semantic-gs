#!/usr/bin/env python3
"""Independent Python accumulation of SEM358 saved-log and role statistics."""
import argparse
from collections import Counter
from fractions import Fraction
import json
import math
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,verify_hashes,write_new_json

def read(p):return json.loads(Path(p).read_text())
def stats(values):
    a=sorted(values);n=len(a)
    def q(p):
        index=(n-1)*p;i=math.floor(index);alpha=index-i
        return a[i] if alpha==0 else (1-alpha)*a[i]+alpha*a[i+1]
    return dict(mean=math.fsum(a)/n,median=statistics.median(a),q25=q(.25),q75=q(.75),minimum=a[0],maximum=a[-1])
def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();c=read(ROOT/'experiments/runs/SEM-358-SAVED-BOTTLENECK-DIAG/config.yaml');d=read(a.batch/'diagnosis.json');rec=read(a.batch/'result.json');verify_hashes(ROOT,{**c['input_hashes'],**rec['code_sha256']});assert sha256(a.batch/'diagnosis.json')==rec['diagnosis_sha256'];checks=0
    for fold,paths in c['logs'].items():
        rows=[json.loads(x) for x in (ROOT/paths['log']).read_text().splitlines() if x.strip()];allviews=Counter()
        for lo,hi in c['windows']:
            selected=[r for r in rows if lo<=r['step']<=hi];w=d['training'][fold]['windows'][f'{lo:04d}-{hi:04d}'];assert len(selected)==1000
            values={k:[r[k] for r in selected] for k in ('loss','grad_norm','lr_factor','scale')}
            references={k:w[k] for k in values}
            for k in selected[0]['losses']:
                values['component/'+k]=[r['losses'][k] for r in selected];references['component/'+k]=w['loss_components'][k]
            for k in ('ce','mask','dice'):
                values['group/'+k]=[math.fsum(v for name,v in r['losses'].items() if name.startswith('loss_'+k)) for r in selected];references['group/'+k]=w['loss_groups'][k]
            for k,v in values.items():
                for metric,number in stats(v).items():assert math.isclose(number,references[k][metric],rel_tol=1e-12,abs_tol=1e-10),(fold,k,metric)
                checks+=1
            counter=Counter(r['sample']['name'] for r in selected);assert dict(counter)==w['views'];allviews.update(counter)
        assert dict(allviews)==d['training'][fold]['view_counts']
    ratios=0
    for head,s in d['role_shift'].items():
        f,e=s['fit'],s['evaluation'];tv=Fraction(0)
        for index in range(5):
            tv+=abs(Fraction(f['gt'][index],f['pixels'])-Fraction(e['gt'][index],e['pixels']))/2
            for vector,key in [('gt','gt_fraction'),('union','union_fraction'),('errors','error_fraction')]:
                if f[vector][index]:
                    ratio=Fraction(e[vector][index]*f['pixels'],f[vector][index]*e['pixels']);assert abs(float(ratio)-s[key+'_evaluation_over_fit'][index])<1e-12
                else:assert s[key+'_evaluation_over_fit'][index] is None
                ratios+=1
        assert abs(float(tv)-s['gt_total_variation'])<1e-12
    write_new_json(a.output,dict(status='passed',diagnosis_sha256=sha256(a.batch/'diagnosis.json'),code_sha256=sha256(Path(__file__)),distributions_checked=checks,view_window_counters=18,role_ratios_checked=ratios,total_variation_pairs=9,no_new_labels_or_predictions=True));print('audited distributions',checks,'ratios',ratios)
if __name__=='__main__':main()
