#!/usr/bin/env python3
"""Matched full-fit statistics plus independent-loss sparse checks, no new selection."""
import json,sys,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_sem373_joint import write
BASE=ROOT/'outputs/quality/SEM-374'
def read(p):return json.loads(p.read_text())
def main():
 rs={a:read(BASE/('eval-'+a+'-r2/result.json')) for a in ['baseline','control','joint']};out={'rgb':{a:r['rgb_metrics'] for a,r in rs.items()},'miou':{a:r['semantic_metrics']['mIoU_with_background'] for a,r in rs.items()},'class_rgb_mae':{a:r['class_rgb_mae'] for a,r in rs.items()},'geometry':{a:r['geometry'] for a,r in rs.items()}}
 out['joint_minus_control']={k:rs['joint']['rgb_metrics'][k]-rs['control']['rgb_metrics'][k] for k in ['PSNR','SSIM','LPIPS','MAE']};out['joint_minus_control']['miou_pp']=100*(out['miou']['joint']-out['miou']['control'])
 rows={a:{v['name']:v for v in r['per_view']} for a,r in rs.items()};out['psnr_improved']=sum(rows['joint'][n]['PSNR']>v['PSNR'] for n,v in rows['control'].items());out['psnr_degraded']=sum(rows['joint'][n]['PSNR']<v['PSNR'] for n,v in rows['control'].items())
 out['class_iou']={a:{n:v['IoU'] for n,v in r['semantic_metrics']['per_class'].items()} for a,r in rs.items()}
 out['regions']=[]
 cr={(v['name'],v['kind']):v for v in rs['control']['fixed_regions']}
 for v in rs['joint']['fixed_regions']:
  p=cr[v['name'],v['kind']];out['regions'].append(dict(name=v['name'],kind=v['kind'],rgb_mae_delta_255=255*(v['rgb_mae']-p['rgb_mae']),boundary_rgb_mae_delta_255=255*(v['boundary_rgb_mae']-p['boundary_rgb_mae']) if p['boundary_rgb_mae'] is not None else None,deck_to_background_control=p['semantic']['confusion'][1][0],deck_to_background_joint=v['semantic']['confusion'][1][0]))
 out['depth']={}
 for arm,path in [('baseline',BASE/'depth-baseline-r2/structure-depth.json'),('control',BASE/'eval-control-r2/structure-depth.json'),('joint',BASE/'eval-joint-r2/structure-depth.json')]:
  counts=collections.Counter();err=collections.Counter();alpha=collections.Counter()
  for v in read(path):
   for key in [v['role'],v['role']+'_'+str(v['class_id'])]:counts[key]+=v['count'];err[key]+=v['absolute_relative_inverse_depth_sum'];alpha[key]+=v['alpha_sum']
  out['depth'][arm]={k:dict(count=counts[k],absolute_relative_inverse_depth=err[k]/counts[k],mean_alpha=alpha[k]/counts[k]) for k in counts}
 out['resource_stages']={}
 for stage in ['preflight-r1','preflight-r2','preflight-r3','preflight-r4','control-r1','control-r2','joint-r2','eval-control-r2','eval-joint-r2','depth-baseline-r2']:
  r=read(BASE/stage/'result.json');out['resource_stages'][stage]={k:r.get(k) for k in ['status','seconds','training_seconds','successful_steps','peak_allocated_mib','peak_reserved_mib','lock_wait_seconds']}
 write(BASE/'summary.json',out);print(json.dumps({k:v for k,v in out.items() if k not in ['regions','geometry','resource_stages']},indent=2))
if __name__=='__main__':main()
