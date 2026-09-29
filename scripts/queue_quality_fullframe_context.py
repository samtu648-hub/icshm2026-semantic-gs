#!/usr/bin/env python3
"""SEM346 serial native probe replay then calibration and exact fresh replay."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,write_new_json

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['artifact-root','probe','output']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 c=ROOT/'experiments/runs/SEM-346-FULLFRAME-CONTEXT-DIAG/config.yaml';cfg=json.loads(c.read_text());r={'status':'running','config_sha256':sha256(c),'script_sha256':sha256(Path(__file__)),'stages':[]};write_new_json(a.output/'started.json',r)
 probe_replay=a.output/'probe-replay';plan=[('probe',0,probe_replay,a.probe)]
 for fold in range(3):plan += [('full',fold,a.output/f'full-oof-{fold}',None),('full',fold,a.output/f'replay-oof-{fold}',a.output/f'full-oof-{fold}')]
 try:
  for mode,fold,dest,source in plan:
   cmd=[sys.executable,str(ROOT/'scripts/run_quality_fullframe_context.py'),'--artifact-root',str(a.artifact_root),'--output',str(dest),'--mode',mode,'--fold',str(fold)]
   if source:cmd+=['--replay-from',str(source)]
   if mode=='full':cmd+=['--probe-replay',str(probe_replay)]
   with (a.output/(dest.name+'.log')).open('xb') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=cfg['resource']['maximum_seconds_per_stage'])
   rr=json.loads((dest/'result.json').read_text());assert rr['status']=='passed_fullframe_context_stage'
   r['stages'].append({'directory':str(dest),'result_sha256':sha256(dest/'result.json')});print(json.dumps({'completed':str(dest)}),flush=True)
  r['status']='completed_fullframe108_and_exact_fresh_replay'
 except BaseException as e:r.update(status='failed_fullframe_queue',error=f'{type(e).__name__}: {e}');raise
 finally:write_new_json(a.output/'result.json',r)
if __name__=='__main__':main()
