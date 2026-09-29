#!/usr/bin/env python3
"""Finish this authorized run after training; no retries, tuning, or further training."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,write_new_json
p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve();r=dict(status='waiting_for_current_training',commands=[],script_sha256=sha256(Path(__file__)),started_unix=time.time());write_new_json(run/'finish_queued.json',r);start=time.monotonic();env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
try:
 while not (run/'result.json').exists():
  if time.monotonic()-start>6000:raise TimeoutError('Training has not produced result; no new training launched')
  time.sleep(10)
 # Give the result writer a chance to close before reading; GPU lock gates work.
 time.sleep(1);assert json.loads((run/'result.json').read_text())['status']=='completed_all400'
 steps=[['scripts/verify_rgb136_uid.py','--artifact-root',str(a.artifact_root),'--run',str(run),'--iteration','30000']]
 steps.extend([['scripts/evaluate_rgb136.py','--artifact-root',str(a.artifact_root),'--run',str(run),'--stage',stage] for stage in ['export','rgb-evaluation','fresh']])
 steps.extend([['scripts/audit_rgb136_checkpoint.py','--run',str(run),'--iteration','30000'],['scripts/compare_rgb136.py','--run',str(run)],['scripts/video_rgb136.py','--run',str(run)]])
 for i,args in enumerate(steps):
  label=Path(args[0]).stem+('-'+args[-1] if args[0].endswith('evaluate_rgb136.py') else '')
  print('START',label,flush=True);t=time.monotonic();cmd=[sys.executable,*args]
  with (run/f'finish-{i}-{label}.log').open('x') as f:completed=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
  r['commands'].append(dict(command=cmd,returncode=completed.returncode,seconds=time.monotonic()-t,script_sha256=sha256(ROOT/args[0])))
  if completed.returncode:raise RuntimeError(f'{label} failed; preserved log and outputs')
  print('DONE',label,flush=True)
 r['status']='completed_training_endpoint_evaluation_and_browser'
except BaseException as e:r.update(status='failed_preserved',error=f'{type(e).__name__}: {e}');raise
finally:r['seconds']=time.monotonic()-start;write_new_json(run/'finish_result.json',r)
