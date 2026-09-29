#!/usr/bin/env python3
"""Continue the authorized SEM379 run after training, preserving every failure."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/quality/SEM-379/r1'
p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);a=p.parse_args()
status={'status':'waiting_for_training','stages':[]};path=OUT/'completion-queue.json'
def save():path.write_text(json.dumps(status,indent=2)+'\n')
save()
try:
 while not (OUT/'mask2former/result.json').exists():time.sleep(20)
 r=json.loads((OUT/'mask2former/result.json').read_text());assert r['status']=='completed',r
 commands=[['scripts/run_sem379.py','--artifact-root',str(a.artifact_root),'--stage','inference'],['scripts/verify_sem379_rgb.py','--artifact-root',str(a.artifact_root)],['scripts/verify_sem379_checkpoints.py'],['scripts/audit_sem379.py'],['scripts/audit_sem379_boundaries.py'],['scripts/build_sem379_comparison.py']]
 for command in commands:
  name=Path(command[0]).stem;status['status']='running_'+name;save();start=time.monotonic()
  with (OUT/(name+'.log')).open('x') as f:subprocess.run([sys.executable,*command],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
  status['stages'].append({'name':name,'seconds':time.monotonic()-start});save()
 status['status']='artifacts_ready_pending_visual_review';save()
except BaseException as e:status['status']='failed_preserved';status['error']=f'{type(e).__name__}: {e}';save();raise
