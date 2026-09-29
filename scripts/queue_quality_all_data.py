#!/usr/bin/env python3
"""Continue the authorized all-data chain after its one active RGB fit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,verify_hashes,write_new_json

def main():
    p=argparse.ArgumentParser()
    for n in ['artifact-root','rgb-run','batch']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--attempt',default='queue')
    a=p.parse_args()
    for n in ['artifact_root','rgb_run','batch']:setattr(a,n,getattr(a,n).resolve())
    if Path(a.attempt).name!=a.attempt:raise ValueError('Attempt must be a directory name')
    q=a.batch/a.attempt;q.mkdir(exist_ok=False);start=time.monotonic()
    frozen={x['path']:x['sha256'] for x in json.loads((ROOT/'experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json').read_text())['artifacts']}
    verify_hashes(a.artifact_root,frozen)
    code=['scripts/run_quality_all300.py','scripts/diagnose_quality_all_data.py','scripts/queue_quality_all_data.py']
    hashes={n:sha256(ROOT/n) for n in code}
    r=dict(status='waiting_for_registered_rgb120',stages=[],code_sha256=hashes,started_unix=time.time())
    write_new_json(q/'started.json',r)
    def status():
        tmp=q/'status.tmp';tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(q/'status.json')
    status();env=os.environ.copy();cuda=a.artifact_root/'r1/toolchain/cuda-12.8'
    env.update(CUDA_HOME=str(cuda),PATH=str(cuda/'bin')+os.pathsep+env.get('PATH',''),LD_LIBRARY_PATH=str(cuda/'lib64')+os.pathsep+env.get('LD_LIBRARY_PATH',''),
        PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',TORCH_CUDA_ARCH_LIST='12.0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',TORCH_HOME=str(a.artifact_root/'r1/weights/torch'))
    try:
        # Time is observational; only an actual failed RGB result blocks the chain.
        while not (a.rgb_run/'result.json').exists():time.sleep(15)
        if json.loads((a.rgb_run/'result.json').read_text())['status']!='completed_all400_rgb_pending_matched_semantics':raise RuntimeError('Parent RGB did not complete; keep all artifacts')
        base=[sys.executable,str(ROOT/'scripts/run_quality_all300.py'),'--artifact-root',str(a.artifact_root),'--rgb-run',str(a.rgb_run),'--batch',str(a.batch)]
        for stage in ['export','static','mask2former','b2','deck','stay_cable']:
            for replay in [False,True]:
                verify_hashes(ROOT,hashes);name=stage+('-replay' if replay else '')
                r.update(status='running_'+name);status();command=base+['--stage',stage]+(['--replay'] if replay else [])
                with (q/(name+'.log')).open('x') as log:subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                result=json.loads((a.batch/name/'result.json').read_text())
                if result['status']!='completed_training_diagnostic_only':raise RuntimeError('Stage incomplete')
                r['stages'].append(dict(stage=name,result_sha256=sha256(a.batch/name/'result.json'),seconds=result['seconds'],trained_steps=result['trained_steps']))
                status()
        r.update(status='fixed_fusion_and_visual_diagnostic');status()
        command=[sys.executable,str(ROOT/'scripts/diagnose_quality_all_data.py'),'--batch',str(a.batch),'--rgb-run',str(a.rgb_run)]
        with (q/'diagnostics.log').open('x') as log:subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        verify_hashes(a.artifact_root,frozen);verify_hashes(ROOT,hashes)
        r.update(status='completed_full_data_training_and_fixed9_diagnostics',frozen_artifacts_unchanged=len(frozen),diagnostic_result_sha256=sha256(a.batch/'diagnostics/result.json'))
    except BaseException as e:r.update(status='failed_preserved',error=f'{type(e).__name__}: {e}');raise
    finally:
        r['seconds']=time.monotonic()-start;status();write_new_json(q/'result.json',r)
        # Append factual completion, never rewrite historical results or claims.
        completed=r['status']=='completed_full_data_training_and_fixed9_diagnostics'
        note=('全400 RGB／全300既有语义训练及固定9图诊断完成；融合复用SEM363，非独立盲测。待人工查看新对照后研究表示/优化伪影。'
              if completed else '全数据队列中断，失败和已完成产物保留；参见queue/result.json，不代表方法质量被否定。')
        report=ROOT/'experiments/runs/SEM-370-ALL300/report.md'
        with report.open('a') as stream:
            stream.write('\n自动队列实际结果：'+note+'\n\n'+json.dumps(r,ensure_ascii=False,indent=2)+'\n')
        with (ROOT/'PROJECT_STATUS.md').open('a') as stream:stream.write('\n2026-09-15 全数据队列终态：'+note+'\n')
        with (ROOT/'TASKS.md').open('a') as stream:stream.write('\nSEM370 自动队列终态：'+r['status']+'；'+note+'\n')
        reg=ROOT/'experiments/registry.yaml';latest=json.loads(reg.read_text())
        for item in latest['experiments']:
            if item['experiment_id']=='SEM-370-ALL300':item.update(status=r['status'],queue_result_path=str((q/'result.json').relative_to(ROOT)))
            if item['experiment_id']=='RGB-120-ALL400' and (a.rgb_run/'result.json').exists():item['status']=json.loads((a.rgb_run/'result.json').read_text())['status']
        reg.write_text(json.dumps(latest,ensure_ascii=False,indent=2)+'\n')
        with (q/'repo_health.log').open('x') as log:
            health=subprocess.run(['bash','scripts/repo_health.sh'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        write_new_json(q/'health.json',dict(returncode=health.returncode,training_complete=completed))

if __name__=='__main__':main()
