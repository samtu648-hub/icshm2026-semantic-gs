#!/usr/bin/env python3
"""One serial SEM351 render/infer and full fresh replay, then CPU audit."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,verify_hashes,write_new_json
CONFIG='experiments/runs/SEM-351-FINAL-CAMERA-ACCEPTANCE/config.yaml'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['artifact-root','data-root','batch']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.batch=a.batch.resolve()
    if not a.batch.is_relative_to(ROOT/'outputs/quality/external_camera'):p.error('Isolated output required')
    a.batch.mkdir(parents=True,exist_ok=False);journal=a.batch/'queue';journal.mkdir()
    c=json.loads((ROOT/CONFIG).read_text());r={'experiment_id':c['experiment_id'],'status':'running',
        'config_sha256':sha256(ROOT/CONFIG),'trained_steps':0,'target_labels_decoded':False,'stages':[]}
    write_new_json(journal/'started.json',r);start=time.monotonic()
    try:
        for stage in ['render','infer','fresh-render','fresh-infer','audit']:
            verify_hashes(ROOT,c['code_sha256']);verify_hashes(ROOT,{CONFIG:r['config_sha256']})
            cmd=[sys.executable,str(ROOT/'scripts'/('audit_quality_final_camera.py' if stage=='audit' else 'run_quality_final_camera.py')),
                 '--artifact-root',str(a.artifact_root),'--data-root',str(a.data_root),'--batch',str(a.batch),'--output',str(a.batch/stage)]
            if stage!='audit':cmd+=['--stage',stage.removeprefix('fresh-')]
            if 'infer' in stage:cmd+=['--render-root',str(a.batch/'render')]
            if stage.startswith('fresh-'):cmd+=['--repeat-from',str(a.batch/stage.removeprefix('fresh-'))]
            with (journal/(stage+'.log')).open('x') as stream:
                subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=c['resource']['maximum_seconds_per_stage'])
            result=json.loads((a.batch/stage/'result.json').read_text())
            if not result['status'].startswith('passed_'):raise ValueError('Unaccepted stage')
            r['stages'].append({'stage':stage,'result_sha256':sha256(a.batch/stage/'result.json')})
            print(json.dumps(r['stages'][-1]),flush=True)
        r['status']='completed_final_camera_fresh_and_independent_audit'
    except BaseException as exc:r.update(status='failed_final_camera_queue',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        r['seconds']=time.monotonic()-start;write_new_json(journal/'result.json',r)


if __name__=='__main__':main()
