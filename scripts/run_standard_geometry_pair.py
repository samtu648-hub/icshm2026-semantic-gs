#!/usr/bin/env python3
"""Persistent serial smoke/full driver; refuses full run without observed placements."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--stage',choices=['smoke','full'],required=True);a=p.parse_args()
    out=ROOT/'outputs/quality/rgb144';iterations=1601 if a.stage=='smoke' else 30000
    diagnosis=json.loads((out/'diagnostic-r1/result.json').read_text());assert diagnosis['status']=='completed_readonly'
    assert len(json.loads((out/'geometry-r1/proposals.json').read_text())['proposals'])>0
    state={'stage':a.stage,'status':'running','stages':[]};dest=out/f'{a.stage}-driver-result.json'
    assert not dest.exists();start=time.monotonic()
    try:
        for arm,ident in [('control','RGB-145-STANDARD-GEOMETRY-CTRL'),('candidate','RGB-146-STANDARD-GEOMETRY-SPLIT')]:
            name=f'{arm}-smoke-r1' if a.stage=='smoke' else f'{arm}-r1'
            cmd=[sys.executable,str(ROOT/'scripts/run_standard_geometry.py'),'--artifact-root',str(a.artifact_root),
                 '--config',str(ROOT/f'experiments/runs/{ident}/config.yaml'),'--output',str(out/name),'--iterations',str(iterations)]
            tick=time.monotonic();subprocess.run(cmd,check=True,cwd=ROOT)
            state['stages'].append({'name':name,'seconds':time.monotonic()-tick});dest.write_text(json.dumps(state,indent=2))
        events=[json.loads(x) for x in (out/('candidate-smoke-r1' if a.stage=='smoke' else 'candidate-r1')/'densification.jsonl').read_text().splitlines()]
        state['guided_total']=events[-1]['guided_total']
        if a.stage=='smoke':state['status']='mechanism_active' if state['guided_total'] else 'mechanism_not_activated_stop'
        else:
            python=a.artifact_root/'r1/envs/r1-cuda128/bin/python'
            for arm,ident in [('control','RGB-145-STANDARD-GEOMETRY-CTRL'),('candidate','RGB-146-STANDARD-GEOMETRY-SPLIT')]:
                cmd=[str(python),str(ROOT/'scripts/evaluate_standard_geometry.py'),'--artifact-root',str(a.artifact_root),'--run',str(out/f'{arm}-r1'),'--experiment',ident]
                subprocess.run(cmd,check=True,cwd=ROOT);state['stages'].append({'name':arm+'-evaluation'});dest.write_text(json.dumps(state,indent=2))
            state['status']='training_and_evaluation_complete'
    except BaseException as exc:state.update(status='failed_preserved',error=repr(exc));raise
    finally:state['seconds']=time.monotonic()-start;dest.write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':main()
