#!/usr/bin/env python3
"""Launch the dedicated clean dependency/extension reproduction container."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('bundle','camera-metadata','reference-dir','cuda-root','output-dir'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--image',required=True)
    args=p.parse_args()
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    child=out/'container'
    child.mkdir()
    free=shutil.disk_usage(out).free
    if free<100*1024**3:
        raise RuntimeError('Disk safety floor: 100GiB required')
    inspected=json.loads(subprocess.check_output(['docker','image','inspect',args.image],text=True))[0]
    mounts=[(args.bundle,'/bundle'),(args.camera_metadata,'/camera-metadata'),
            (args.reference_dir,'/frozen-reference'),(args.cuda_root,'/cuda'),
            (ROOT/'reproduction/sem161','/work')]
    command=['docker','run','--rm','--network','none','--gpus','all','--shm-size','1g']
    for source,dest in mounts:
        if not source.is_dir():
            raise ValueError(f'Missing authorised input directory: {source}')
        command.extend(['--mount',f'type=bind,src={source.resolve()},dst={dest},readonly'])
    command.extend(['--mount',f'type=bind,src={child},dst=/out',args.image,'python','/work/clean_verify.py'])
    provenance={'experiment_id':'SUBMIT-040-CLEAN-UBUNTU-REPRO','image_id':inspected['Id'],
                'image_tag':args.image,'docker_base':'nvidia/cuda:12.8.1-base-ubuntu24.04',
                'disk_free_before_bytes':free,'network':'none during verification',
                'source_hashes':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (ROOT/'reproduction/sem161/clean_verify.py',ROOT/'reproduction/sem161/requirements.txt',ROOT/'reproduction/sem161/runtime_extras.txt',ROOT/'reproduction/sem161/Dockerfile')},
                'started_unix':time.time()}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    with (out/'container.log').open('w') as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line,end='',flush=True)
        code=process.wait()
    provenance.update(exit_code=code,finished_unix=time.time())
    (out/'execution.json').write_text(json.dumps(provenance,indent=2)+'\n')
    if code:
        raise RuntimeError('Clean reproduction failed; logs and partial outputs retained')


if __name__=='__main__':
    main()
