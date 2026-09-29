#!/usr/bin/env python3
"""Audit the exact public source snapshot and approved media before upload."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

from scan_secrets import PATTERNS


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',type=Path,required=True)
    parser.add_argument('--site',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--original-handoff',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();repo=args.repository.resolve();checks=[];failures=[]
    def check(ok,label):
        checks.append(label)
        if not ok:failures.append(label)
    manifest=json.loads((repo/'publication/source_manifest.json').read_text())
    names=[n for n in subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=repo,text=True).split('\0') if n]
    check(set(names)-{'publication/source_manifest.json'}==set(manifest['files']),'manifest matches exact Git candidates')
    for name in names:
        p=repo/name
        if not p.is_file():continue
        check(not p.is_symlink(),'no source symlink '+name)
        check(p.stat().st_size<50*1024**2,'source under file limit '+name)
        check(p.suffix.lower() not in {'.pt','.pth','.ckpt','.safetensors','.ply','.mp4','.mov','.zip','.doc','.docx','.pdf'},'no raw/large binary '+name)
        check(not (name=='.env' or name.startswith(('.ssh/','checkpoints/','logs/','data/Contest Dataset/'))),'allowed source path '+name)
        if name!='publication/source_manifest.json':check(sha(p)==manifest['files'][name]['public_sha256'],'published source hash '+name)
        if p.suffix=='.py':
            try:ast.parse(p.read_text());valid=True
            except SyntaxError:valid=False
            check(valid,'Python parses '+name)
    for base in [repo,args.site]:
        for p in base.rglob('*'):
            if not p.is_file() or any(x in p.parts for x in ['.git','__pycache__']):continue
            try:text=p.read_text()
            except UnicodeError:continue
            check(not re.search(r'/(?:home|Users)/[A-Za-z0-9_.-]+/',text),'no personal paths '+str(p.relative_to(base)))
            for label,pattern in PATTERNS.items():check(not pattern.search(text),'no '+label+' '+str(p.relative_to(base)))
            check(not re.search(r'\bssh\s+[^\s]+@(?:\d{1,3}\.){3}\d{1,3}',text),'no private SSH endpoint '+str(p.relative_to(base)))
    release=json.loads((repo/'publication/release.json').read_text())
    check(sha(args.archive)==release['sha256'],'Release archive matches pinned digest')
    records=json.loads((repo/'experiments/registry.yaml').read_text())['experiments']
    historical=json.loads((repo/'scripts/report_assets/paper_handoff_registry_20260923.json').read_text())
    check(len(historical)==351,'351 frozen historical entries')
    source_ids={e['experiment_id'] for e in historical}
    check(len([e for e in records if e['experiment_id'] in source_ids])==351,'historical entries retained in full registry')
    for e in historical:
        current=next(x for x in records if x['experiment_id']==e['experiment_id'])
        check(current.get('metrics')==e.get('metrics'),'scientific metrics retained '+e['experiment_id'])
    check(len([p for p in (args.site/'native').iterdir() if p.suffix.lower() in {'.png','.jpg'}])==44,'44 independent images')
    check(len(list((args.site/'videos').glob('*.mp4')))==8,'eight videos')
    if args.original_handoff:
        for folder in ['figures','native','videos']:
            for p in (args.site/folder).iterdir():
                if p.is_file():check(sha(p)==sha(args.original_handoff/folder/p.name),'selected media unchanged '+folder+'/'+p.name)
    result={'status':'failed' if failures else 'passed','checks':len(checks),'failures':failures,'source_files':len(names),
            'archive_sha256':release['sha256'],'no_training':True,'no_inference':True}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False));raise SystemExit(bool(failures))


if __name__=='__main__':main()
