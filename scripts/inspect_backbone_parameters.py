#!/usr/bin/env python3
"""Expand pinned upstream defaults plus the recorded actual training arguments."""
import argparse,ast,hashlib,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();record=json.loads((a.run/'started.json').read_text());cmd=record['command'];source=Path(cmd[cmd.index('--source')+1]);sys.path.insert(0,str(source))
from arguments import ModelParams,OptimizationParams,PipelineParams
parser=argparse.ArgumentParser();groups={n:cls(parser) for n,cls in [('model',ModelParams),('optimization',OptimizationParams),('pipeline',PipelineParams)]};args,_=parser.parse_known_args(cmd[cmd.index('--')+1:]);resolved={n:vars(g.extract(args)) for n,g in groups.items()}
resolved['arguments_source_sha256']=hashlib.sha256((source/'arguments/__init__.py').read_bytes()).hexdigest();resolved['method']=source.name;resolved['actual_command']=cmd
out=a.run/'resolved_parameters.json'
with out.open('x') as f:json.dump(resolved,f,indent=2);f.write('\n')
print(json.dumps({'method':source.name,'optimization':resolved['optimization']},indent=2))
