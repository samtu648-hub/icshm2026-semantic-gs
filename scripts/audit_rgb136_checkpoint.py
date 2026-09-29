#!/usr/bin/env python3
"""CPU audit of exact saved training state; no optimization or image generation."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,write_new_json
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--iteration',type=int,required=True);a=p.parse_args()
import torch
torch.set_num_threads(4)
path=a.run/'model'/f'chkpnt{a.iteration}.pth';state,iteration=torch.load(path,map_location='cpu',weights_only=False);assert iteration==a.iteration and state['embedding_appearance'] is None and state['mlp_feature_bank'] is None;groups=[g['name'] for g in state['optimizer']['param_groups']];assert 'embedding_appearance' not in groups
count=0;elements=0
def check(x):
 global count,elements
 if isinstance(x,torch.Tensor):assert torch.isfinite(x).all();count+=1;elements+=x.numel()
 elif isinstance(x,dict):
  for y in x.values():check(y)
 elif isinstance(x,(tuple,list)):
  for y in x:check(y)
check(state);r=dict(iteration=iteration,status='passed_finite_saved_state',checkpoint_sha256=sha256(path),script_sha256=sha256(Path(__file__)),tensor_count=count,elements=elements,anchors=len(state['anchor']),offsets_per_anchor=state['offset'].shape[1],optimizer_groups=groups,appearance_absent=True);write_new_json(a.run/f'checkpoint_audit_{iteration}.json',r);print(json.dumps(r))
