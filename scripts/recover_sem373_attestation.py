#!/usr/bin/env python3
"""Attest already-completed SEM373 updates after tuple/list JSON audit mismatch."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_sem373_joint import digest,write

def main():
 import torch
 b=ROOT/'outputs/quality/SEM-373';out=b/'joint-r1/completion-attestation.json';assert not out.exists()
 ctrl=json.loads((b/'control-r1/result.json').read_text());joint=json.loads((b/'joint-r1/result.json').read_text());sem=json.loads((b/'semantic-r1/result.json').read_text())
 assert joint['status']=='failed_preserved' and joint['successful_steps']==2000 and joint['endpoint_reload_exact']
 assert joint['error']=='AssertionError()' and "ctrl['initial_state']==r['initial_state']" in joint['traceback']
 assert ctrl['initial_state']==joint['initial_state'] and ctrl['sampling_sha256']==joint['sampling_sha256']
 q=torch.load(b/'semantic-r1/endpoint.pt',map_location='cpu',weights_only=False)
 tuple_groups=[(type(g['betas']).__name__,g['betas']) for g in q['rgb_optimizer']['param_groups']]
 assert all(kind=='tuple' for kind,_ in tuple_groups)
 assert all(isinstance(g['betas'],list) for g in ctrl['initial_state']['rgb_groups'])
 assert ctrl['initial_state']['rgb_optimizer']==joint['initial_state']['rgb_optimizer']==sem['final_state']['rgb_optimizer']
 assert ctrl['initial_state']['semantic_optimizer']==joint['initial_state']['semantic_optimizer']==sem['final_state']['semantic_optimizer']
 for name in ['control','joint']:
  result=ctrl if name=='control' else joint
  assert digest(b/(name+'-r1')/'endpoint.pt')==result['endpoint_sha256']
  rows=[json.loads(x) for x in (b/(name+'-r1')/'training.jsonl').read_text().splitlines()]
  assert len(rows)==2000 and [x['step'] for x in rows]==list(range(1,2001))
 write(out,dict(status='completed_training_recovered_final_audit_only',original_result_sha256=digest(b/'joint-r1/result.json'),original_error_preserved=True,root_cause='Python optimizer betas tuple compared against JSON-decoded list; exact persisted states agree',verified_matched_initial_model_both_optimizers_and_sampling=True,verified_successful_steps_per_arm=2000,original_checkpoint_sha256=joint['endpoint_sha256'],additional_training_steps=0,tuple_type_evidence=tuple_groups))
 print(out)

if __name__=='__main__':main()
