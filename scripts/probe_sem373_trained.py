#!/usr/bin/env python3
"""Measure gradient competition at the frozen trained branch before either arm."""
import argparse,fcntl,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_sem373_joint import setup,load_stage,semantic,nll,rgb_loss,write,digest,CONFIG

def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);a=p.parse_args();c=json.loads(CONFIG.read_text());out=ROOT/c['output']/'trained-gradient-r1';out.mkdir(exist_ok=False);r=dict(status='running',config_sha256=digest(CONFIG),rows=[]);start=time.monotonic()
 try:
  with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);s=setup(a,c);t=s.torch;load_stage(s,ROOT/c['output']/'semantic-r1/endpoint.pt');s.model._xyz.requires_grad_(True)
   for name in c['diagnostic_views']:
    rg=t.autograd.grad(rgb_loss(s,s.views[name],name),s.model._xyz)[0];sg=t.autograd.grad(nll(s,semantic(s,s.views[name],('get_xyz',)),name),s.model._xyz)[0]
    assert t.isfinite(rg).all() and t.isfinite(sg).all() and rg.norm()>0 and sg.norm()>0
    dot=(rg*sg).sum();r['rows'].append(dict(name=name,rgb_xyz_norm=float(rg.norm()),semantic_xyz_norm=float(sg.norm()),weighted_semantic_to_rgb_norm_ratio=float(.1*sg.norm()/rg.norm()),cosine=float(dot/(rg.norm()*sg.norm())),rgb_gradient_dot_semantic=float(dot)))
   r.update(status='completed',interpretation='Fixed9 local gradient competition; no quality-driven recipe changes; zero optimizer steps',checkpoint_sha256=digest(ROOT/c['output']/'semantic-r1/endpoint.pt'))
 except BaseException as e:r.update(status='failed_preserved',error=repr(e),traceback=traceback.format_exc());raise
 finally:r['seconds']=time.monotonic()-start;write(out/'result.json',r)

if __name__=='__main__':main()
