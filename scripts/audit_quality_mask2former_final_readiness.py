#!/usr/bin/env python3
"""Independent SEM349 CPU readiness audit: saved counts/roles/RNG, no images/GPU."""
import argparse
import hashlib
import json
import random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=read(a.preflight/'result.json');cpath=ROOT/'experiments/runs/SEM-349-MASK2FORMER-FINAL/config.yaml';c=read(cpath)
    assert r['status']=='ready_cpu_roles_sampling_budget_no_gpu_execution'
    assert r['binding']['config_sha256']==sha(cpath)
    for n,h in r['artifacts'].items():assert sha(a.preflight/n)==h
    for n,h in {**c['input_hashes'],**c['code_sha256']}.items():assert sha(ROOT/n)==h
    split=read(ROOT/'experiments/splits/sem_001_scaffold_static_v1.json')
    train=sorted(split['semantic_train_labeled']);development=sorted(split['semantic_validation_labeled'])
    assert len(train)==219 and len(development)==74 and not set(train)&set(development)
    assert r['access']['train']==r['training_label_views']==c['train_names']==train
    assert r['access']['prediction']==c['prediction_names']==development
    entries=read(ROOT/'experiments/runs/SEM-001/label_manifest.json')['records']
    chosen=[e for e in entries if e['image'] in set(train)]
    expected=[sum(e['pixel_counts'].get(str(i),0) for e in chosen) for i in range(5)]
    assert expected==r['class_counts'] and sum(expected)==219*989*1320
    rows=read(a.preflight/'sampling.json');assert len(rows)==6000
    rng=random.Random(c['seed']);digest=hashlib.sha256();seen=set()
    for row in rows:
        expected={'name':rng.choice(train),'top':rng.randrange(222),'left':rng.randrange(553),'flip':rng.uniform(0,1)<.5}
        assert row==expected and row['name'] not in development
        seen.add(row['name']);digest.update(json.dumps(row,sort_keys=True,separators=(',',':')).encode()+b'\n')
    assert digest.hexdigest()==r['sampling_sha256'] and len(seen)==r['sampled_training_views']==219
    assert r['target_labels_decoded'] is False and r['trained_steps']==0
    result=dict(status='passed_independent_final_readiness_no_gpu',config_sha256=sha(cpath),
                preflight_sha256=sha(a.preflight/'result.json'),audit_code_sha256=sha(Path(__file__)),
                training_views=219,prediction_views=74,sampling_steps=6000,all_training_views_sampled=True,
                target_labels_decoded=False,new_training=False)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
