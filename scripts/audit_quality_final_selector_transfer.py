#!/usr/bin/env python3
"""Independent native74 replay using an already audited final head. No fit."""
import argparse
from pathlib import Path
import resource
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.continue_quality_final_selector import execution, base, RECEIPT
import numpy as np
from PIL import Image
from scripts.diagnose_quality_selector_fit_gap import reference_prediction,cube
from src.quality_static_training import TrainingLabels
from src.quality_protocol import sha256,verify_hashes,write_new_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True);p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True);a=p.parse_args();started=time.time()
    rec=base.read(a.output/'started.json')
    verify_hashes(ROOT,{RECEIPT:rec['execution_receipt_sha256'],**rec['code_sha256']})
    c,receipt,saved,old=execution()
    seal=base.read(a.output/'predictions_sealed.json');verify_hashes(a.output,seal['files'])
    if base.read(a.output/'effective_config.json')!=c:raise ValueError('Effective dependency configuration differs')
    verify_hashes(a.output,old['fit_artifacts'])
    fit=base.read(a.output/'fit.json');previous=base.read(a.output/'audit_fit.json')
    if (previous['status']!='passed_independent_final_fit' or previous['model_sha256']!=sha256(a.output/'model.npz')
            or previous['fit_sha256']!=sha256(a.output/'fit.json') or fit['training_views']!=c['train_names']):
        raise ValueError('Missing accepted saved fit audit')
    with np.load(a.output/'model.npz',allow_pickle=False) as z:model={k:z[k] for k in z.files}
    sources=base.final_sources(c)
    labels=TrainingLabels(a.artifact_root/'r2/sem_001/labels',c['prediction_names'],
                         base.read(ROOT/'experiments/runs/SEM-001/label_manifest.json'),[1320,989])
    rows=[]
    for name in c['prediction_names']:
        base.guard(a,c);data=base.load(sources,name);actual=reference_prediction(data,model)
        stored=np.asarray(Image.open(a.output/'predictions'/name))
        if not np.array_equal(actual,stored):raise ValueError('Independent prediction differs: '+name)
        gt=labels.read(name);rows.append(dict(name=name,cube=cube(gt,data[0],stored,reference=True).tolist()))
        del data,gt
    verify_hashes(a.output,seal['files'])
    write_new_json(a.output/'audit.json',dict(status='passed_final_selector_audit',fit_audit=previous,rows=rows,
        native_prediction_replays=74,seconds=time.time()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        audit_code_sha256=sha256(Path(__file__)),new_cpu_fits=0))


if __name__=='__main__':main()
