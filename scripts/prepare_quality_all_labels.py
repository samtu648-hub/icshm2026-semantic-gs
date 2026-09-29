#!/usr/bin/env python3
"""Reuse293 existing aligned labels and materialize seven former guard labels."""
import argparse
import json
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256,write_new_json
from scripts.prepare_semantic_labels import read_camera,make_label

def main():
    p=argparse.ArgumentParser()
    for n in ['dataset-root','artifact-root','split','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    import numpy as np
    from PIL import Image
    split=json.loads(a.split.read_text());old=json.loads((ROOT/'experiments/runs/SEM-001/label_manifest.json').read_text())
    records={r['image']:r for r in old['records']};new=[]
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'labels').mkdir()
    camera,distortion=read_camera(a.dataset_root/'camera_parameters/cameras.txt')
    for name in split['semantic_train']:
        target=a.output/'labels'/name
        if name in records:
            src=a.artifact_root/'r2/sem_001/labels'/name
            assert sha256(src)==records[name]['label_png_sha256']
            shutil.copy2(src,target);new.append(records[name])
        else:
            source=a.dataset_root/'json'/(Path(name).stem+'.json')
            label,shapes=make_label(source,camera,distortion);Image.fromarray(label).save(target)
            ids,counts=np.unique(label,return_counts=True)
            new.append(dict(image=name,source_json='json/'+source.name,source_json_sha256=sha256(source),
                label_png='labels/'+name,label_png_sha256=sha256(target),shape_counts=dict(shapes),
                pixel_counts={str(int(k)):int(v) for k,v in zip(ids,counts)}))
    assert len(new)==300
    # Old aggregate totals are not valid for all300; build only current totals.
    result={k:old[k] for k in ['schema_version','class_mapping','source_camera','output_grid','coordinate_alignment','overlap_rule','ignore_policy','source_rgb_used']}
    result.update(label_manifest_id='sem370_all300_v1',split_id=split['split_id'],image_count=300,records=new,
        pixel_counts_train={str(k):sum(r['pixel_counts'].get(str(k),0) for r in new) for k in range(5)})
    write_new_json(a.output/'label_manifest.json',result)
    print(json.dumps(dict(reused=293,new=7,total=300)))

if __name__=='__main__':main()
