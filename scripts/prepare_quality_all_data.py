#!/usr/bin/env python3
"""Materialize released all400 with unchanged RGB110 initialization and camera policy."""
import argparse
import json
import shutil
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_rgb_colmap_scene import find_image_paths, parse_colmap_images, parse_single_simple_radial_camera, write_colmap_text
from src.quality_protocol import sha256, write_new_json

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import cv2
    import numpy as np
    old=ROOT/'outputs/quality/runs/RGB-110-SCAFFOLD-CTRL-full-r1/scene'
    source=find_image_paths(a.dataset_root)
    records=parse_colmap_images(a.dataset_root/'camera_parameters/images.txt')
    names=sorted(source); labeled=sorted(p.name for p in (a.dataset_root/'images').glob('*.png'))
    assert len(names)==400 and len(labeled)==300 and set(labeled)<=set(names)
    assert set(r.name for r in records)==set(names)
    a.output.mkdir(parents=True,exist_ok=False)
    scene=a.output/'scene'; (scene/'images').mkdir(parents=True)
    _,w,h,f,cx,cy,k=parse_single_simple_radial_camera(a.dataset_root/'camera_parameters/cameras.txt')
    assert (w,h)==(1320,989)
    K=np.array([[f,0,cx],[0,f,cy],[0,0,1]],dtype=np.float64)
    coef=np.array([k,0,0,0,0],dtype=np.float64)
    hashes={}; regenerated=[]
    for name in names:
        prior=old/'images'/name; target=scene/'images'/name
        if prior.exists(): target.symlink_to(prior.resolve())
        else:
            raw=cv2.imread(str(source[name])); assert raw.shape==(h,w,3)
            assert cv2.imwrite(str(target),cv2.undistort(raw,K,coef,None,K))
            regenerated.append(name)
        hashes[name]=sha256(target)
    # The new observations affect photometric training only. Keep old DLT points.
    for r in records:r.observations=[]
    sparse=scene/'sparse/0'
    write_colmap_text(sparse,records,w,h,f,cx,cy,[])
    for name in ('points3D.ply','points3D.txt'):
        shutil.copy2(old/'sparse/0'/name,sparse/name)
    (sparse/'test.txt').write_text('')
    split=dict(schema_version=1,split_id='released_all400_rgb_all300_sem_v1',
        rgb_train=names,semantic_train=labeled,evaluation=[],
        semantic_train_labeled=labeled,semantic_validation_labeled=[],
        evaluation_policy='training-fit diagnostics only; former74 and guards now training; no independent holdout',
        dataset_manifest_sha256=sha256(ROOT/'data/manifests/contest_dataset_v1.json'))
    write_new_json(a.output/'split.json',split)
    write_new_json(a.output/'access.json',dict(train=names,evaluation=[],partition='released_all400_v1'))
    write_new_json(scene/'preparation_manifest.json',dict(
        source_coordinate_frame='official_colmap_reference',distortion_policy='same released K full-frame cv2.undistort; no crop',
        counts=dict(training_images=400,evaluation_images=0,semantic_labels=300),
        initialization='RGB110 unchanged 303-derived DLT; only photometric image coverage expanded',
        initial_ply_sha256=sha256(sparse/'points3D.ply'),regenerated_images=regenerated,image_hashes=hashes,
        source_camera_hashes={n:sha256(a.dataset_root/'camera_parameters'/n) for n in ('cameras.txt','images.txt')}))
    print(json.dumps(dict(output=str(a.output),images=400,labels=300,regenerated=len(regenerated))))

if __name__=='__main__':main()
