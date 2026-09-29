#!/usr/bin/env python3
"""Independent cache/source audit and precise retained-RGB137 coverage accounting."""
import argparse
import json
from pathlib import Path
import sys
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256
from src.photo_geometry import project
from src.quality_cameras import read_camera_requests
from scripts.prepare_rgb_colmap_scene import parse_colmap_images


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output
    geo=json.loads((out/'geometry-r1/proposals.json').read_text());diag=json.loads((out/'diagnostic-r1/result.json').read_text())
    sparse=ROOT/'outputs/quality/rgb137/scene-r2/sparse/0';scene=ROOT/'outputs/quality/all_data/RGB-120-r2/scene';cams=read_camera_requests(scene/'sparse/0')
    point_ids={int(s.split()[0]) for s in (sparse/'points3D.txt').read_text().splitlines() if s and not s.startswith('#')}
    old={r.name:np.array([[o.x,o.y] for o in r.observations if o.point_id in point_ids]) for r in parse_colmap_images(sparse/'images.txt')}
    props={p['id']:p for p in geo['proposals']};coverage=[]
    for row in diag['summary']:
        positions=[next(o['xy'] for o in props[i]['observations'] if o['camera']==row['view']) for i in row['proposal_ids']]
        distances=cKDTree(old[row['view']]).query(positions)[0].tolist() if positions else []
        coverage.append({'region':row['region'],'new_coverage_gt3px_from_retained_old':sum(d>3 for d in distances),'distance_px':distances})
    # Keep original GPU output unchanged; corrected coverage is a separate audit.
    overlay=out/'retained-track-overlays';overlay.mkdir(exist_ok=True)
    for name in sorted({r['view'] for r in diag['summary']}):
        im=Image.open(scene/'images'/name).convert('RGB');d=ImageDraw.Draw(im)
        for x,y in old[name]:d.ellipse((x-1,y-1,x+1,y+1),fill=(60,130,220))
        for prop in props.values():
            for o in prop['observations']:
                if o['camera']==name:
                    x,y=o['xy'];uv,_=project(cams[name],np.array(prop['xyz']));d.ellipse((x-4,y-4,x+4,y+4),outline=(255,240,0),width=2);d.line((x,y,*uv),fill=(255,0,0),width=2)
        im.save(overlay/name)
    oldhash=json.loads((out/'preserved-inputs.json').read_text())
    for name,h in oldhash.items():assert sha256(ROOT/name)==h,name
    for name,h in json.loads((out/'cache-provenance.json').read_text()).items():assert sha256(ROOT/name)==h,name
    pointcheck=[]
    for prop in props.values():
        assert len({o['camera'] for o in prop['observations']})>=4
        for i,o in enumerate(prop['observations']):
            uv,z=project(cams[o['camera']],np.array(prop['xyz']));err=float(np.linalg.norm(uv-o['xy']));assert z>0 and err<=2
            assert abs(err-prop['reprojection_error_px'][i])<1e-8
            r=cams[o['camera']].pose.rotation;t=cams[o['camera']].pose.translation;c=-r.T@t;ray=np.array(prop['xyz'])-c;dist=np.linalg.norm(ray);ray/=dist
            relative=np.sqrt(ray@np.array(prop['covariance'])@ray)/dist;assert relative<=.02
            assert abs(relative-prop['relative_depth_std'][i])<1e-10
        assert prop['max_angle_deg']>=1
    pair_reasons=Counter();forward=0;accepted=0
    for file in (out/'geometry-r1/pairs').glob('*.npz'):
        q=np.load(file);bits=q['reject_bits'];forward+=len(bits);accepted+=int((bits==0).sum())
        for bit,name in [(1,'forward_confidence'),(2,'reverse_confidence'),(4,'cycle'),(8,'epipolar')]:pair_reasons[name]+=int(((bits&bit)!=0).sum())
    result={'status':'passed','accepted_proposals':len(props),'all_accepted_reprojection_and_uncertainty_recomputed':True,
            'old_models_and_cameras_unchanged':True,'cache_and_matching_source_hashes_verified':True,'retained_old_point_count':len(point_ids),
            'corrected_coverage':coverage,'correction':'Original diagnostic old coverage counted all nonnegative published observations, including 769 observations without an accepted RGB137 3D point. Original output retained; this audit and overlays use only retained point IDs. No matching, proposals, contributions or training changed.',
            'raw_forward_matches':forward,'accepted_pair_matches':accepted,'pair_rejection_counts_overlapping':dict(pair_reasons)}
    (out/'photo-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='corrected_coverage'},ensure_ascii=False))


if __name__=='__main__':main()
