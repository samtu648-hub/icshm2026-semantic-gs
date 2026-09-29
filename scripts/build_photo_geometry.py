#!/usr/bin/env python3
"""RGB144: frozen LoFTR pairs, auditable multiview tracks, independent-camera checks."""
import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.spatial import cKDTree
from src.photo_geometry import center, epipolar_error, resized_to_colmap, triangulate_track
from src.quality_cameras import read_camera_requests
from src.quality_protocol import sha256
from scripts.prepare_rgb_colmap_scene import parse_colmap_images

FIXED = ['159.png','160.png','140.png','146.png','081.png','057.png','064.png','054.png','174.png']


def save(path, obj):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def select_pairs(scene, old_scene, output):
    cams = read_camera_requests(scene/'sparse/0')
    records = parse_colmap_images(old_scene/'sparse/0/images.txt')
    points = {}
    for line in (old_scene/'sparse/0/points3D.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            row=line.split(); points[int(row[0])] = [float(v) for v in row[1:4]]
    sets = {r.name: {o.point_id for o in r.observations if o.point_id in points} for r in records}
    names = sorted(cams); neighbor = {}
    for a in names:
        candidates = []
        for b in names:
            if a == b: continue
            shared = sorted(sets[a] & sets[b])
            if not shared: continue
            xyz = np.array([points[i] for i in shared])
            va, vb = xyz-center(cams[a]), xyz-center(cams[b])
            cosine = np.sum(va*vb,1)/(np.linalg.norm(va,axis=1)*np.linalg.norm(vb,axis=1))
            angle = float(np.median(np.degrees(np.arccos(np.clip(cosine,-1,1)))))
            if .5 <= angle <= 20:
                candidates.append({'camera':b,'shared_tracks':len(shared),'median_angle_deg':angle})
        candidates.sort(key=lambda q: (-q['shared_tracks'],q['camera']))
        chosen=[]
        for lo,hi in ((.5,3),(3,8),(8,20.0000001)):
            chosen += [c for c in candidates if lo <= c['median_angle_deg'] < hi][:2]
        chosen += [c for c in candidates if c not in chosen][:6-len(chosen)]
        neighbor[a]=chosen
    all_pairs=sorted({tuple(sorted((a,c['camera']))) for a in names for c in neighbor[a]})
    fixed_pairs=sorted({tuple(sorted((a,c['camera']))) for a in FIXED for c in neighbor[a]})
    result={'neighbors':neighbor,'fixed_pairs':fixed_pairs,'all_pairs':all_pairs,
            'selection':'shared tracks; median angle bins; no candidate-quality selection',
            'old_track_scene':str(old_scene.relative_to(ROOT)),
            'input_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in [scene/'sparse/0/cameras.txt',scene/'sparse/0/images.txt',old_scene/'sparse/0/images.txt',old_scene/'sparse/0/points3D.txt']}}
    save(output/'pairs.json',result)
    return result


def match_pairs(args, plan, phase):
    sys.path.insert(0,str(args.dependencies/'site'))
    import cv2
    import torch
    import kornia
    torch.set_num_threads(4); torch.manual_seed(0)
    weights=args.dependencies/'loftr_outdoor.ckpt'
    matcher=kornia.feature.LoFTR(pretrained=None)
    matcher.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True)['state_dict'])
    matcher=matcher.eval().cuda()
    for p in matcher.parameters(): p.requires_grad_(False)
    cams=read_camera_requests(args.scene/'sparse/0'); cache={}
    def photo(n):
        if n not in cache:
            im=cv2.imread(str(args.scene/'images'/n),cv2.IMREAD_GRAYSCALE)
            assert im.shape == (989,1320)
            cache[n]=cv2.resize(im,(1024,768),interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(cache[n]).cuda().float()[None,None]/255
    pairs=plan[phase+'_pairs']; target=args.output/'pairs'; target.mkdir(exist_ok=True)
    start=time.monotonic(); done=0
    for a,b in pairs:
        path=target/f'{Path(a).stem}-{Path(b).stem}.npz'
        if path.exists(): continue
        t=time.monotonic()
        with torch.inference_mode():
            left,right=photo(a),photo(b)
            f=matcher({'image0':left,'image1':right})
            f={k:v.cpu().numpy() for k,v in f.items()}
            rev=matcher({'image0':right,'image1':left})
            rev={k:v.cpu().numpy() for k,v in rev.items()}
        u,v=resized_to_colmap(f['keypoints0']),resized_to_colmap(f['keypoints1'])
        ru,rv=resized_to_colmap(rev['keypoints1']),resized_to_colmap(rev['keypoints0'])
        cycle=np.full(len(u),np.inf); rc=np.zeros(len(u))
        if len(ru) and len(u):
            # Require the same reverse pair at both endpoints, not independent NN matches.
            _,idx=cKDTree(np.column_stack((ru,rv))).query(np.column_stack((u,v)))
            cycle=np.maximum(np.linalg.norm(u-ru[idx],axis=1),np.linalg.norm(v-rv[idx],axis=1))
            rc=rev['confidence'][idx]
        epi=epipolar_error(u,v,cams[a],cams[b])
        reason=(f['confidence']<.8).astype(np.uint8) | ((rc<.8).astype(np.uint8)<<1) | ((cycle>1.5).astype(np.uint8)<<2) | ((epi>1.5).astype(np.uint8)<<3)
        np.savez_compressed(path, camera0=a,camera1=b,xy0=u,xy1=v,confidence=f['confidence'],reverse_confidence=rc,
                            cycle_px=cycle,epipolar_px=epi,reject_bits=reason,reverse_xy0=ru,reverse_xy1=rv,
                            seconds=time.monotonic()-t,input_sha256=[sha256(args.scene/'images'/a),sha256(args.scene/'images'/b)])
        done+=1
        print(json.dumps({'phase':phase,'pair':[a,b],'raw':len(u),'accepted':int((reason==0).sum()),'seconds':time.monotonic()-t}),flush=True)
        save(args.output/'progress.json',{'phase':phase,'cached_pairs':len(list(target.glob('*.npz'))),'new_pairs':done,'wall_seconds':time.monotonic()-start})
    save(args.output/f'{phase}-matching-resources.json',{'wall_seconds':time.monotonic()-start,'new_pairs':done,'total_pairs':len(pairs),
        'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
        'weight_sha256':sha256(weights),'kornia':kornia.__version__,'torch':torch.__version__})


def build_tracks(args,plan,phase):
    cams=read_camera_requests(args.scene/'sparse/0'); nodes=[]; edges=[]; bycam=defaultdict(list)
    for a,b in plan[phase+'_pairs']:
        p=np.load(args.output/'pairs'/f'{Path(a).stem}-{Path(b).stem}.npz')
        for j in np.flatnonzero(p['reject_bits']==0):
            n=len(nodes); conf=float(min(p['confidence'][j],p['reverse_confidence'][j]))
            for c,xy in [(a,p['xy0'][j]),(b,p['xy1'][j])]:
                bycam[c].append(len(nodes)); nodes.append({'camera':c,'xy':xy.tolist(),'confidence':conf})
            edges.append((n,n+1))
    parent=np.arange(len(nodes)); size=np.ones(len(nodes),dtype=int)
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):
        a,b=find(a),find(b)
        if a==b:return
        if size[a]<size[b]:a,b=b,a
        parent[b]=a;size[a]+=size[b]
    for a,b in edges: union(a,b)
    for cam,ids in bycam.items():
        xy=np.array([nodes[i]['xy'] for i in ids])
        for a,b in cKDTree(xy).query_pairs(1.5):union(ids[a],ids[b])
    groups=defaultdict(list)
    for i in range(len(nodes)):groups[find(i)].append(i)
    accepted=[]; reasons=Counter(); rejects=args.output/f'{phase}-rejected-tracks.jsonl'
    with rejects.open('w') as out:
        for stable_id,ids in enumerate(sorted(groups.values(),key=lambda g:min(g))):
            members=defaultdict(list)
            for i in ids: members[nodes[i]['camera']].append(nodes[i])
            obs=[]; conflict=False
            for c,values in sorted(members.items()):
                xy=np.array([o['xy'] for o in values]); mid=np.median(xy,axis=0)
                # Reject chains that merge spatially incompatible observations in one camera.
                if np.max(np.linalg.norm(xy[:,None]-xy[None,:],axis=2))>1.5:conflict=True
                obs.append({'camera':c,'xy':mid.tolist(),'confidence':min(o['confidence'] for o in values),'merged_observations':len(values)})
            rec={'observations':obs,'accepted':False,'reason':'duplicate_camera_conflict'} if conflict else triangulate_track(obs,cams)
            rec['id']=stable_id;rec['confidence']=min(o['confidence'] for o in obs);reasons[rec['reason']]+=1
            if rec['accepted']:accepted.append(rec)
            else:out.write(json.dumps(rec,allow_nan=False)+'\n')
    result={'phase':phase,'proposals':accepted,'summary':dict(reasons),'track_count':len(groups),'pair_count':len(plan[phase+'_pairs']),
            'provenance':{'pairs_sha256':sha256(args.output/'pairs.json'),'source_sha256':sha256(Path(__file__)),
                          'geometry_source_sha256':sha256(ROOT/'src/photo_geometry.py'),'config_sha256':sha256(args.config)},
            'scope':'training photographs; independent of Gaussian model, not an independent validation set'}
    save(args.output/('proposals.json' if phase=='all' else 'fixed-proposals.json'),result)
    print(json.dumps({'phase':phase,'summary':dict(reasons)}),flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--scene',type=Path,required=True);p.add_argument('--old-scene',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--dependencies',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--phase',choices=['select','fixed','all','merge-fixed','merge-all'],required=True)
    args=p.parse_args()
    for key in ('scene','old_scene','output','dependencies','config'):
        setattr(args,key,getattr(args,key).resolve())
    args.output.mkdir(parents=True,exist_ok=True)
    plan=json.loads((args.output/'pairs.json').read_text()) if (args.output/'pairs.json').exists() else select_pairs(args.scene,args.old_scene,args.output)
    if args.phase=='select':return
    phase=args.phase.removeprefix('merge-')
    if not args.phase.startswith('merge-'):
        with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX); match_pairs(args,plan,phase)
    build_tracks(args,plan,phase)


if __name__=='__main__':main()
