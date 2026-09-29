#!/usr/bin/env python3
"""Frozen RGB140: photo evidence, actual compositing weights and finite removals."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--geometry-cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--experiment',default='RGB-144-STANDARD-GEOMETRY-DIAG')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    import numpy as np
    from PIL import Image,ImageDraw
    from scipy.ndimage import binary_dilation,binary_erosion
    from scipy.spatial import cKDTree
    from src.quality_cameras import read_camera_requests
    from src.photo_geometry import project
    from scripts.prepare_rgb_colmap_scene import parse_colmap_images
    cache=json.loads(a.geometry_cache.read_text());proposals=cache['proposals'];cams=read_camera_requests(a.scene/'sparse/0')
    fixed=['159.png','160.png','140.png','146.png','081.png','057.png','064.png','054.png','174.png']
    roi=json.loads((ROOT/'experiments/runs/RGB-132-THREE-BACKBONE-DIAG/config.yaml').read_text())['regions']
    old_ids={int(line.split()[0]) for line in (ROOT/'outputs/quality/rgb137/scene-r2/sparse/0/points3D.txt').read_text().splitlines() if line and not line.startswith('#')}
    old={r.name:np.array([[o.x,o.y] for o in r.observations if o.point_id in old_ids]) for r in parse_colmap_images(ROOT/'outputs/quality/rgb137/scene-r2/sparse/0/images.txt')}
    obs={n:[] for n in fixed};regions={};summary=[]
    for proposal in proposals:
        for i,o in enumerate(proposal['observations']):
            if o['camera'] in obs:obs[o['camera']].append((proposal,i,o))
    for n in fixed:
        stem=Path(n).stem;label=np.asarray(Image.open(ROOT/f'outputs/quality/all_data/SEM-370-r1/labels/{n}'))
        for k,c in [('stay_cable',2),('deck',1)]:regions[f'{stem}-{k}']=(n,label==c)
        structure=label!=0;regions[f'{stem}-boundary']=(n,binary_dilation(structure,iterations=2)&~binary_erosion(structure,iterations=2))
        for key,r in roi.items():
            if key.startswith(stem+'-'):
                mask=np.zeros((989,1320),bool)
                for x0,y0,x1,y1 in r['boxes']:mask[y0:y1,x0:x1]=True
                regions[key]=(n,mask)
        im=Image.open(a.scene/'images'/n).convert('RGB');draw=ImageDraw.Draw(im)
        if len(old[n]):
            for x,y in old[n]:draw.ellipse((x-1,y-1,x+1,y+1),fill=(60,130,220))
        for pr,i,o in obs[n]:
            x,y=o['xy'];q,_=project(cams[n],np.array(pr['xyz']));
            draw.ellipse((x-4,y-4,x+4,y+4),outline=(255,240,0),width=2);draw.line((x,y,*q),fill=(255,0,0),width=2)
        im.save(a.output/f'{stem}-evidence.png')
    with (ROOT/'outputs/quality/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        source=ROOT/'outputs/quality/rgb140/source-r1'
        sys.path[:0]=[str(source),str(source/'submodules/diff-gaussian-rasterization'),str(source/'submodules/simple-knn')]
        import torch
        from types import SimpleNamespace
        from gaussian_renderer import GaussianModel,render
        from utils.general_utils import build_rotation
        from src.quality_native_gaussian import make_native_view
        torch.set_num_threads(4);torch.manual_seed(0)
        ply=ROOT/'outputs/quality/rgb140/full-r2/model/point_cloud/iteration_30000/point_cloud.ply';before=sha256(ply)
        model=GaussianModel(3);model.load_ply(str(ply));saved_opacity=model._opacity.detach().clone()
        for v in vars(model).values():
            if isinstance(v,torch.Tensor):v.requires_grad_(False)
        pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False,antialiasing=False);bg=torch.zeros(3,device='cuda')
        def pred(n,color=None):return render(make_native_view(cams[n]),model,pipe,bg,override_color=color)['render']
        def target(n):return torch.tensor(np.asarray(Image.open(a.scene/'images'/n)).copy(),device='cuda').permute(2,0,1).float()/255
        def png(im,path):Image.fromarray((im.detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)).save(path)
        ids=set();baseline={}
        for n in fixed:
            im=pred(n);gt=target(n);baseline[n]=im.detach();png(im,a.output/f'{Path(n).stem}-rgb140.png')
            png((im-gt).abs()*4,a.output/f'{Path(n).stem}-residual4x.png')
            for key,(view,mask) in regions.items():
                if n!=view:continue
                hits=[(pr,i,o) for pr,i,o in obs[n] if 0<=int(o['xy'][0])<1320 and 0<=int(o['xy'][1])<989 and mask[int(o['xy'][1]),int(o['xy'][0])]]
                evidence_mask=np.zeros_like(mask)
                for pr,i,o in hits:evidence_mask[int(o['xy'][1]),int(o['xy'][0])]=True
                contribution_mask=binary_dilation(evidence_mask,iterations=2)&mask if hits else mask
                m=torch.tensor(contribution_mask,device='cuda');colors=torch.ones_like(model.get_xyz,requires_grad=True)
                # d(sum of pixel colour)/d(per-Gaussian colour) is its actual alpha*T contribution.
                rgb=pred(n,colors);weights=torch.autograd.grad(rgb[0,m].sum(),colors)[0][:,0]
                top=torch.argsort(weights,descending=True,stable=True)[:2].tolist();ids.update(top)
                distances=cKDTree(old[n]).query(np.array([o['xy'] for _,_,o in hits]))[0].tolist() if hits and len(old[n]) else []
                summary.append({'region':key,'view':n,'accepted_proposals':len(hits),'proposal_ids':[pr['id'] for pr,_,_ in hits],
                    'new_coverage_gt3px_from_old':sum(d>3 for d in distances),'old_distance_px':distances,
                    'top_ids':top,'compositing_weight_sum':[float(weights[i]) for i in top],
                    'selection_scope':'2px neighborhood of reliable correspondences' if hits else 'ROI only; no reliable photo evidence',
                    'baseline_mae_255':float((im-gt).abs()[:,torch.tensor(mask,device='cuda')].mean())*255,
                    'classification':'evidence_insufficient' if not hits else 'pending_finite_removal_and_depth_proxy'})
                del rgb,colors,weights
            print('region contributions',n,flush=True)
        # At up to 12 deterministic locations per fixed view show exact composited centre-depth layers.
        layers=[];xyz=model.get_xyz.detach();rot=build_rotation(model._rotation);s=model.get_scaling
        cov=(rot*s[:,None,:]**2)@rot.transpose(1,2)
        for n in fixed:
            c=cams[n];r=xyz.new_tensor(c.pose.rotation);t=xyz.new_tensor(c.pose.translation);cp=xyz@r.T+t
            z=cp[:,2];uv=cp[:,:2]/z[:,None]*xyz.new_tensor([c.matrix[0,0],c.matrix[1,1]])+xyz.new_tensor([c.matrix[0,2],c.matrix[1,2]])
            j=torch.zeros((len(xyz),2,3),device='cuda');j[:,0,0]=c.matrix[0,0]/z;j[:,1,1]=c.matrix[1,1]/z
            j[:,0,2]=-c.matrix[0,0]*cp[:,0]/z**2;j[:,1,2]=-c.matrix[1,1]*cp[:,1]/z**2;j=j@r
            v=j@cov@j.transpose(1,2);det=(v[:,0,0]*v[:,1,1]-v[:,0,1]*v[:,1,0]).clamp_min(1e-20)
            inv=torch.stack((v[:,1,1],-v[:,0,1],-v[:,1,0],v[:,0,0]),1).reshape(-1,2,2)/det[:,None,None]
            positions=obs[n];indices=np.linspace(0,len(positions)-1,min(12,len(positions)),dtype=int) if positions else []
            for index in indices:
                pr,i,o=positions[index];q,depth=project(c,np.array(pr['xyz']));sigma=float(np.sqrt(np.array(c.pose.rotation)[2]@np.array(pr['covariance'])@np.array(c.pose.rotation)[2]))
                front=z<depth-2*sigma;back=z>depth+2*sigma;near=~(front|back)
                color=torch.stack((front,near,back),1).float();rgb=pred(n,color)
                x,y=np.floor(o['xy']).astype(int);diff=uv-xyz.new_tensor(o['xy']);mahal=torch.einsum('ni,nij,nj->n',diff,inv,diff)
                layers.append({'view':n,'proposal_id':pr['id'],'xy':o['xy'],'photo_camera_z':float(depth),'photo_z_sigma':sigma,
                    'composited_front_near_back':rgb[:,y,x].tolist(),'projected_3sigma_coverage':int(((mahal<=9)&(z>0)).sum()),
                    'interpretation':'current Gaussian centre-depth layers only; not surface ground truth'})
            print('depth proxies',n,flush=True)
        rows=[];ids=sorted(ids)
        for vi,n in enumerate(sorted(cams)):
            with torch.no_grad():
                base=pred(n);gt=target(n);baseerr=(base-gt).abs().mean()
                for ident in ids:
                    model._opacity[ident]=-100;abl=pred(n);model._opacity[ident]=saved_opacity[ident]
                    row={'view':n,'id':ident,'changed_pixels':int(((abl-base).abs().max(0).values>1e-6).sum()),'mae_delta_255':float((abl-gt).abs().mean()-baseerr)*255}
                    if n in fixed:
                        local={}
                        for item in summary:
                            if item['view']==n and ident in item['top_ids']:
                                mask=torch.tensor(regions[item['region']][1],device='cuda');local[item['region']]=float(((abl-gt).abs()-(base-gt).abs())[:,mask].mean())*255
                                png(abl,a.output/f'{Path(n).stem}-remove-{ident}.png')
                        row['region_delta_255']=local
                    rows.append(row)
            if vi%25==0:print('finite removals',vi+1,'of400',len(ids),'contributors',flush=True)
        assert torch.equal(model._opacity,saved_opacity) and sha256(ply)==before
        result={'experiment_id':a.experiment,'status':'completed_readonly','summary':summary,'depth_layers':layers,'removals':rows,
                'contributor_count':len(ids),'parent_ply_sha256':before,'parent_unchanged':True,'geometry_cache_sha256':sha256(a.geometry_cache),
                'seconds':time.monotonic()-start,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                'scope':'training photo evidence and frozen-model counterfactuals; no independent geometric ground truth'}
        (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
