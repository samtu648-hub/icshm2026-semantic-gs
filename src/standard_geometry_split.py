"""Process-local RGB145/146 hook: identical budget, optional first-child XYZ placement."""
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
from src.photo_geometry import project


def associate(xyz, scales, rotations, proposals, cameras, used, fraction=.25):
    """CPU geometric association, deterministic and identical in both arms."""
    available=[i for i,p in enumerate(proposals) if p['id'] not in used]
    empty={'parent_row':[],'proposal_index':[],'normalized_distance':[],'projection_cameras':[],'reject_code':[]}
    if not len(xyz) or not available:return [],empty
    points=np.array([proposals[i]['xyz'] for i in available])
    _,query=cKDTree(points).query(xyz,k=min(8,len(available)))
    query=np.asarray(query).reshape(len(xyz),-1)
    eligible=[]; audit={k:[] for k in empty}
    for row in range(len(xyz)):
        cov=(rotations[row]*scales[row][None,:]**2) @ rotations[row].T
        for q in query[row]:
            idx=available[q];p=proposals[idx];delta=np.array(p['xyz'])-xyz[row]
            distance=float(np.linalg.norm((rotations[row].T@delta)/scales[row]))
            count=0;code=1
            if distance<=3:
                for obs in p['observations']:
                    cam=cameras[obs['camera']];r=cam.pose.rotation
                    z=r@xyz[row]+cam.pose.translation
                    if z[2]<=0:continue
                    uv=project(cam,xyz[row])[0]
                    j=np.array([[cam.matrix[0,0]/z[2],0,-cam.matrix[0,0]*z[0]/z[2]**2],
                                [0,cam.matrix[1,1]/z[2],-cam.matrix[1,1]*z[1]/z[2]**2]])@r
                    image_cov=j@cov@j.T
                    d=np.array(obs['xy'])-uv
                    try: mahal=float(d@np.linalg.solve(image_cov,d))
                    except np.linalg.LinAlgError:continue
                    count+=int(mahal<=9)
                code=0 if count>=3 else 2
            for key,value in zip(audit,(row,idx,distance,count,code)):audit[key].append(value)
            if code==0:eligible.append((distance,-p['confidence'],p['id'],row,idx))
    selected=[];parent_used=set();proposal_used=set()
    for dist,_,pid,row,idx in sorted(eligible):
        if len(selected)>=math.floor(len(xyz)*fraction):break
        if row in parent_used or pid in proposal_used:continue
        selected.append({'parent_row':row,'proposal_index':idx,'proposal_id':pid,'distance_sigma':dist})
        parent_used.add(row);proposal_used.add(pid)
    return selected,audit


def install(model_class, config, proposals, cameras, output):
    import torch
    from utils.general_utils import build_rotation
    output=Path(output); (output/'associations').mkdir(exist_ok=True)
    original_split=model_class.densify_and_split
    original_postfix=model_class.densification_postfix
    state={'iteration':0,'used':set(),'pending':None,'guided_total':0,'associated_total':0,'seconds':0.}

    def observe(iteration,camera,loss,model):
        state['iteration']=iteration
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite photometric loss')
        if iteration%100==0:
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in (model._xyz,model._features_dc,model._features_rest,model._opacity,model._scaling,model._rotation)):
                raise FloatingPointError('nonfinite parameter gradient')
            with (output/'numeric.jsonl').open('a') as f:
                f.write(json.dumps({'iteration':iteration,'camera':camera.image_name,'loss':float(loss),'points':len(model.get_xyz)})+'\n')

    def postfix(model,new_xyz,*args,**kwargs):
        if state['pending'] is not None:
            associations,parent_ids=state['pending'];rows=[]
            for a in associations:
                row=a['parent_row'];p=proposals[a['proposal_index']]
                old=new_xyz[row].clone(); new=old.new_tensor(p['xyz'])
                if config['arm']=='candidate':new_xyz[row]=new
                state['used'].add(p['id'])
                rows.append(a|{'parent_id':int(parent_ids[row]),'native_child_xyz':old.tolist(),'photo_xyz':new.tolist(),
                              'displacement':float(torch.linalg.vector_norm(new-old)),'applied':config['arm']=='candidate'})
            state['associated_total']+=len(rows)
            state['guided_total']+=len(rows) if config['arm']=='candidate' else 0
            with (output/'placements.jsonl').open('a') as f:
                f.write(json.dumps({'iteration':state['iteration'],'placements':rows})+'\n')
        return original_postfix(model,new_xyz,*args,**kwargs)

    def split(model,grads,threshold,extent,N=2):
        tick=time.monotonic()
        if N!=2:raise ValueError('binary split required')
        n=len(model.get_xyz);padded=torch.zeros(n,device=grads.device);padded[:len(grads)]=grads.flatten()
        eligible=(padded>=threshold)&(model.get_scaling.max(1).values>model.percent_dense*extent)
        ids=eligible.nonzero().flatten()
        target=math.ceil(config['initial_points']+(config['population_envelope']-config['initial_points'])*min(1.,max(0.,(state['iteration']-500)/14400)))
        k=min(len(ids),config['split_cap'],max(0,target-n))
        chosen=ids[torch.argsort(padded[ids],descending=True,stable=True)[:k]]
        # Native boolean selection orders parents by index, irrespective of score order.
        parents=chosen.sort().values
        associations,audit=associate(model.get_xyz[parents].detach().cpu().numpy(),model.get_scaling[parents].detach().cpu().numpy(),
                                    build_rotation(model._rotation[parents]).detach().cpu().numpy(),proposals,cameras,state['used'],config['guided_fraction'])
        np.savez_compressed(output/'associations'/f'{state["iteration"]:05d}.npz',parent_ids=parents.cpu().numpy(),**audit)
        state['pending']=(associations,parents)
        masked=torch.zeros_like(padded);masked[chosen]=padded[chosen]
        try:original_split(model,masked,threshold,extent,N)
        finally:state['pending']=None
        state['seconds']+=time.monotonic()-tick
        with (output/'densification.jsonl').open('a') as f:
            f.write(json.dumps({'iteration':state['iteration'],'before_split':n,'eligible':len(ids),'selected':k,'population_target':target,
                'associated':len(associations),'guided_total':state['guided_total'],'associated_total':state['associated_total'],
                'association_seconds_cumulative':state['seconds'],'arm':config['arm']})+'\n')
    model_class.densify_and_split=split;model_class.densification_postfix=postfix
    return observe,state
