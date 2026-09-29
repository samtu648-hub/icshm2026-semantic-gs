#!/usr/bin/env python3
"""CPU evaluation of sealed diagnostic inputs; no parameter selection."""
import argparse,json,hashlib,sys,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import label
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.diagnose_quality_expert_strata import boundary_bins
PAL=np.array([[30,30,30],[80,160,245],[245,210,65],[220,80,80],[170,100,190]],dtype=np.uint8)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cm(gt,p):return np.bincount((gt.astype(np.int64)*5+p).ravel(),minlength=25).reshape(5,5)
def met(c):
 c=np.asarray(c);tp=c.diagonal();union=c.sum(0)+c.sum(1)-tp;iou=np.divide(tp,union,out=np.full(5,np.nan),where=union>0)
 return {'miou':float(np.nanmean(iou)),'iou':[None if np.isnan(v) else float(v) for v in iou],'fn':(c.sum(1)-tp).tolist(),'gt':c.sum(1).tolist(),'background_to_stay_cable':int(c[0,2])}
def img(p):return np.array(Image.open(p))
def savepanel(ims,titles,p,box=None):
 box=box or (0,0,1320,989);w=660;h=round(w*(box[3]-box[1])/(box[2]-box[0]));canvas=Image.new('RGB',(w*3,(h+36)*2),'white');d=ImageDraw.Draw(canvas);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
 for i,(im,t) in enumerate(zip(ims,titles)):
  x=i%3*w;y=i//3*(h+36);d.text((x+5,y+5),t,fill='black',font=font);canvas.paste(Image.fromarray(im).crop(box).resize((w,h),Image.Resampling.NEAREST),(x,y+36))
 canvas.save(p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--artifact-root',type=Path,required=True);a=ap.parse_args();c=read(a.config);root=ROOT/c['output'];seal=read(root/'result.json');assert seal['status']=='completed_frozen_input_diagnostic';out=root/'evaluation';out.mkdir(exist_ok=False);rows=[];sources={};start=time.time()
 for x in seal['rows']:assert sha(root/x['arm']/x['name'])==x['prediction_sha256']
 labelhash={x['image']:x['label_png_sha256'] for x in read(ROOT/'experiments/runs/SEM-001/label_manifest.json')['records']}
 def gt(name):
  p=a.artifact_root/'r2/sem_001/labels'/name;assert sha(p)==labelhash[name];return img(p)
 if 'run' in c:
  arms=['render','photo'];sc=read(ROOT/'outputs/quality/final_transfer/SEM-363-transfer-r1/effective_config.json')['final_sources']
  for name in c['names']:
   g=gt(name);pred={k:img(root/k/name) for k in arms};base=pred['render'];photo=pred['photo'];correct=base==g
   for k in ['static','b2','deck','stay_cable']:
    for rel in (['step-3000/predictions/','step-6000/predictions/'] if k=='b2' else ['predictions/']):
     src=ROOT/sc[k];p=src/rel/name;rec=read(src/'result.json');assert sha(p)==rec['artifacts'][rel+name];m=img(p)
     correct|=((m==1)&(g==(1 if k=='deck' else 2))) if k in ['deck','stay_cable'] else (m==g)
   common=~correct;fix=(base!=g)&(photo==g);harm=(base==g)&(photo!=g);sp=boundary_bins(g);byclass=[]
   for cl in range(5):
    select=g==cl;byclass.append({'gt':int(select.sum()),'render_wrong':int(((base!=g)&select).sum()),'photo_wrong':int(((photo!=g)&select).sum()),'fixed':int((fix&select).sum()),'damaged':int((harm&select).sum()),'common':int((common&select).sum()),'common_fixed_by_photo':int((common&select&(photo==g)).sum()),'boundary_strata':[[int((select&(sp==b)).sum()),int((fix&select&(sp==b)).sum()),int((harm&select&(sp==b)).sum())] for b in range(4)]})
   r={'name':name,'confusions':{k:cm(g,v).tolist() for k,v in pred.items()},'by_class':byclass,'deck_background':{k:int(((g==1)&(v==0)).sum()) for k,v in pred.items()}}
   if name[:3] in c['rois']:
    x0,y0,x1,y1=c['rois'][name[:3]];r['roi']={'box':c['rois'][name[:3]],'confusions':{k:cm(g[y0:y1,x0:x1],v[y0:y1,x0:x1]).tolist() for k,v in pred.items()}}
   rows.append(r);diff=np.full((*g.shape,3),245,dtype=np.uint8);diff[fix]=[0,155,160];diff[harm]=[210,45,155];diff[(base!=g)&(photo!=g)]=[70,70,70]
   ims=[img(ROOT/c['real_root']/name),img(ROOT/c['render_root']/name),PAL[g],PAL[base],PAL[photo],diff];titles=[name[:3]+' Real input','RGB110 input','GT','SEM349 on RGB110','Same SEM349 on real','Teal=fixed pink=damaged grey=still wrong']
   savepanel(ims,titles,out/(name[:3]+'.png'))
   if name[:3] in c['rois']:savepanel(ims,titles,out/(name[:3]+'_crop.png'),c['rois'][name[:3]])
   print(name,r['deck_background'],'cable',byclass[2]['render_wrong'],byclass[2]['photo_wrong'],'common fixed',byclass[2]['common_fixed_by_photo'],flush=True)
 else:
  arms=['train768-infer768','train768-infer864','train864-infer768','train864-infer864']
  for fold,names in c['views'].items():
   for name in names:
    g=gt(name);pred={}
    for train in [768,864]:
     src=ROOT/c['run'+str(train)]/('full-'+fold);rec=read(src/'result.json');p=src/'predictions'/name;assert sha(p)==rec['artifacts']['predictions/'+name]
     pred[f'train{train}-infer{train}']=img(p);pred[f'train{train}-infer{1632-train}']=img(root/f'train{train}-infer{1632-train}'/name)
    sp=boundary_bins(g);r={'name':name,'fold':fold,'confusions':{k:cm(g,v).tolist() for k,v in pred.items()},'strata':{str(b):{k:cm(g[sp==b],v[sp==b]).tolist() for k,v in pred.items()} for b in range(4)}};rows.append(r)
    savepanel([img(ROOT/c['render_root']/name),PAL[g]]+[PAL[pred[k]] for k in arms],[name+' RGB110','GT']+arms,out/(name[:3]+'.png'))
 groups={'all':rows}
 if 'views' in c:groups.update({f:[r for r in rows if r['fold']==f] for f in c['views']})
 summary={group:{arm:met(np.sum([r['confusions'][arm] for r in rs],axis=0)) for arm in arms} for group,rs in groups.items()}
 (out/'result.json').write_text(json.dumps({'experiment_id':c['experiment_id'],'source_sha256':sha(root/'result.json'),'config_sha256':sha(a.config),'code_sha256':sha(__file__),'summary':summary,'rows':rows,'seconds':time.time()-start,'note':'historical development diagnostic, no fitting/tuning/independent generalization claim'},indent=2)+'\n');print(summary)
if __name__=='__main__':main()
