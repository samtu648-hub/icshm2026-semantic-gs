#!/usr/bin/env python3
"""REPORT004: build a portable paper handoff from sealed records and media only.

No project training/rendering modules are imported. Images are scientific
compositions of saved pixels; class-ID arrays are never modified or rescored.
"""
import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from collections import Counter

os.environ.setdefault('MPLCONFIGDIR', '/tmp/icshm-report004-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ID = 'REPORT-004-PAPER-HANDOFF'
CLASSES = ['background', 'deck', 'stay_cable', 'tower', 'foundation']
PALETTE = np.array([[30,30,30],[80,160,245],[245,210,65],[220,80,80],[170,100,190]], dtype=np.uint8)
ACCENT, BLUE, RED = '#087f8c', '#3976a3', '#b84b47'


def read(p):
    return json.loads(Path(p).read_text())


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(p, data):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def clean(text):
    text = text.replace(str(ROOT), '$PROJECT_ROOT')
    # Preserve audit identity while removing personal home-directory names.
    return re.sub(r'/(?:home|Users)/[^/\s"<>`]+', '$USER_HOME', text)


def inline(text):
    s = html.escape(str(text))
    s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
    s = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', s)
    return s


def probe(p):
    return read_json_command(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=codec_name,codec_tag_string,pix_fmt,width,height,nb_frames,avg_frame_rate:format=duration,size',
        '-of','json',str(p)])


def read_json_command(cmd):
    return json.loads(subprocess.check_output(cmd, text=True))


def classify(e):
    key=e['experiment_id']; status=str(e.get('status','')).lower()
    if any(x in status for x in ['completed','done','passed','frozen','selected','rejected']): state='已完成'
    elif any(x in status for x in ['blocked','unavailable']): state='阻塞'
    elif any(x in status for x in ['failed','failure']): state='失败'
    elif any(x in status for x in ['planned','pending','preregistered','proposed','registered','not_started','ready']): state='仅计划／待执行'
    elif any(x in status for x in ['drop','cancel']): state='停止／未采用'
    else: state='见原始状态'
    if key.startswith(('REPORT','VIS','SUBMIT')): kind='材料／工程'
    elif key.startswith(('DATA','QUAL')) or any(x in key for x in ['DIAG','AUDIT','COMPARISON','PROFILE']): kind='数据／诊断／评价'
    elif any(x in key for x in ['PREFLIGHT','-PF','-P0','-EQ']): kind='预检'
    elif key.startswith(('RGB','SEM','GEO')): kind='模型实验／方法探索'
    else: kind='历史／其它'
    return state,kind


class Package:
    def __init__(self, out):
        self.out=out; out.mkdir(parents=True,exist_ok=True)
        self.assets=[]; self.gaps=[]; self.figures=[]; self.videos=[]
        self.source_map={}; self.sources={}; self.metrics=[]
        for x in ['figures','native','videos','posters','data','evidence/reports','evidence/configs','evidence/results','evidence/context']:
            (out/x).mkdir(parents=True,exist_ok=True)
        self.content=read(ROOT/'scripts/report_assets/paper_handoff_content.json')
        self.author_guide=read(ROOT/'scripts/report_assets/paper_handoff_author_guide.json')
        # Never include this handoff in the historical experiment denominator.
        self.registry=read(ROOT/'scripts/report_assets/paper_handoff_registry_20260923.json')
        if len(self.registry)!=351:
            raise ValueError('Historical registry changed; review cutoff before rebuilding')
        known={e['experiment_id'] for e in self.registry}
        self.supplemental=[dict(experiment_id=p.parent.name,report_path=str(p.relative_to(ROOT)),
            status='historical_stage_report_not_separately_registered',config=str(p.parent.relative_to(ROOT)/'config.yaml'))
            for p in [ROOT/'experiments/runs/SEM-002/report.md'] if p.is_file() and p.parent.name not in known]

    def resolve(self, short):
        exact=[e for e in self.registry+self.supplemental if e['experiment_id']==short]
        candidates=exact or sorted([e for e in self.registry if e['experiment_id'].startswith(short+'-')],key=lambda e:len(e['experiment_id']))
        if not candidates: raise ValueError('Unknown evidence reference '+short)
        return candidates[0]

    def relative(self,p):
        # Keep a repository's logical symlink path so a source can be verified
        # without embedding its private artifact-store destination.
        p=Path(p).absolute()
        try:return str(p.relative_to(ROOT))
        except ValueError:return clean(str(p))

    def asset(self,target,sources,transform,**extra):
        rows=[]
        for p in sources:
            p=Path(p)
            rows.append({'path':self.relative(p),'sha256':digest(p)})
            self.sources[self.relative(p)]=digest(p)
        self.assets.append(dict(path=str(target.relative_to(self.out)),sha256=digest(target),bytes=target.stat().st_size,
                                sources=rows,transform=transform,**extra))

    def copy(self,source,relative):
        source=Path(source); dest=self.out/relative; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,dest); self.asset(dest,[source],'byte-exact copy')
        return relative

    def result(self,relative):
        source=ROOT/relative; data=read(source)
        dest=self.out/'evidence/results'/relative.replace('/','__')
        dest.write_text(clean(json.dumps(data,ensure_ascii=False,indent=2))+'\n')
        self.asset(dest,[source],'JSON text snapshot; personal paths replaced with portable placeholders')
        self.source_map[relative]=str(dest.relative_to(self.out))
        return data

    def native(self,source,label,semantic=False,view=None):
        source=Path(source); dest=self.out/'native'/f'{label}.png'
        with Image.open(source) as im:
            arr=np.asarray(im)
            if semantic:
                if arr.ndim==2:
                    if int(arr.max())>4:raise ValueError(f'Not class IDs: {source}')
                    arr=PALETTE[arr]
                    transform='class-ID 0..4 mapped to shared presentation palette; geometry unchanged'
                else:
                    # Historical presentation images already use the shared palette.
                    valid=np.zeros(arr.shape[:2],bool)
                    for c in PALETTE:valid |= np.all(arr[:,:,:3]==c,axis=2)
                    if not valid.all():raise ValueError(f'Unknown semantic colors: {source}')
                    transform='lossless existing shared-palette semantic image'
            else:
                arr=np.asarray(im.convert('RGB')); transform='saved RGB pixels, lossless PNG container'
            Image.fromarray(arr).save(dest)
        self.asset(dest,[source],transform,view=view,semantic=semantic,dimensions=list(Image.open(dest).size))
        return str(dest.relative_to(self.out))

    def metric(self,label,value,unit,role,source,field,experiments):
        self.metrics.append(dict(label=label,value=value,unit=unit,role=role,source=source,field=field,experiments=experiments))

    def fig(self,fid,title,caption,en,role,refs,files,native=None,selection=None):
        self.figures.append(dict(id=fid,title=title,caption_zh=caption,caption_en=en,role=role,refs=refs,files=files,
                                 native=native or [],selection=selection or '预定义主题及既有固定样例；未进行新质量筛选',use='正文' if fid not in ['F04','F10'] else '正文讨论或补充材料'))

    def plot(self,name,fig,sources):
        outputs=[]
        for ext in ['svg','png']:
            p=self.out/'figures'/f'{name}.{ext}'
            fig.savefig(p,dpi=230,bbox_inches='tight',facecolor='white')
            self.asset(p,[ROOT/s for s in sources],'scientific plot from saved numbers; no rescoring')
            outputs.append(str(p.relative_to(self.out)))
        plt.close(fig);return outputs

    def panel(self,name,tiles,columns=3,crop=None,legend=True):
        w=660; title_h=42; h=495 if crop is None else 360
        rows=math.ceil(len(tiles)/columns)
        canvas=Image.new('RGB',(columns*w,rows*(h+title_h)+(56 if legend else 0)),'white'); draw=ImageDraw.Draw(canvas)
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
        for i,(label,rel,semantic) in enumerate(tiles):
            x=(i%columns)*w;y=(i//columns)*(h+title_h)
            draw.text((x+10,y+10),label,fill='#172d39',font=font)
            im=Image.open(self.out/rel).convert('RGB')
            if crop:im=im.crop(crop)
            ratio=min(w/im.width,h/im.height);size=(round(im.width*ratio),round(im.height*ratio))
            im=im.resize(size,Image.Resampling.NEAREST if semantic else Image.Resampling.LANCZOS)
            canvas.paste(im,(x+(w-size[0])//2,y+title_h+(h-size[1])//2))
        yy=rows*(h+title_h)+17
        if legend:
            for j,(cls,color) in enumerate(zip(CLASSES,PALETTE)):
                xx=15+j*(columns*w//5);draw.rectangle((xx,yy,xx+18,yy+18),fill=tuple(color));draw.text((xx+25,yy-2),cls,fill='#172d39',font=font)
        p=self.out/'figures'/f'{name}.png';canvas.save(p)
        self.asset(p,[self.out/t[1] for t in tiles],f'paired scientific panel; shared grid; crop={crop}; semantic nearest-neighbor, RGB Lanczos for panel only',crop_xyxy=crop)
        return str(p.relative_to(self.out))

    def evidence(self):
        self.records=[]
        entries=[dict(e,config=e.get('config',e.get('config_path',''))) for e in self.registry+self.supplemental]
        for e in entries:
            eid=e['experiment_id'];state,kind=classify(e)
            record=dict(experiment_id=eid,status=e.get('status','not_available'),state=state,kind=kind,
                parent_experiment=e.get('parent_experiment','not_available'),git_commit=e.get('git_commit','not_available'),
                seed=e.get('seed','not_available'),dataset_manifest=e.get('dataset_manifest','not_available'),
                split_id=e.get('split_id','not_available'),conclusion=e.get('conclusion',e.get('summary','')),
                artifacts=e.get('artifacts',{}),metrics=e.get('metrics',{}),files={})
            for key,folder in [('report_path','reports'),('config','configs')]:
                src=e.get(key,'')
                if isinstance(src,str) and src and (ROOT/src).is_file():
                    dest=f'evidence/{folder}/{eid}'+('.md' if folder=='reports' else '.yaml')
                    self.source_map[src]=dest;record['files'][key]=dest
                    record['files'][key+'_source_sha256']=digest(ROOT/src)
                else:
                    self.gaps.append(dict(experiment_id=eid,kind=key,source=src,reason='原登记未提供或本地未找到；不推断为实验失败'))
            self.records.append(record)
        for e,record in zip(entries,self.records):
            for key in ['report_path','config']:
                if key not in record['files']:continue
                src=ROOT/e[key];dest=self.out/record['files'][key]
                text=clean(src.read_text())
                if key=='report_path':
                    def link(m):
                        label,target=m.groups(); target=target.strip('<>')
                        if target.startswith(('https://','http://','#','mailto:')):return m.group(0)
                        base,_,anchor=target.partition('#')
                        resolved=(src.parent/base).resolve()
                        try:rel=str(resolved.relative_to(ROOT))
                        except ValueError:rel=''
                        if rel in self.source_map:
                            return '['+label+']('+os.path.relpath(self.out/self.source_map[rel],dest.parent)+('' if not anchor else '#'+anchor)+')'
                        return f'{label}（原引用 `{clean(target)}`；未随包，见来源记录）'
                    text=re.sub(r'(?<!!)\[([^\]]+)\]\(([^)]+)\)',link,text)
                    text='> 历史报告快照；“当前”等措辞对应原记录日期。个人路径已替换，不可搬迁的链接已显式标注。\n\n'+text
                dest.write_text(text)
                self.asset(dest,[src],'portable text snapshot; personal paths sanitized; unavailable local links annotated')
            detail=self.out/'evidence/reports'/f"{e['experiment_id']}.html"
            blocks=['<h2>登记与来源</h2><pre>'+html.escape(clean(json.dumps(record,ensure_ascii=False,indent=2)))+'</pre>']
            for key,label in [('report_path','历史报告'),('config','配置快照')]:
                if key in record['files']:
                    p=self.out/record['files'][key]
                    blocks.append(f'<h2>{label}</h2><p><a href="{html.escape(os.path.relpath(p,detail.parent))}">打开可编辑原文</a></p><pre>{html.escape(p.read_text())}</pre>')
            detail.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(e['experiment_id'])+'</title><style>body{font:16px/1.7 system-ui;margin:24px auto;max-width:1050px;padding:0 20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 monospace}a{color:#006c79}</style><a href="../../index.html#experiments">返回实验索引</a><h1>'+html.escape(e['experiment_id'])+'</h1><p>历史记录，当前角色以交接首页及 ADR-175 为准。</p>'+''.join(blocks)+'</html>')
            record['detail']=str(detail.relative_to(self.out))
        # Portable context: exact scientific meaning, home paths anonymized.
        for src in ['DECISIONS.md','RISKS.md','METRICS.md','docs/experiments/research_summary_20260908.md','docs/paper/development_evidence_outline.md','third_party/README.md','docs/submission/camera_coordinates.md','docs/submission/interface.md']:
            dest=self.out/'evidence/context'/src.replace('/','__')
            dest=dest.with_suffix('.txt');dest.write_text(clean((ROOT/src).read_text()))
            self.asset(dest,[ROOT/src],'plain-text historical context; paths sanitized; references retained as non-clickable provenance')
        manifest=ROOT/'data/manifests/contest_dataset_v1.json'
        self.copy(manifest,'evidence/context/contest_dataset_v1.json')
        for src in ['experiments/splits/sem_001_scaffold_static_v1.json','experiments/splits/spatial_confirmation_v1.json','experiments/splits/spatial_confirmation_v2.json']:
            if (ROOT/src).exists():
                dest=self.out/'evidence/context'/Path(src).name;dest.write_text(clean((ROOT/src).read_text()));self.asset(dest,[ROOT/src],'split snapshot with portable paths')
        save(self.out/'data/supplemental_reports.json',self.records[len(self.registry):])
        self.records=self.records[:len(self.registry)]
        save(self.out/'data/experiment_index.json',json.loads(clean(json.dumps(self.records,ensure_ascii=False))))
        save(self.out/'data/registry_snapshot.json',json.loads(clean(json.dumps(self.registry,ensure_ascii=False))))

    def figures_build(self):
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
        profile_src='outputs/quality/QUAL-009/r1/result.json';profile=self.result(profile_src)
        fig,axs=plt.subplots(1,2,figsize=(12,4.2),gridspec_kw={'width_ratios':[1.3,1]})
        vals=[100*profile['class_summary'][c]['pixel_share_all_labeled'] for c in CLASSES]
        bars=axs[0].barh(CLASSES,vals,color=PALETTE/255)
        axs[0].invert_yaxis();axs[0].set(xlabel='Share of labeled pixels (%)',title='A. Released 300-view label distribution',xlim=(0,103))
        axs[0].bar_label(bars,fmt='%.3f%%',padding=4)
        axs[1].axis('off');axs[1].set_title('B. Keep evaluation roles separate')
        lines=['400 released RGB images / 300 labeled views','Historical semantics: 219 train / 74 dev / 7 guard','Spatial v1/v2: train-only initialization','All300 / all400: training fit','Unlabeled100 / trajectory144: no GT','Spatial folds have prior development exposure']
        for i,s in enumerate(lines):axs[1].text(0,.94-i*.16,s,fontsize=10,va='top',wrap=True)
        fig.tight_layout();files=self.plot('F01_dataset_and_roles',fig,[profile_src])
        self.fig('F01','数据与评价角色','300张标签的像素组成及不同证据角色。stay_cable不是最少的前景类；小构件长尾、尺度和标注形态需要单独讨论。','Label composition of 300 released views and distinct evaluation roles. Pixel frequency alone does not explain thin-structure difficulty.','数据画像，不是模型成绩',['QUAL-009','SEM-001','QUAL-012'],files)

        fig,axs=plt.subplots(3,1,figsize=(12,6.8));fig.subplots_adjust(hspace=.6)
        schemes=[('Historical system: RGB030 + SEM161',['Official camera','Frozen Scaffold RGB\n+ static 3D semantics','2D rendered-domain\nclass specialists','Protected fusion\n5-class output']),('Existing full-data demonstration: RGB170 + SEM437',['Official camera','Frozen ImprovedGS\nRGB170 50k','Mask2Former\nSEM437 12k','RGB + semantics\nall300 fit / no-GT video']),('Future unknown-view recipe: not a completed semantic system',['Official camera','RGB166\n1M-budget recipe','Matching all300\nsemantic fit: pending','Joint review\nstill pending'])]
        for ax,(title,labels) in zip(axs,schemes):
            ax.axis('off');ax.set_title(title,loc='left',fontweight='bold',pad=10)
            for i,label in enumerate(labels):
                x=i*.255;box=FancyBboxPatch((x,.12),.215,.67,boxstyle='round,pad=0.007',facecolor='#eef5f6',edgecolor='#96afb7',transform=ax.transAxes);ax.add_patch(box)
                ax.text(x+.1075,.45,label,ha='center',va='center',fontsize=10,transform=ax.transAxes)
                if i<3:ax.annotate('',xy=(x+.249,.45),xytext=(x+.22,.45),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':ACCENT})
        files=self.plot('F02_system_roles',fig,['DECISIONS.md','experiments/runs/SEM-437-RGB170-MASK2FORMER-CONTEXT50-LONG12K-ALL300/report.md'])
        self.fig('F02','三条系统角色，避免画成一条混合链','历史专家融合、已有全量展示和未来待完成系统分开绘制。后期SEM437推理不包含早期SEM161专家链。','Separate the historical specialist-fusion system, existing full-data demonstration, and pending RGB166 semantic system.','方法与完成状态',['SEM-161','SEM-437','QUAL-026'],files)

        history_src='outputs/research_report_20260908/data/content.json';hist=self.result(history_src)['history']
        fig,ax=plt.subplots(figsize=(11,4));vs=[100*x['miou'] for x in hist];bars=ax.bar([x['short'] for x in hist],vs,color=[BLUE]*5+[ACCENT]);ax.bar_label(bars,fmt='%.3f',padding=3);ax.set(ylim=(0,100),ylabel='Five-class pooled mIoU (%)',xlabel='Historical system version (different recipes / seeds)',title='Same historical 74 development views — system evolution, not one-factor ablation');fig.tight_layout()
        files=self.plot('F03_historical_progress',fig,[history_src]);natives=[];tiles=[]
        old=ROOT/'outputs/research_report_20260908/assets/images'
        for label,k,sem in [('Photo','reference',False),('RGB030','rgb',False),('GT','gt',True),('SEM001','sem001',True),('SEM111','sem111',True),('SEM161','sem161',True)]:
            p=self.native(old/'054'/f'{k}.png',f'F03_054_{k}',sem,'054');natives.append(p);tiles.append((label,p,sem))
        files.append(self.panel('F03_054_semantic_evolution',tiles))
        self.fig('F03','第一次大幅进步：三维基线到类别专家链','历史74图五类mIoU：SEM001 61.610304%→SEM161 83.387017%（+21.776713pp）。054为既有固定样例；同视角图展示背景误报与构件变化。','On the same reused 74 development views, SEM001 to SEM161 increases five-class mIoU from 61.610304% to 83.387017%. View 054 is a pre-existing fixed example.','历史74图开发；非盲测',['SEM-001','SEM-111','SEM-141','SEM-151','SEM-161','SEM-329'],files,natives)
        self.metric('SEM001→SEM161 五类mIoU增量',(hist[4]['miou']-hist[0]['miou'])*100,'pp','historical74',history_src,'100 * (history[4].miou - history[0].miou)',['SEM-001','SEM-161'])

        natives=[];files=[]
        for view,box in [('160',[678,182,1015,648]),('183',[594,487,733,617])]:
            tiles=[]
            for label,k,sem in [('Photo','reference',False),('GT','gt',True),('SEM111','sem111',True),('SEM141','sem141',True),('SEM151','sem151',True),('SEM161','sem161',True)]:
                p=self.native(old/view/f'{k}.png',f'F04_{view}_{k}',sem,view);natives.append(p);tiles.append((label,p,sem))
            files.append(self.panel(f'F04_{view}_class_specialists',tiles,crop=box))
        self.fig('F04','类别专家的局部作用与代价','160桥面/索区与183基础为既有固定样例；提供相同坐标裁剪及原尺寸单图。类别保护和单向修改有收益，也保留漏检与边界问题。','Matched crops from fixed views 160 and 183 illustrate class-specific changes. Protected updates improve selected errors while preserving some omissions and boundary failures.','历史74图定性例；同视角、同裁剪',['SEM-111','SEM-141','SEM-151','SEM-161'],files,natives)

        m2src='outputs/quality/final_comparison/SEM-350-evaluation-r1/result.json';m2=self.result(m2src)['evaluation']['metrics']
        fig,ax=plt.subplots(figsize=(10,4));keys=['old_sem161','new_b2','new_raw','new_fused'];labels=['RGB030 + SEM161','RGB110 + B2 fusion','RGB110 + M2F raw','RGB110 + M2F fused'];vv=[m2[x]['mIoU_with_background']*100 for x in keys];bars=ax.barh(labels,vv,color=[BLUE,BLUE,ACCENT,BLUE]);ax.invert_yaxis();ax.bar_label(bars,fmt='%.3f%%',padding=3);ax.set(xlim=(0,102),xlabel='Five-class pooled mIoU (%)',title='SEM350: historical74 — B2 fusion is not raw B2');fig.tight_layout()
        files=self.plot('F05_mask2former_comparison',fig,[m2src]);natives=[]
        cfg=read(ROOT/'experiments/runs/SEM-349-MASK2FORMER-FINAL/config.yaml')
        for view in ['159','057']:
            oldroot=old/view
            # 057 is intentionally included as a documented regression.
            gt_source=(oldroot/'gt.png') if oldroot.exists() else ROOT/'outputs/quality/all_data/SEM-370-r1/labels'/f'{view}.png'
            photo=(oldroot/'reference.png') if oldroot.exists() else ROOT/'outputs/quality/RGB-150/scene/images'/f'{view}.png'
            tiles=[]
            for label,source,sem in [('Photo',photo,False),('RGB110',ROOT/cfg['prediction_export']/'images'/f'{view}.png',False),('GT',gt_source,True),('M2F raw',ROOT/'outputs/quality/mask2former/SEM-349-r1/fit/predictions'/f'{view}.png',True)]:
                p=self.native(source,f'F05_{view}_{label.replace(" ","_")}',sem,view);natives.append(p);tiles.append((label,p,sem))
            files.append(self.panel(f'F05_{view}_m2f',tiles,columns=2))
        self.fig('F05','强二维基线：总体提高，057仍退化','同74固定评价中，Mask2Former raw为86.513086%，旧SEM161为83.387017%；68图提高、6图下降。B2一栏是融合版80.917724%，不是raw 75.324650%。RGB底座和训练预算不完全相同。','Fixed historical74 comparison: Mask2Former raw reaches 86.513086% versus 83.387017% for SEM161, with 68 improved and 6 regressed views. B2 denotes the fused system. Different RGB backbones and training budgets prevent a pure capacity attribution.','历史74图开发；完整方法包比较',['SEM-349','SEM-350'],files,natives,selection='159为既有固定困难图；057为SEM350登记的最大raw退化图（评分后诊断）')

        bsrc='outputs/quality/backbone_comparison/browser-r1/data.json';b=self.result(bsrc)['summaries']
        fig,axs=plt.subplots(1,3,figsize=(12,4));names=['Scaffold UID0','Standard','Mip']
        for ax,key,title in zip(axs,['PSNR','SSIM','LPIPS'],['PSNR (dB), higher better','SSIM, higher better','LPIPS, lower better']):
            v=[s['rgb'][key] for s in b];bars=ax.bar(names,v,color=[BLUE,ACCENT,BLUE]);ax.bar_label(bars,fmt='%.3f',padding=3);ax.set_title(title);ax.tick_params(axis='x',labelrotation=18);ax.set_xlabel('All400 reconstruction fit');ax.set_ylim(0,max(v)*1.22)
        fig.suptitle('Original full-data backbone comparison: Scaffold uses deployment UID0');fig.tight_layout();files=self.plot('F06_rgb_backbones',fig,[bsrc])
        self.fig('F06','RGB骨干比较必须交代外观条件','原始全量三骨干部署口径比较。Scaffold采用固定UID0，训练UID原生分数另有记录；随后RGB136无外观控制不能省略。相同30k也不是相同计算成本。','Full-data RGB comparison under the recorded deployment convention. Scaffold uses fixed UID0; its native training-UID score and later no-appearance control must be distinguished. Equal iterations do not imply equal compute.','全400训练拟合；固定外观约定',['QUAL-007','RGB-130','RGB-136'],files)

        spatial=[];srcs=[]
        for code in ['011','013']:
            src=f'outputs/quality/QUAL-{code}/r1/result.json';srcs.append(src);spatial.append(self.result(src))
        fig,axs=plt.subplots(1,3,figsize=(12,4.3));mk=['foreground_miou','stay_cable_iou','boundary_f1'];mt=['Foreground mIoU','stay_cable IoU','2px boundary F1']
        for ax,k,title in zip(axs,mk,mt):
            means=[]
            for d in spatial:
                mm=d['two_seed_means'];means.append(100*(mm['absgs']['holdout'][k]-mm['standard']['holdout'][k]))
            bars=ax.bar(['v1','v2'],means,color=[BLUE,ACCENT]);ax.bar_label(bars,fmt='%+.3f',padding=3);ax.set(title=title,ylabel='AbsGS minus Standard (pp)',xlabel='Spatial fold / mean of 2 semantic seeds',ylim=(0,max(means)*1.24))
        fig.suptitle('Repeated overall direction; strongly region-dependent magnitude');fig.tight_layout();files=self.plot('F07_spatial_confirmation',fig,srcs)
        self.fig('F07','跨空间fold与语义seed：方向重复，幅度不均','每个fold展示两个语义seed的汇总分数均值差，而非集成预测分数。v1/v2均为已暴露开发数据，RGB只有seed0；部分区域/类别退化，v2训练和渲染成本增加。','Differences of pooled-score means across two semantic seeds per fold, not prediction ensembles. Both folds are historically exposed development evidence; RGB uses one seed. Regional regressions and compute premiums remain.','空间开发确认；非盲测',['QUAL-010','QUAL-011','QUAL-012','QUAL-013'],files)

        rsrc='outputs/quality/RGB-166/full-r1/evaluation-r1/result.json';rgb=self.result(rsrc)
        fig,axs=plt.subplots(1,2,figsize=(12,4.2));steps=[30000,35000,40000,45000,50000];axs[0].plot(np.array(steps)/1000,[rgb['all400'][str(s)]['PSNR'] for s in steps],'-o',color=ACCENT);axs[0].set(xlabel='RGB166 iteration (thousands)',ylabel='All400 PSNR (dB)',title='A. One RGB166 training trajectory')
        for code in ['436','437']:
            e=self.resolve('SEM-'+code);m=e['metrics'];ss=[6000,8000,10000,12000];vv=[100*m[f'training_miou_{x}'] for x in ss];axs[1].plot(np.array(ss)/1000,vv,'-o',label='SEM'+code)
        axs[1].set(xlabel='Semantic iteration (thousands)',ylabel='All300 five-class mIoU (%)',title='B. RGB170 fixed; full300 training fit');axs[1].legend();fig.tight_layout();files=self.plot('F08_training_lengths',fig,[rsrc,'experiments/registry.yaml'])
        self.fig('F08','训练长度：保留完整曲线','RGB166使用同一次50k训练中的五端点；语义保留6k/8k/10k/12k波动。SEM437同时改变上下文和目标尺度；空间v1/v2长训最优端点不同，不能将12k写成泛化最优。','Saved endpoints from one RGB166 trajectory and the complete SEM436/437 curves. These are training-fit scores. Context mixing also changes object scale; spatial folds do not share one best semantic endpoint.','全400/全300训练拟合',['RGB-166','SEM-434','SEM-435','SEM-436','SEM-437'],files)

        csrc='outputs/quality/QUAL-026/r1/result.json';capacity=self.result(csrc);d=capacity['comparison_15M_minus_1M'];fig,ax=plt.subplots(figsize=(9,4.2));vv=[d[x]['PSNR'] for x in ['train','holdout','buffer']];bars=ax.bar(['Train353','Holdout30','Buffer17'],vv,color=[ACCENT,RED,RED]);ax.bar_label(bars,fmt='%+.3f',padding=4);ax.axhline(0,color='#6b7d86',lw=.8);ax.set(ylabel='PSNR delta: 1.5M minus 1.0M (dB)',title='QUAL026: more capacity improves seen fit but hurts held-out views',ylim=(-1.05,.23));fig.tight_layout();files=self.plot('F09_capacity_tradeoff',fig,[csrc]);native=[]
        for view in ['159','260']:
            src=ROOT/'outputs/quality/QUAL-026/r1/review/full'/f'{view}.jpg'
            p=self.copy(src,f'native/F09_{view}_saved_comparison.jpg');native.append(p);files.append(p)
            tiles=[]
            for label,source in [('Photo',ROOT/'outputs/quality/all_data/RGB-120-r2/scene/images'/f'{view}.png'),
                ('RGB175 / spatial 1.0M',ROOT/'outputs/quality/QUAL-026/r1/images/RGB175_1M'/f'{view}.png'),
                ('RGB176 / spatial 1.5M',ROOT/'outputs/quality/QUAL-026/r1/images/RGB176_15M'/f'{view}.png')]:
                p=self.native(source,f'F09_{view}_{label.split()[0]}',False,view);native.append(p);tiles.append((label,p,False))
            files.append(self.panel(f'F09_{view}_matched_capacity',tiles,legend=False))
        self.fig('F09','必须保留的反例：容量收益没有迁移','严格配对RGB175/176只改变100万→150万预算；holdout排除于训练与初始化。150万在holdout30下降0.867008dB，26/30图退化。提供原尺寸照片/两臂单图；另附原审阅对照，其中RGB166/RGB170仅提供全量语境，不参与留出因果比较。','Matched RGB175/176 differ only in the 1.0M versus 1.5M budget. Training and initialization exclude holdout/buffer. The larger budget loses 0.867008 dB on holdout30. Native photo and paired outputs are included; RGB166/170 in the archived panels are full-data context, not holdout controls.','历史暴露空间确认；严格配对容量比较',['QUAL-026','RGB-175','RGB-176'],files,native,selection='159为既有困难视角；260为报告登记的最大holdout退化视角')

        files=[];native=[]
        for view in ['350','357']:
            tiles=[]
            for label,src,sem in [('Photo',ROOT/'outputs/quality/RGB-150/scene/images'/f'{view}.png',False),('RGB170',ROOT/'outputs/quality/RGB-170/full-r1/evaluation-r1/images/50000'/f'{view}.png',False),('SEM437 (no GT)',ROOT/'outputs/quality/SEM-437/r2/inference/predictions/12000'/f'{view}.png',True)]:
                p=self.native(src,f'F10_{view}_{label.split()[0]}',sem,view);native.append(p);tiles.append((label,p,sem))
            files.append(self.panel(f'F10_{view}_unresolved',tiles))
        # A complete current same-view four-way set complements the failure examples.
        tiles=[]
        for label,src,sem in [('Photo',ROOT/'outputs/quality/RGB-150/scene/images/159.png',False),('RGB170',ROOT/'outputs/quality/RGB-170/full-r1/evaluation-r1/images/50000/159.png',False),('GT',ROOT/'outputs/quality/all_data/SEM-370-r1/labels/159.png',True),('SEM437',ROOT/'outputs/quality/SEM-437/r2/inference/predictions/12000/159.png',True)]:
            p=self.native(src,f'F10_159_{label}',sem,'159');native.append(p);tiles.append((label,p,sem))
        files.append(self.panel('F10_159_current_fit',tiles,columns=2))
        self.fig('F10','当前表现与未解决视角并列','159展示全300训练内照片/RGB/GT/预测；350和357为无标签视角，357仍无deck。无GT图不能计算准确率；不对原结果做修补、美化或生成式补画。','View 159 is an in-training photo/RGB/GT/prediction example. Unlabeled views 350 and 357 are qualitative only; deck remains absent in 357. No generative retouching or model rerun is used.','159训练拟合；350/357无GT定性',['SEM-437','SEM-439','SEM-440'],files,native)

    def videos_build(self):
        specs=[
            ('V01a','历史三维语义 SEM001','outputs/project_progress_20260918/assets/videos/sem001.mp4',['SEM-001'],'shared'),
            ('V01b','历史类别专家 SEM161','outputs/project_progress_20260918/assets/videos/sem161.mp4',['SEM-161'],'shared'),
            ('V01c','历史 Mask2Former SEM349','outputs/project_progress_20260918/assets/videos/m2f.mp4',['SEM-349'],'shared'),
            ('V02','原始三骨干同轨迹','outputs/quality/backbone_comparison/browser-r1/three-backbones-trajectory.mp4',['QUAL-007'],'rgb'),
            ('V04','已有展示 RGB170 + SEM437 · 144帧','outputs/quality/SEM-437/r2/review/videos/trajectory144_rgb_sem437_12k_side_by_side.mp4',['SEM-437'],'late'),
            ('V05','已有展示 · 100张无标注视角','outputs/quality/SEM-437/r2/review/videos/unlabeled301_400_rgb_sem437_12k_side_by_side.mp4',['SEM-437'],'late'),
            ('V06','容量确认 · RGB170 / RGB175 / RGB176','outputs/quality/QUAL-026/r1/trajectory/rgb170-vs-rgb175-vs-rgb176-trajectory.mp4',['QUAL-026'],'rgb')]
        for vid,title,src,refs,palette in specs:
            rel=self.copy(ROOT/src,f'videos/{vid}.mp4');self.add_video(vid,title,rel,[src],refs,palette)
        # Match by saved filename, not independently sampled or regenerated poses.
        a=ROOT/'outputs/quality/RGB-166/full-r1/trajectory-r1/30000';b=a.parent/'50000'
        names=sorted(p.name for p in a.glob('frame*.png'))
        assert len(names)==144 and names==sorted(p.name for p in b.glob('frame*.png'))
        traj=self.result('outputs/quality/RGB-166/full-r1/trajectory-r1/result.json')
        assert traj['view_count']==144
        video=self.out/'videos/V03.mp4'
        sources=[a/n for n in names]+[b/n for n in names]
        fingerprint=hashlib.sha256(''.join(digest(p) for p in sources).encode()).hexdigest()
        stamp=self.out/'data/V03_inputs.json'
        if not (video.exists() and stamp.exists() and read(stamp).get('fingerprint')==fingerprint):
            command=['ffmpeg','-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1320x536','-r','12','-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
            process=subprocess.Popen(command,stdin=subprocess.PIPE)
            font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
            try:
                for name in names:
                    canvas=Image.new('RGB',(1320,536),'white');draw=ImageDraw.Draw(canvas)
                    for i,(base,label) in enumerate([(a,'RGB166 / same-run 30k'),(b,'RGB166 / same-run 50k')]):
                        im=Image.open(base/name).convert('RGB').resize((660,495),Image.Resampling.LANCZOS)
                        canvas.paste(im,(i*660,41));draw.text((i*660+12,10),f'{label} | {name}',fill='#172d39',font=font)
                    process.stdin.write(canvas.tobytes())
            finally:process.stdin.close()
            if process.wait()!=0:raise RuntimeError('Saved-frame video encoding failed')
            save(stamp,dict(fingerprint=fingerprint,frame_names=names,fps=12,trajectory=traj['trajectory_sha256']))
        self.asset(video,sources,'paired saved frames by exact filename; 144 frames at 12fps; resize and labeled composition only',fps=12)
        self.add_video('V03','RGB166 同轨迹30k与50k',str(video.relative_to(self.out)),['outputs/quality/RGB-166/full-r1/trajectory-r1/result.json'],['RGB-166'],'rgb')
        self.videos.sort(key=lambda v:v['id'])

    def add_video(self,vid,title,rel,sources,refs,palette):
        p=self.out/rel;meta=probe(p);poster=self.out/'posters'/f'{vid}.jpg'
        subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-ss','1','-i',str(p),'-frames:v','1','-q:v','2',str(poster)],check=True)
        self.asset(poster,[p],'poster frame at 1 second, JPEG q2')
        legends={'shared':'background 深灰 / deck 蓝 / stay_cable 黄 / tower 红 / foundation 紫',
                 'late':'background 黑 / deck 橙红 / stay_cable 黄 / tower 蓝 / foundation 橄榄绿','rgb':'纯RGB对比，无语义配色'}
        notes={'V02':'Scaffold为原三骨干固定UID0部署口径；不是后期无外观Scaffold或ImprovedGS。',
               'V06':'RGB170为全400语境；RGB175/176为匹配空间模型。视频没有GT，不承担holdout量化结论。',
               'V03':'144个保存帧严格按相同文件名成对，12fps，未重新渲染。',
               'V05':'301–400按视角编号播放，是图像序列，不是连续时间采集或独立测试集。'}
        stream=meta['streams'][0]
        english_titles={'V01a':'Historical static 3D semantics (SEM001)','V01b':'Historical specialist fusion (SEM161)',
            'V01c':'Historical Mask2Former (SEM349)','V02':'Three RGB backbones on the saved trajectory',
            'V03':'RGB166 same-run 30k and 50k endpoints','V04':'RGB170 and SEM437 on the 144-pose trajectory',
            'V05':'RGB170 and SEM437 on 100 unlabeled released views','V06':'Capacity comparison: RGB170 context and matched RGB175/176'}
        self.videos.append(dict(id=vid,title=title,path=rel,poster=str(poster.relative_to(self.out)),refs=refs,sources=sources,
            role='无GT定性；100张图已用于全量RGB训练' if vid=='V05' else '无GT固定轨迹，仅定性',
            legend=legends[palette],palette=palette,probe=meta,
            caption_zh=f'{title}。保留真实伪影与原视频帧序。'+notes.get(vid,'历史144相机视频单独播放；各阶段系统不同，不能作为单变量消融。'),
            caption_en=english_titles[vid]+': saved outputs only, no ground truth; qualitative evidence. '+('Paired by exact saved frame name at 12 fps.' if vid=='V03' else 'Original frame order and appearance are retained.'),
            frame_count=int(stream.get('nb_frames',0)),fps=stream['avg_frame_rate'],duration=float(meta['format']['duration'])))

    def refs_html(self,refs):
        return ' · '.join('<a href="evidence/reports/'+html.escape(self.resolve(r)['experiment_id'])+'.html">'+html.escape(r)+'</a>' for r in refs)

    def result_tables(self):
        tables=[]
        def table(title,role,columns,rows,source,note):
            tables.append(dict(title=title,role=role,columns=columns,rows=rows,source=source,note=note))
        src='outputs/research_report_20260908/data/content.json'
        hist=read(ROOT/src)['history']
        table('历史系统演进','historical74',['实验','五类mIoU (%)','seed'],
              [[x['short'],100*x['miou'],str(x['seed'])] for x in hist],src,'同74开发视角；不同配方及seed，不是单变量消融。')
        src='outputs/quality/final_comparison/SEM-350-evaluation-r1/result.json'
        mm=read(ROOT/src)['evaluation']['metrics']
        table('SEM350固定完整系统比较','historical74',['系统','五类mIoU (%)','前景mIoU (%)']+[x+' IoU (%)' for x in CLASSES],
            [[key,100*m['mIoU_with_background'],100*m['mIoU_without_background']]+[100*m['per_class'][c]['IoU'] for c in CLASSES] for key,m in mm.items()],src,
            'new_b2是融合版；raw B2 75.324650%另见REPORT002重算。RGB030/RGB110、B2 3k/M2F 6k不能作为纯容量消融。')
        metric_keys=['five_class_miou','foreground_miou','stay_cable_iou','boundary_f1']
        for num in ['011','013']:
            src=f'outputs/quality/QUAL-{num}/r1/result.json';d=read(ROOT/src);rows=[]
            for seed,pair in d['paired'].items():
                for arm in ['absgs','standard']:
                    for region in ['holdout','region-1','region-2','region-3']:
                        rows.append([seed,arm,region]+[100*pair[arm][region][k] for k in metric_keys])
            table('空间'+('v1' if num=='011' else 'v2')+'：分seed、分区域','spatial_development',
                  ['语义seed','骨干','区域']+[x+' (%)' for x in metric_keys],rows,src,
                  '每行来自该seed对应区域的汇总分数。RGB seed仅0；历史暴露开发fold，不作显著性声明。')
        src='outputs/quality/backbone_comparison/browser-r1/data.json';rows=[]
        for s in read(ROOT/src)['summaries']:
            rows.append([s['arm']]+[s['rgb'][k] for k in ['PSNR','SSIM','LPIPS']]+[s['rgb_worker_seconds'],s['render_median_ms'],s['rgb_deploy_mib']])
        table('全量原三骨干及计算成本','all400_training_fit',['骨干','PSNR (dB)','SSIM','LPIPS','训练 (s)','渲染中位 (ms)','RGB部署 (MiB)'],rows,src,
              'Scaffold为固定UID0；模型primitive的anchor/Gaussian单位不同，不能直接比较数量。')
        src='outputs/quality/RGB-166/full-r1/evaluation-r1/result.json';d=read(ROOT/src)
        table('RGB166同轨迹训练长度','all400_training_fit',['step','PSNR (dB)','SSIM','LPIPS'],
              [[s]+[d['all400'][str(s)][k] for k in ['PSNR','SSIM','LPIPS']] for s in [30000,35000,40000,45000,50000]],src,
              '排除历史RGB165 parent_30k；本表只比较同一次RGB166训练端点。')
        rows=[];endpoint_rows=[]
        for code in ['436','437']:
            e=self.resolve('SEM-'+code)
            inf=self.result(f'outputs/quality/SEM-{code}/r2/inference/result.json')
            for step in [6000,8000,10000,12000]:rows.append(['SEM'+code,step,100*e['metrics'][f'training_miou_{step}']])
            for name,m in inf['endpoints']['12000']['metrics']['per_class'].items():
                endpoint_rows.append(['SEM'+code,name]+[100*m[k] for k in ['IoU','precision','recall']])
        table('SEM436/437完整训练曲线','all300_training_fit',['实验','step','五类mIoU (%)'],rows,'experiments/registry.yaml',
              '全300标签参与训练；12k选择仅为全量开发端点，不能外推空间最优。')
        table('后期12k端点逐类质量','all300_training_fit',['实验','类别','IoU (%)','precision (%)','recall (%)'],endpoint_rows,
              'outputs/quality/SEM-437/r2/inference/result.json','SEM436/437分别取各自封存inference/result.json；文件均随包。逐类取舍不可由总体均值代替。')
        src='outputs/quality/QUAL-026/r1/result.json';d=read(ROOT/src)['comparison_15M_minus_1M']
        table('QUAL026容量增量','spatial_development',['划分','PSNR差 (dB)','SSIM差','LPIPS差'],
              [[s]+[d[s][k] for k in ['PSNR','SSIM','LPIPS']] for s in ['train','holdout','buffer']],src,
              'RGB176(1.5M)减RGB175(1.0M)；RGB170不参加此因果比较。')
        save(self.out/'data/result_tables.json',tables)
        md=['# 已核对结果表','',self.content['subtitle'],'','每张表保留独立评价角色。数值来自保存记录，不进行新模型评分。']
        for t in tables:
            md+=['','## '+t['title'],'','证据角色：'+t['role'],'',t['note'],'','| '+' | '.join(t['columns'])+' |','| '+' | '.join(['---']*len(t['columns']))+' |']
            for row in t['rows']:md.append('| '+' | '.join(f'{x:.6f}' if isinstance(x,float) else str(x) for x in row)+' |')
            packaged=self.source_map.get(t['source'],'data/registry_snapshot.json')
            md+=['','来源：['+t['source']+']('+packaged+')']
        (self.out/'results_summary.md').write_text('\n'.join(md)+'\n')

    def documents(self):
        c=self.content
        intro=['# '+c['title'],'',c['subtitle'],'',c['intro'],'','## 快速使用','']+['- '+x for x in c['reading']]
        intro+=['','## 负责写作的伙伴先读这里','','[作者必读：在线性阅读中理解项目](author_guide.html) · [可编辑理解文档](author_guide.md)','',self.author_guide['purpose']]
        for route in self.author_guide['reading_route']:intro+=['','**'+route['title']+'：** '+route['text']]
        for role in c['roles']:intro+=['','### '+role['name'],'',role['model']+'。'+role['description']]
        intro+=['','## 证据边界','']+['- '+x for x in c['limits']]
        (self.out/'README.md').write_text('\n'.join(intro)+'\n\n离线入口：[index.html](index.html)。正文底稿：[paper_notes.md](paper_notes.md)。完整历程：[development_history.md](development_history.md)。图注：[captions.md](captions.md)。\n')
        history=['# 完整开发历程','',c['subtitle'],'',c['intro']]
        for p in c['phases']:
            history+=['','## '+p['id']+' · '+p['title'],'',p['period'],'','**问题：** '+p['question'],'','**尝试：** '+p['approach'],'','**结果：** '+p['result'],'','**决定与边界：** '+p['decision'],'','**写作用途：** '+p['use'],'','来源：'+self.refs_md(p['refs'])]
        (self.out/'development_history.md').write_text('\n'.join(history)+'\n')
        paper=['# 论文写作底稿','',c['subtitle'],'','先读[作者必读](author_guide.html)，完成任务、方法、证据边界与失败分析的理解，再使用以下段落。','','以下为可改写段落，不是已提交英文论文。最终模型身份、作者贡献与参考文献由团队定稿。']
        for s in c['paper_sections']:paper+=['','## '+s['title'],'','建议位置：'+s['placement'],'',s['text'],'','来源：'+self.refs_md(s['refs'])]
        paper+=['','## 写作核对','']+['- '+x for x in c['limits']]
        paper+=['','第三方方法来源与许可证索引：[third_party 历史记录](evidence/context/third_party__README.txt)。原始论文链接可从该记录追溯；本次未重新核验外部网页。']
        (self.out/'paper_notes.md').write_text('\n'.join(paper)+'\n')
        captions=['# 图片与视频中英图注','',c['subtitle'],'','图片共享图例：background 深灰、deck 蓝、stay_cable 黄、tower 红、foundation 紫。视频使用各自下列图例。']
        for f in self.figures:
            captions+=['','## '+f['id']+' · '+f['title'],'','中文：'+f['caption_zh'],'','English: '+f['caption_en'],'','证据角色：'+f['role'],'','样例选择：'+f['selection'],'','来源：'+self.refs_md(f['refs']),'','素材：'+ ' · '.join(f'[{Path(x).name}]({x})' for x in f['files']),'','原尺寸单图：'+(' · '.join(f'[{Path(x).name}]({x})' for x in f['native']) or '本组为可编辑SVG图表／方法示意，无原生照片单图。')]
        for v in self.videos:captions+=['','## '+v['id']+' · '+v['title'],'','中文：'+v['caption_zh'],'','English: '+v['caption_en'],'','证据角色：'+v['role'],'','图例：'+v['legend'],'',f"保存规格：{v['frame_count']}帧，{v['fps']} fps，{v['duration']:.2f}秒。",'',f"[视频]({v['path']}) · "+self.refs_md(v['refs'])]
        (self.out/'captions.md').write_text('\n'.join(captions)+'\n')
        gaps=['# 历史缺项与口径说明','','共'+str(len(self.gaps))+'项报告／配置缺项，逐条保留；没有图片或视频缺失。']
        for g in self.gaps:
            why='早期仅有历史摘要，原始产物未保存' if g['experiment_id'].startswith('E') else '阶段汇总或前置门未通过后未创建独立配置；以随包报告为准'
            gaps+=['','- '+g['experiment_id']+' / '+g['kind']+'：'+why+'。']
        gaps+=['','另外，RGB132—135使用config_path而非config字段，已找到原配置并纳入包；SEM002为单独阶段汇总报告，随包保存但不增加351条登记分母。',
               '', '报告中的未随包引用指大型运行产物或原目录上下文；主要科学结果JSON、报告和配置均有独立快照。个人绝对路径统一替换为$PROJECT_ROOT/$USER_HOME，源文件SHA256仍保留。']
        (self.out/'source_notes.md').write_text('\n'.join(gaps)+'\n')

    def refs_md(self,refs):
        return ' · '.join('['+r+'](evidence/reports/'+self.resolve(r)['experiment_id']+'.html)' for r in refs)

    def author_documents(self):
        """One evidence-linked teaching source generates the readable page and text."""
        guide=self.author_guide
        figures={f['id']:f for f in self.figures}; videos={v['id']:v for v in self.videos}
        md=['# '+guide['title'],'',self.content['subtitle'],'',guide['purpose'],
            '','[材料包首页](index.html) · [论文底稿](paper_notes.md) · [完整历程](development_history.md)','',
            '## 三遍阅读路线']
        routes=[]
        for r in guide['reading_route']:
            md+=['','### '+r['title'],'',r['text']]
            routes.append('<article><h3>'+inline(r['title'])+'</h3><p>'+inline(r['text'])+'</p></article>')
        toc=[];chapters=[]
        preview={'A04':'F02','A05':'F03','A09':'F09','A10':'F10'}
        for s in guide['sections']:
            sid=s['id'];toc.append(f'<li><a href="#{sid}">{sid} · {inline(s["title"])}</a></li>')
            md+=['','## '+sid+' · '+s['title'],'','**本章问题：** '+s['question']]
            for paragraph in s['paragraphs']:md+=['',paragraph]
            media=[]
            for fid in s['figures']:
                f=figures[fid];media.append((fid+' '+f['title'],'index.html#'+fid))
            for vid in s['videos']:
                v=videos[vid];media.append((vid+' '+v['title'],'index.html#'+vid))
            links=' · '.join(f'<a href="{target}">{inline(label)}</a>' for label,target in media)
            md+=['','**配合查看：** '+' · '.join(f'[{label}]({target})' for label,target in media),
                 '','**原始证据：** '+self.refs_md(s['refs'])]
            figure_html=''
            if sid in preview:
                fid=preview[sid];f=figures[fid]
                src=next(p for p in f['files'] if p.endswith('.png'))
                figure_html=f'<figure class="guide-figure"><a href="{src}"><img src="{src}" loading="lazy" alt="{inline(f["title"])}"></a><figcaption>{fid} · {inline(f["caption_zh"])} <a href="index.html#{fid}">查看本组全部素材与图注</a></figcaption></figure>'
                md+=['',f'![{fid} {f["title"]}]({src})']
            q=s['checkpoint']
            md+=['','**应掌握的结论：** '+s['takeaway'],'','**自测：** '+q['question'],
                 '','**参考回答：** '+q['answer']]
            chapters.append(f'<section id="{sid}" class="guide-chapter"><p class="eyebrow">{sid} / PROJECT UNDERSTANDING</p><h2>{inline(s["title"])}</h2><p class="guide-question">{inline(s["question"])}</p>'+''.join('<p>'+inline(p)+'</p>' for p in s['paragraphs'])+figure_html+f'<p class="guide-media"><b>配合查看：</b>{links}</p><p class="source"><b>原始证据：</b>{self.refs_html(s["refs"])}</p><aside class="guide-takeaway"><b>应掌握的结论</b><p>{inline(s["takeaway"])}</p></aside><details class="guide-check"><summary>自测：{inline(q["question"])}</summary><p><b>参考回答：</b>{inline(q["answer"])}</p></details><p class="meta"><a href="#contents">返回阅读目录</a></p></section>')
        md+=['','## 术语速查']
        terms=[]
        for item in guide['glossary']:
            md+=['','- **'+item['term']+'：** '+item['meaning']]
            terms.append('<dt>'+inline(item['term'])+'</dt><dd>'+inline(item['meaning'])+'</dd>')
        context=[('指标定义','evidence/context/METRICS.txt'),('相机与输出接口','evidence/context/docs__submission__interface.txt'),('第三方方法来源','evidence/context/third_party__README.txt'),('缺项与来源边界','source_notes.md')]
        md+=['','## 深入核对','', ' · '.join(f'[{label}]({href})' for label,href in context)]
        page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{inline(guide['title'])}</title><link rel="stylesheet" href="style.css"></head><body>
<header><a class="brand" href="index.html">IC-SHM 2026 / PAPER HANDOFF</a><nav aria-label="作者文档导航"><a href="#contents">阅读目录</a><a href="#glossary">术语</a><a href="index.html#figures">图片与视频</a><a href="index.html#experiments">实验索引</a></nav></header>
<main class="author-guide"><section id="contents"><p class="eyebrow">{inline(self.content['subtitle'])}</p><h1>理解项目之后<br>再写论文</h1><p class="lead">{inline(guide['purpose'])}</p><div class="downloads"><a href="author_guide.md" download>下载可编辑理解文档</a><a href="paper_notes.md" download>论文段落底稿</a><a href="index.html">返回完整材料包</a></div><div class="guide-route">{''.join(routes)}</div><h2>十二个必须讲清楚的问题</h2><ol class="guide-toc">{''.join(toc)}</ol><p class="meta">正文连续阅读；自测请先自行回答，再展开核对。原始报告保留历史语境，当前模型身份以A04与ADR-175为准。</p></section>
{''.join(chapters)}<section id="glossary"><h2>术语速查</h2><dl class="guide-glossary">{''.join(terms)}</dl><h3>深入核对</h3><p>{' · '.join(f'<a href="{href}">{label}</a>' for label,href in context)}</p></section><footer>REPORT-004 · 作者理解版 v2 · 证据截止2026-09-23 · 本文整理不包含新实验或新模型结论。</footer></main></body></html>'''
        sources=[ROOT/'scripts/report_assets/paper_handoff_author_guide.json',Path(__file__)]
        for name,content in [('author_guide.md','\n'.join(md)+'\n'),('author_guide.html',page)]:
            target=self.out/name;target.write_text(content)
            self.asset(target,sources,'evidence-linked author guide; shared structured content, no new scientific measurements')
        self.copy(sources[0],'data/author_guide.json')

    def webpage(self):
        c=self.content
        roles=''.join(f'<article><h3>{inline(x["name"])}</h3><strong>{inline(x["model"])}</strong><p>{inline(x["description"])}</p></article>' for x in c['roles'])
        phases=''.join(f'<article class="phase"><div class="phase-number">{p["id"]}</div><div><p class="meta">{inline(p["period"])}</p><h3>{inline(p["title"])}</h3><p><b>问题</b>　{inline(p["question"])}</p><p><b>尝试</b>　{inline(p["approach"])}</p><p><b>结果</b>　{inline(p["result"])}</p><p><b>决定</b>　{inline(p["decision"])}</p><p class="source">{self.refs_html(p["refs"])}</p></div></article>' for p in c['phases'])
        figures=[]
        for f in self.figures:
            preview=next((x for x in f['files'] if x.endswith('.png')),f['files'][0])
            links=' · '.join(f'<a href="{x}" download>{Path(x).name}</a>' for x in f['files'])
            native=' · '.join(f'<a href="{x}">{Path(x).name}</a>' for x in f['native'])
            additional=''.join(f'<a href="{x}"><img loading="lazy" src="{x}" alt="{f["id"]} paired example {i}"></a>' for i,x in enumerate(f['files']) if x!=preview and x.endswith(('.png','.jpg')))
            figures.append(f'<article class="figure" id="{f["id"]}"><p class="meta">{f["id"]} / {inline(f["role"])}</p><h3>{inline(f["title"])}</h3><a href="{preview}"><img src="{preview}" loading="lazy" alt="{inline(f["title"])}"></a><p>{inline(f["caption_zh"])}</p><p lang="en" class="english">{inline(f["caption_en"])}</p><p class="source">{self.refs_html(f["refs"])}</p><details><summary>下载本组图片、查看配对样例和原尺寸单图</summary><p class="file-links">{links}</p><p>选择依据：{inline(f["selection"])}</p><div class="extra-images">{additional}</div><p class="file-links">{native}</p></details></article>')
        videos=''.join(f'<article class="video" id="{v["id"]}"><h3>{v["id"]} · {inline(v["title"])}</h3><video controls preload="metadata" poster="{v["poster"]}" src="{v["path"]}"></video><p>{inline(v["caption_zh"])}</p><p class="meta">{inline(v["role"])} · {v["frame_count"]}帧 · {v["duration"]:.2f}秒</p><p class="legend">视频图例：{inline(v["legend"])}</p><p><a href="{v["path"]}" download>下载 MP4</a> · {self.refs_html(v["refs"])}</p></article>' for v in self.videos)
        paper=''.join(f'<details class="paper"><summary>{inline(s["title"])} <span>{inline(s["placement"])}</span></summary><p>{inline(s["text"])}</p><p class="source">{self.refs_html(s["refs"])}</p></details>' for s in c['paper_sections'])
        negatives=''.join(f'<tr><th>{inline(x["family"])}</th><td>{inline(x["finding"])}</td><td>{inline(x["lesson"])}<br>{self.refs_html(x["refs"])}</td></tr>' for x in c['negative_families'])
        rows=[]
        for e in self.records:
            summary=e['conclusion'] or '详见随包历史报告与配置；不由状态词推断质量结论。'
            rows.append(f'<tr data-state="{e["state"]}"><th><a href="{e["detail"]}">{inline(e["experiment_id"])}</a></th><td>{inline(e["kind"])}<br><b>{inline(e["state"])}</b><small>{inline(e["status"])}</small></td><td>{inline(e["split_id"])}<small>seed: {inline(e["seed"])}</small></td><td>{inline(summary)}</td></tr>')
        limits=''.join('<li>'+inline(x)+'</li>' for x in c['limits'])
        states=''.join(f'<option value="{x}">{x}</option>' for x in sorted({e['state'] for e in self.records}))
        body=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>开发历程与论文素材交接包</title><link rel="stylesheet" href="style.css"></head><body>
<header><a class="brand" href="#start">IC-SHM 2026 / PAPER HANDOFF</a><nav aria-label="章节"><a href="author_guide.html">作者必读</a><a href="#journey">开发历程</a><a href="#figures">论文图</a><a href="#videos">视频</a><a href="#writing">写作底稿</a><a href="#experiments">实验索引</a></nav></header>
<main><section id="start"><p class="eyebrow">{c['subtitle']}</p><h1>从桥梁重建<br>到可靠语义</h1><p class="lead">{inline(c['intro'])}</p><div class="downloads"><a href="author_guide.html">作者必读：先理解项目</a><a href="paper_notes.md" download>可编辑论文底稿</a><a href="development_history.md" download>完整开发历程</a><a href="captions.md" download>中英图注与素材清单</a></div><div class="facts"><div><b>351</b><span>历史登记，含失败／诊断／计划</span></div><div><b>10</b><span>论文图片主题，附原尺寸素材</span></div><div><b>6</b><span>视频主题，全部本地可播放</span></div></div><div class="roles">{roles}</div><div class="reading"><h2>写作者阅读路线</h2><p>先读 <a href="author_guide.html">12章项目理解文档</a>：任务与数据 → 实际方法 → 关键转向 → 空间验证 → 失败与贡献边界。每章均有图、实验来源和自测参考回答。</p><p>再看 <a href="#F03">F03 早期大幅提升</a>、<a href="#F05">F05 强二维基线</a>、<a href="#F07">F07 空间确认</a>、<a href="#F09">F09 容量反例</a>，再按实验 ID 回查证据。</p><p>图像为已保存结果；当前推荐未知视角RGB配方的匹配最终语义仍待完成。</p></div></section>
<section id="journey"><p class="eyebrow">01 / DEVELOPMENT</p><h2>每次转向，都有一个被检验的问题</h2>{phases}</section>
<section id="figures"><p class="eyebrow">02 / FIGURE LIBRARY</p><h2>十组可直接选用的论文图</h2><p>点击打开原图。统计图同时提供SVG；组合图附原尺寸单图和记录的裁剪范围。统一图片图例：deck蓝、stay_cable黄、tower红、foundation紫、background深灰。</p>{''.join(figures)}</section>
<section id="videos"><p class="eyebrow">03 / VIDEO LIBRARY</p><h2>六类视频，保留真实变化与伪影</h2><p>均为保存输出，无GT准确率结论。历史演进拆为三段，以免暗示不同系统为单变量对照。每段使用下面独立标注的图例。</p><div class="video-grid">{videos}</div></section>
<section id="writing"><p class="eyebrow">04 / WRITING NOTES</p><h2>从证据到论文段落</h2><p><a href="paper_notes.md" download>下载完整 Markdown 底稿</a> · <a href="results_summary.md" download>下载详细结果表</a>。展开可复制段落；方法名称与英文图注保留，中文正文供伙伴改写。</p>{paper}<h3>失败路线按假设组织</h3><div class="table-wrap"><table><thead><tr><th>方向</th><th>观察</th><th>写作结论与来源</th></tr></thead><tbody>{negatives}</tbody></table></div><h3>不能省略的边界</h3><ul>{limits}</ul></section>
<section id="experiments"><p class="eyebrow">05 / EVIDENCE INDEX</p><h2>351 条登记，逐项追溯</h2><p>状态标签为检索分组，原始状态始终显示；质量、运行完成与是否采用是不同问题。查看详情可读报告、配置，进一步检查commit、父实验和来源哈希。</p><div class="filters"><label>搜索实验／方法／划分<input id="search" type="search" placeholder="例如 SEM-161、spatial、failed"></label><label>运行状态<select id="state"><option value="">全部</option>{states}</select></label><output id="count" aria-live="polite">351 / 351</output></div><div class="table-wrap"><table id="experiment-table"><thead><tr><th>实验</th><th>类型／原始状态</th><th>划分与seed</th><th>登记结论</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p id="empty" hidden>没有匹配记录，请更换关键词。</p><p><a href="data/experiment_index.json" download>结构化实验索引</a> · <a href="data/asset_manifest.json" download>素材来源和SHA256</a> · <a href="data/gaps.json">缺失与历史边界</a> · <a href="evidence/context/DECISIONS.txt">完整决策历史</a></p></section>
<footer><p>REPORT-004-PAPER-HANDOFF · 本地离线材料 · 无训练／新推理／上传</p><p>来源：351条历史登记及已保存实验结果。新资料包独立保存，旧模型、报告页和素材未覆盖。</p></footer></main><script src="app.js"></script></body></html>'''
        (self.out/'index.html').write_text(body)
        for src,dest in [('paper_handoff.css','style.css'),('paper_handoff.js','app.js')]:self.copy(ROOT/'scripts/report_assets'/src,dest)

    def finish(self):
        # A rebuilt artifact must not inherit an earlier revision's passing receipt.
        save(self.out/'data/verification.json',dict(handoff_revision=2,status='not_yet_audited',
            note='Run independent source, browser and package checks for this build before distribution.'))
        save(self.out/'data/figures.json',self.figures);save(self.out/'data/videos.json',self.videos)
        save(self.out/'data/gaps.json',self.gaps);save(self.out/'data/metrics.json',self.metrics)
        save(self.out/'data/content.json',self.content)
        save(self.out/'data/asset_manifest.json',{'schema_version':1,'experiment_id':ID,'assets':self.assets})
        save(self.out/'data/source_hashes.json',self.sources)
        save(self.out/'data/build.json',dict(experiment_id=ID,evidence_cutoff='2026-09-23',built_date='2026-09-29',
             source_registry_entries=len(self.registry),source_report_count=sum(
                 (ROOT/(e.get('report_path') or f"experiments/runs/{e['experiment_id']}/report.md")).is_file()
                 for e in [*self.registry,*self.supplemental]),
             state_counts=dict(Counter(e['state'] for e in self.records)),figure_groups=len(self.figures),videos=len(self.videos),gap_count=len(self.gaps),
             handoff_revision=2,author_guide_chapters=len(self.author_guide['sections']),author_guide_terms=len(self.author_guide['glossary']),
             palette=PALETTE.tolist(),classes=CLASSES,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             working_tree='pre-existing uncommitted work; each source separately hash-bound',builder_sha256=digest(Path(__file__)),
             no_training=True,no_inference=True,no_model_selection=True,no_upload=True))
        files={str(p.relative_to(self.out)):{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(self.out.rglob('*')) if p.is_file() and p.name!='package_manifest.json'}
        save(self.out/'package_manifest.json',{'schema_version':1,'experiment_id':ID,'files':files})
        print(json.dumps({'status':'built','files':len(files),'bytes':sum(x['bytes'] for x in files.values()),'gaps':len(self.gaps),'output':str(self.out)},ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'outputs/paper_handoff_20260929');args=parser.parse_args()
    p=Package(args.output.resolve());p.evidence();p.figures_build();p.videos_build();p.result_tables();p.documents();p.author_documents();p.webpage();p.finish()


if __name__=='__main__':main()
