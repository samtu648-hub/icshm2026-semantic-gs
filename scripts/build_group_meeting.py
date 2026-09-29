#!/usr/bin/env python3
"""Build the offline meeting deck using saved evidence; no model execution."""
import argparse, hashlib, html, json, os, re, shutil, subprocess, zipfile
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'scripts/report_assets'

def read(p): return json.loads(p.read_text())
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()
def dump(p, d): p.write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n')
def clean(s):
    # Local source snapshots retain scientific values, but remove personal roots.
    return re.sub(r'/(?:home|Users)/[^/\s"<>]+', '$USER_HOME', s)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, default=ROOT/'outputs/group_meeting_20260915'); a=ap.parse_args()
    out=a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    for n in ['assets/images','assets/videos','sources','data']: (out/n).mkdir(exist_ok=True, parents=True)
    manifest=[]; seen={}
    def copy(src, dest=None):
        logical=Path(os.path.abspath(src))
        src=src.resolve()
        if src in seen: return seen[src]
        if not src.is_file(): raise FileNotFoundError(src)
        h=sha(src); dest=dest or 'assets/images/'+h[:18]+src.suffix
        target=out/dest;target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src,target);seen[src]=dest
        label=str(logical.relative_to(ROOT)) if logical.is_relative_to(ROOT) else logical.name
        manifest.append(dict(source=label,path=dest,sha256=h,bytes=target.stat().st_size))
        return dest
    records=read(ROOT/'experiments/registry.yaml')['experiments']; byid={x['experiment_id']:x for x in records}
    def record(prefix): return next(x for x in records if x['experiment_id']==prefix or x['experiment_id'].startswith(prefix+'-'))
    def metrics(prefix):
        r=record(prefix);return r.get('metrics_summary',r.get('metrics'))
    old=read(ROOT/'outputs/research_report_20260908/data/content.json')
    backbone=ROOT/'outputs/quality/backbone_comparison/browser-r1'
    newer=ROOT/'outputs/quality/SEM-375/r1/comparison'
    cross=ROOT/'outputs/quality/SEM-376/r1/comparison'
    audit=read(cross.parent/'audit.json')
    assert audit['fresh_probability_exact'] and audit['frozen18_exact'] and audit['rgb_inputs_unchanged']
    assert audit['training_steps']==0
    bb=read(backbone/'data.json');sc=read(newer/'data.json');cr=read(cross/'data.json')
    # Reuse existing scene / ROI / palette metadata and native PNGs.
    galleries={}
    for key, base, data, field in [('backbone',backbone,bb,'images'),('matched',newer,sc,'items'),('cross',cross,cr,'items')]:
        galleries[key]=[]
        for item in data[field]:
            if item['name'] not in ['159.png','160.png','140.png','146.png','081.png','057.png','064.png','054.png','174.png']: continue
            row={k:item[k] for k in ['name','box','structure_box']};row['tiles']=[]
            for t in item['tiles']: row['tiles'].append(dict(label=t['label'],path=copy(base/t['path'])))
            galleries[key].append(row)
    for key,name,tile,boxkey in [('cable','081.png',0,'structure_box'),('foundation','057.png',0,'box'),('shore','159.png',1,'box')]:
        row=next(x for x in galleries['backbone'] if x['name']==name)
        crop_box=[680,290,866,400] if key=='foundation' else row[boxkey]
        Image.open(out/row['tiles'][tile]['path']).crop(crop_box).save(out/f'assets/images/challenge-{key}.png')
    historical=[]
    for view in ['159','081','140']:
        tiles=[]
        for key,label in [('reference','真实照片'),('rgb','冻结 RGB030'),('gt','GT'),('sem001','SEM001'),('sem111','SEM111'),('sem141','SEM141'),('sem151','SEM151'),('sem161','SEM161'),('sem329','SEM329')]:
            tiles.append(dict(label=label,path=copy(ROOT/f'outputs/research_report_20260908/assets/images/{view}/{key}.png')))
        historical.append(dict(name=view+'.png',box=[150,620,850,989] if view=='159' else [0,0,1320,989],structure_box=[0,0,1320,989],tiles=tiles))
    galleries['history']=historical
    videos={}
    for key,src,label in [
        ('intro','outputs/research_report_20260908/assets/videos/rgb-sem329-overlay.mp4','历史冻结 RGB030 + SEM329 语义叠加 · 144帧 / 6秒'),
        ('backbone',str(backbone.relative_to(ROOT)/'three-backbones-trajectory.mp4'),'Scaffold UID0 / Standard / Mip · 上RGB，下语义 · 144帧 / 12秒'),
        ('matched',str(newer.relative_to(ROOT)/'trajectory.mp4'),'旧Scaffold / Standard / Mip / 新Scaffold · 上RGB，下语义'),
        ('cross',str(cross.relative_to(ROOT)/'trajectory.mp4'),'Standard×SEM371 / Standard×SEM375 / Scaffold×SEM371 / Scaffold×SEM375'),
        ('joint','outputs/quality/SEM-373/browser-r1/matched-trajectory.mp4','SEM373 控制与联合优化 · 详见来源记录'),
        ('depth','outputs/quality/SEM-374/browser-r1/matched-trajectory.mp4','SEM374 控制与稀疏深度监督 · 详见来源记录')]:
        path=copy(ROOT/src,f'assets/videos/{key}.mp4');poster=f'assets/videos/{key}.jpg'
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(out/path),'-frames:v','1','-vf','scale=1280:-1',str(out/poster)],check=True)
        probe=read_json_cmd(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,codec_name:format=duration','-of','json',str(out/path)])
        assert int(probe['streams'][0]['nb_frames'])==144
        videos[key]=dict(path=path,poster=poster,label=label,probe=probe)
    # Original reports/configs + registry provide full provenance and negative results.
    provenance=[]
    for r in records:
        exp=r['experiment_id']; sections=[]; sources=[]
        for field in ['report_path','config']:
            rel=r.get(field)
            if not isinstance(rel,str): continue
            p=ROOT/rel
            if not p.is_file():continue
            text=clean(p.read_text());sources.append(dict(path=rel,sha256=sha(p)))
            sections.append('<h2>'+html.escape(rel)+'</h2><pre>'+html.escape(text)+'</pre>')
        filename='sources/'+exp+'.html'
        entry=dict(r,source_link=filename,source_files=sources);provenance.append(entry)
        title=html.escape(exp)
        (out/filename).write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>'+title+'</title><style>body{font:16px/1.8 system-ui;max-width:1050px;margin:40px auto;padding:20px;color:#172a33}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f5;padding:20px}a{color:#146957}</style><a href="../index.html#appendix">← 返回证据附录</a><h1>'+title+'</h1><p>构建时来源快照；历史“当前/下一步”仅代表原记录阶段。个人根目录已脱敏。</p>'+''.join(sections)+'<h2>登记及来源哈希</h2><pre>'+html.escape(clean(json.dumps(entry,ensure_ascii=False,indent=2)))+'</pre></html>')
    history=old['history'][:5]
    later=[dict(label='SEM161 正式基线',value=metrics('SEM-350')['miou_with_background']['old_sem161'],id=record('SEM-161')['experiment_id']),dict(label='SEM329 第二源复核',value=next(x['miou'] for x in old['history'] if x['short']=='SEM329'),id=record('SEM-329')['experiment_id']),dict(label='SEM349 原始 M2F',value=metrics('SEM-350')['miou_with_background']['new_raw'],id=record('SEM-350')['experiment_id']),dict(label='SEM363 学习式融合',value=metrics('SEM-363')['mIoU_with_background'],id=record('SEM-363')['experiment_id'])]
    # Data inherited from sealed artifacts; presentation creates no new quality scores.
    D=dict(date='2026-09-15',galleries=galleries,videos=videos,history=history,later=later,backbone=bb['summaries'],scaffold_native=bb['scaffold_native_appearance'],matched=sc['summaries'],cross=cr['summaries'],rgb136=metrics('RGB-136'),registry=provenance,palette=old['palette'],class_names=old['class_names'])
    D=json.loads(clean(json.dumps(D,ensure_ascii=False)))
    dump(out/'data/content.json',D)
    for file in ['group_meeting.css','group_meeting.js']:
        shutil.copyfile(ASSETS/file,out/file.replace('group_meeting','meeting'))
    template=(ASSETS/'group_meeting.html').read_text()
    payload=json.dumps(D,ensure_ascii=False).replace('<','\\u003c')
    (out/'index.html').write_text(template.replace('__DATA__',payload))
    dump(out/'data/asset_manifest.json',manifest)
    sourcefiles=['scripts/build_group_meeting.py','scripts/report_assets/group_meeting.html','scripts/report_assets/group_meeting.css','scripts/report_assets/group_meeting.js','experiments/registry.yaml','PROJECT_STATUS.md']
    build=dict(built_at=datetime.now(timezone.utc).isoformat(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),uncommitted_worktree=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)),sources={p:sha(ROOT/p) for p in sourcefiles},scope='Saved evidence only; no training/inference or model mutation',presentation_id='REPORT-001-GROUP-MEETING',parents=['VIS-006-SEM329-FLYTHROUGH','QUAL-007-BACKBONE-COMPARISON','SEM-376-CROSS-RGB-INFERENCE'])
    dump(out/'data/build_provenance.json',build)
    (out/'README.txt').write_text('离线组会汇报｜2026-09-15\n解压整个文件夹，双击 index.html。保留 assets、sources、data 和 meeting.js/css。\n左右方向键翻页；目录跳转；证据附录可返回原页；全屏键位于右上。视频默认暂停，可调倍速。\n每页讲述提示可展开。图像选择对照与视角后，点击困难区域/结构区域；拖拽任一图可同步裁剪。\n统计依据见来源快照；历史开发集与全数据拟合分开解释。无GT轨迹不提供准确率。\n')
    files={p.relative_to(out).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.rglob('*') if p.is_file() and p.name!='package_manifest.json'}
    dump(out/'package_manifest.json',dict(files=files))
    with zipfile.ZipFile(out.with_suffix('.zip'),'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,out.name+'/'+p.relative_to(out).as_posix())
    print(json.dumps(dict(output=str(out),experiments=len(records),media=len(manifest),videos=len(videos)),ensure_ascii=False))

def read_json_cmd(cmd):return json.loads(subprocess.check_output(cmd,text=True))
if __name__=='__main__': main()
