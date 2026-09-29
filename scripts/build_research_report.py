#!/usr/bin/env python3
"""Build a portable Chinese research report from sealed results and real media."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PALETTE = np.array([[30,30,30],[80,160,245],[245,210,65],[220,80,80],[170,100,190]],dtype=np.uint8)
CLASSES = ["background","deck","stay_cable","tower","foundation"]
LABELS = ["背景","桥面","索结构","桥塔","基础"]
SAMPLES = ["054","081","140","159","160","183","180","145","088","061"]
NOTES = {
    "054":"固定样例：索线与桥塔。第二源未改变SEM326；原有结构误差仍需观察。",
    "081":"固定样例：桥面和遮挡。SEM161的桥面补充存在局部代价；新基础复核不改变桥面预测。",
    "140":"固定样例：基础边缘。第二源恢复40像素，其中20是真基础、20是真背景，收益与误报同时可见。",
    "159":"固定样例：斜视桥面与水岸。第二源未改变SEM326；适合检查桥面连续性与重建伪影。",
    "160":"固定样例：近景桥面与复杂遮挡。类别专家累积改善后仍有边界问题；第二源在此无新增变化。",
    "183":"固定样例：细基础结构。恢复101个真基础像素，同时返回52个背景及4个其它错误类别像素。",
    "180":"封存逐图结果中的最大对父增益（+0.718368pp）：第二源救回116个真基础，同时返回77个背景。基础边缘收益与误报并存。",
    "145":"封存逐图结果中的最大对父退化（-0.487149pp）：第二源恢复的19像素全部是真背景，本例没有救回真基础。",
    "088":"封存逐图结果中的最大对正式基线退化（-0.976516pp）：纠正185个背景误报，却误删191个真基础；第二源没有恢复。",
    "061":"依据SEM161事前固定最差8张集合补充：困难视角代表，三臂预测完全相同。基础复核没有改变既有错误，不能解释为困难场景已经解决。",
}
ROUTES = [
    {"name":"RGB骨干","question":"怎样得到可靠的新视角RGB？","attempt":"同条件比较Standard 3DGS、Mip-Splatting、Scaffold-GS；另作MCMC预检。","result":"Scaffold以PSNR 20.596 dB / SSIM 0.8502 / LPIPS-Alex 0.3224胜过主比较候选，成为冻结骨干；MCMC预检未胜匹配控制。","decision":"采用Scaffold；保留工程回退。","ids":["RGB-060","RGB-030-FULL","RGB-011-R2-ALLOC","RGB-020-FULL","RGB-040-PF"]},
    {"name":"几何初始化与RGB改进","question":"几何先验、清点或背景建模能否改善重建？","attempt":"MapAnything / DA3初始化，以及appearance、支持度清点、延长pruning、瞬态与环境层。","result":"已有局部或对匹配控制的收益；困难视角、LPIPS或真实结构代价使当前配方未替换冻结RGB。","decision":"保留正负证据；按同轮控制和冻结骨干分别判断。","ids":["GEO-010-10K","GEO-020-3K-INT-R1","RGB-074-RESIDUAL-SUPPORT-PRUNE-TOL-PF","RGB-080-LATE-PRUNE","RGB-092-TRANSIENT","RGB-101-ENVIRONMENT"]},
    {"name":"三维语义监督","question":"五类标签能否直接稳定地附着到三维表示？","attempt":"SEM001静态基底；贡献门控、冲突正则及受限视角残差。","result":"静态基底mIoU为61.610%；索背景误报明显。部分贡献门控减少误报，却伤害桥塔；冲突正则和残差未形成保留方案。","decision":"保留SEM001作为3D基底和回退。","ids":["SEM-001","SEM-002","SEM-003","SEM-004"]},
    {"name":"结构清理与一致性","question":"清掉不符合桥梁结构的预测是否有效？","attempt":"anchor清理、Hough拓扑、索网模板和跨视角anchor支持。","result":"减少误报常伴随真实索线的大量删除；同源错误可在多视角反复出现。SEM080的check索recall仅保留12.49%。","decision":"当前硬清理配方停止；一致性不是独立正确性证据。","ids":["SEM-070-POST-CLEAN","SEM-080-STRUCTURAL-DIAG","SEM-090-CABLE-NETWORK-DIAG","SEM-130-ANCHOR-CROSSVIEW-CONSENSUS-OOF"]},
    {"name":"渲染域训练","question":"二维识别能否利用重建RGB补充三维语义？","attempt":"真实RGB与渲染RGB域差诊断、渲染域SegFormer训练、保护融合。","result":"SEM102直接替换伤害deck/tower；SEM111保护融合将同74图mIoU提高到76.063%。","decision":"保留二维证据与三维类别保护组合。","ids":["SEM-100-RENDER-SEG-PF","SEM-101-RENDERED-SEG-OOF","SEM-102-RENDERED-SEG-FINAL","SEM-111-PROTECTED-FUSION-FINAL"]},
    {"name":"类别专家","question":"薄索、基础误报和桥面缺口如何分别处理？","attempt":"原生分辨率拉索二分类、foundation单向审核、deck单向恢复。","result":"SEM141→151→161的mIoU为77.490%→83.093%→83.387%；基础审核贡献显著，桥面恢复仍有背景误补。","decision":"组成正式SEM161，并保留逐级回退。","ids":["SEM-141-CABLE-BINARY-FINAL","SEM-151-FOUNDATION-GATE-FINAL","SEM-161-DECK-BINARY-FINAL"]},
    {"name":"几何、边界与点级细化","question":"更细表示和几何证据能否修复结构边缘？","attempt":"多视角投影、XYZ/RGBD、offset责任、COLMAP轨迹、点级／嵌套OOF、foundation边界与距离损失。","result":"一些匹配消融改善，但召回、空间稳定性或覆盖未满足原条件；稀疏轨迹几乎未覆盖桥面漏检。","decision":"当前配方未替代SEM161；稀疏覆盖失败不能外推稠密方法无效。","ids":["SEM-200-MV-DECK-GEOMETRY-DIAG","SEM-230-OFFSET-SEMANTIC-OOF","SEM-240-OFFSET-RESPONSIBILITY-OOF","SEM-250-COLMAP-TRACK-DECK-DIAG","SEM-263-NESTED-POINT-OOF","SEM-300-FOUNDATION-SIGNED-DISTANCE-OOF"]},
    {"name":"DINO特征与辅助复核","question":"异质预训练特征能否提供额外纠错信息？","attempt":"DINOv3来源预检；公开DINOv2对冻结MiT；限制为基础复核后做OOF和final。","result":"DINOv3是权重权限阻塞。DINOv2复核有正面OOF证据，但SEM314最终召回下降3.260个百分点，169–175区域退化。","decision":"保留OOF信号；当前final配方未晋级。","ids":["SEM-310-DINOV3-FROZEN-FEATURE-OOF","SEM-311-DINOV2-FROZEN-FEATURE-OOF","SEM-313-DINOV2-FOUNDATION-REVIEW-OOF","SEM-314-DINOV2-FOUNDATION-REVIEW-FINAL"]},
    {"name":"稠密对应与第二参考源","question":"可靠参考视角能否纠正基础误报并减少误删？","attempt":"RoMa可运行性、pose-top3选源、几何／标签诊断、定向基础复核、第二可靠源否决。","result":"SEM326有最终开发收益；SEM329在同74张图上进一步提高IoU与召回折中，mIoU 83.705003%。","decision":"SEM329为优先研究候选；SEM326精度略高，SEM161仍是正式包。","ids":["SEM-320-ROMAV2-DENSE-PREFLIGHT","SEM-324-DENSE-LABEL-AGREEMENT","SEM-326-DENSE-FOUNDATION-REVIEW-FINAL","SEM-327-SOURCE-BOUNDARY-DIAG","SEM-329-SECOND-SOURCE-VETO-FINAL"]},
]


def read(path):return json.loads(path.read_text(encoding="utf-8"))
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(2**20),b""):h.update(chunk)
    return h.hexdigest()
def write_json(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def load_image(path):
    with Image.open(path) as im:return np.asarray(im).copy()


def bbox(mask,pad=24):
    y,x=np.where(mask)
    if not x.size:return [0,0,mask.shape[1],mask.shape[0]]
    return [max(0,int(x.min())-pad),max(0,int(y.min())-pad),min(mask.shape[1],int(x.max())+pad+1),min(mask.shape[0],int(y.max())+pad+1)]


def panel(images,titles,path,columns=3):
    w,h=images[0].size;gap=12;rows=(len(images)+columns-1)//columns
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",18 if w<400 else 22)
    probe=ImageDraw.Draw(Image.new("RGB",(1,1)));wrapped=[]
    for title in titles:
        lines=[];line=""
        for word in title.split():
            proposed=(line+" "+word).strip()
            if line and probe.textlength(proposed,font=font)>w-16:lines.append(line);line=word
            else:line=proposed
        wrapped.append("\n".join(lines+[line]))
    head=max(50,max(t.count("\n")+1 for t in wrapped)*28+18)
    out=Image.new("RGB",(columns*(w+gap)-gap,rows*(h+head+gap)-gap),(250,250,247));draw=ImageDraw.Draw(out)
    for i,(im,title) in enumerate(zip(images,wrapped)):
        x=(i%columns)*(w+gap);y=(i//columns)*(h+head+gap)
        draw.multiline_text((x+8,y+8),title,font=font,fill=(33,52,47),spacing=4);out.paste(im,(x,y+head))
        if out.crop((x,y+head,x+w,y+head+h)).tobytes()!=im.tobytes():raise ValueError("Panel pixels changed")
    out.save(path)


def gather(args):
    root=args.project_root;out=args.output_dir;artifact=args.artifact_root
    for name in ("assets/images","assets/charts","assets/videos","sources","data"):(out/name).mkdir(parents=True,exist_ok=True)
    registry=read(root/"experiments/registry.yaml");byid={r["experiment_id"]:r for r in registry["experiments"]}
    scorepath=root/"outputs/sem329/developer-score-20260908-r1/result.json"
    if digest(scorepath)!="1fa850cce879345bd9c323be96ae840431d775cb8070418772608053c88792c1":raise ValueError("Sealed quality result changed")
    score=read(scorepath)
    # Recalculate chart quantities from sealed matrices, without reopening74 GT images.
    for arm in ("baseline","dense","candidate"):
        m=score["summary"][arm];cm=np.array(m["confusion_matrix_rows_gt_columns_prediction"],dtype=np.int64)
        iou=np.diag(cm)/(cm.sum(0)+cm.sum(1)-np.diag(cm))
        if abs(iou.mean()-m["mIoU_with_background"])>1e-12:raise ValueError("Stored mIoU does not match sealed counts")
    regcopy=[]
    for record in registry["experiments"]:
        exp=record["experiment_id"];safe=re.sub(r"[^A-Za-z0-9_-]","_",exp);item=dict(record);source=[]
        for field in ("report_path","config"):
            value=record.get(field,"")
            candidate=root/value if isinstance(value,str) else None
            if candidate is not None and candidate.is_file():
                text=candidate.read_text();source.append((field,value,digest(candidate),text))
        content="".join(f"<h2>{html.escape(k)} · {html.escape(path)}</h2><p>SHA256 {h}</p><pre>{html.escape(t)}</pre>" for k,path,h,t in source)
        sourcefile="sources/"+safe+".html"
        (out/sourcefile).write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'+html.escape(exp)+'</title><style>body{max-width:1100px;margin:40px auto;padding:20px;font:15px/1.8 sans-serif;color:#21342f}pre{white-space:pre-wrap;overflow-wrap:anywhere}h2{margin-top:40px}</style><a href="../index.html#registry">返回研究报告</a><h1>'+html.escape(exp)+'</h1><p>本页为构建时的实验来源快照；历史“当前/下一步”按原运行阶段理解。</p>'+content+'<h2>Registry record</h2><pre>'+html.escape(json.dumps(record,ensure_ascii=False,indent=2))+'</pre></html>',encoding="utf-8")
        item["source_link"]=sourcefile;item["provenance"]=[{"kind":k,"path":p,"sha256":h} for k,p,h,t in source];regcopy.append(item)
    source_manifest=[]
    def source(path,role):
        label=str(path.relative_to(root)) if path.is_relative_to(root) else str(path.relative_to(artifact)) if path.is_relative_to(artifact) else str(path.relative_to(args.data_root))
        source_manifest.append({"role":role,"path":label,"sha256":digest(path)})
    source(scorepath,"latest_paired_score")
    for p in (root/"experiments/registry.yaml",root/"docs/experiments/research_summary_20260908.md",root/"data/manifests/contest_dataset_v1.json",root/"experiments/splits/sem_001_scaffold_static_v1.json"):
        source(p,"report_source")
    (out/"sources/stage-summary.md").write_text((root/"docs/experiments/research_summary_20260908.md").read_text(),encoding="utf-8")
    history=[]
    for short,exp in [("SEM001","SEM-001"),("SEM111","SEM-111-PROTECTED-FUSION-FINAL"),("SEM141","SEM-141-CABLE-BINARY-FINAL"),("SEM151","SEM-151-FOUNDATION-GATE-FINAL"),("SEM161","SEM-161-DECK-BINARY-FINAL"),("SEM329","SEM-329-SECOND-SOURCE-VETO-FINAL")]:
        rec=byid[exp];history.append({"short":short,"experiment_id":exp,"miou":score["summary"]["candidate"]["mIoU_with_background"] if short=="SEM329" else rec["metrics_summary"]["mIoU_with_background"],"seed":rec["seed"],"git_commit":rec["git_commit"],"config":rec["config"],"split":rec["split_id"]})
    rgbs=[]
    for short,exp in [("Standard","RGB-011-R2-ALLOC"),("Mip","RGB-020-FULL"),("Scaffold","RGB-030-FULL")]:
        resultpath=artifact/"r1/runs"/exp/"r1_metrics_summary.json";r=read(resultpath);source(resultpath,exp)
        rgbs.append({"short":short,"experiment_id":exp,"metrics":r["metrics"],"source_sha256":digest(resultpath),"seed":byid[exp]["seed"],"split":byid[exp]["split_id"]})
    latest_seal=read(root/"outputs/sem329/developer-predict-20260908-r1/access_seal.json")
    label_hashes={r["image"]:r["label_png_sha256"] for r in read(root/"experiments/runs/SEM-001/label_manifest.json")["records"]}
    semroots={
        "sem001":artifact/"r2/sem_001/full/validation_semantic_png",
        "sem111":artifact/"r2/sem_111/final_protected_fusion/full-20260902-r1/predictions",
        "sem141":artifact/"r2/sem_141/developer_audit/final-20260904-r1/predictions",
        "sem151":artifact/"r2/sem_151/foundation_gate_final/full-20260904-r1/predictions",
        "sem161":artifact/"r2/sem_161/developer_audit/final-20260904-r1/predictions",
        "sem326":root/"outputs/sem326/predict-20260908-r1/predictions",
        "sem329":root/"outputs/sem329/developer-predict-20260908-r1/predictions",
    }
    predhash={r["target"]:r["sha256"] for r in read(root/"outputs/sem329/developer-predict-20260908-r1/result.json")["predictions"]}
    densehash={r["target"]:r["sha256"] for r in read(root/"outputs/sem326/predict-20260908-r1/result.json")["predictions"]}
    samples=[]
    for id in SAMPLES:
        name=id+".png";rgb_rel=next(p for p in latest_seal["rgb_sha256"] if Path(p).name==name)
        paths={"reference":artifact/"rgb_r1_spatial_v1/scene_400_eval/images"/name,"rgb":artifact/rgb_rel,"gt":artifact/"r2/sem_001/labels"/name,
               **{k:v/name for k,v in semroots.items()},"standard":artifact/"r1/runs/RGB-011-R2-ALLOC/r1_validation_renders"/name,
               "mip":artifact/"r1/runs/RGB-020-FULL/r1_validation_renders"/name}
        for key,expected in [("rgb",latest_seal["rgb_sha256"][rgb_rel]),("sem161",latest_seal["baseline_sha256"][name]),("sem326",densehash[name]),("sem329",predhash[name]),("gt",label_hashes[name])]:
            if digest(paths[key])!=expected:raise ValueError("Sealed input differs: "+key+"/"+name)
        arrays={k:load_image(p) for k,p in paths.items()}
        for key,a in arrays.items():
            if a.shape[:2]!=(989,1320):raise ValueError("Pixel space dimensions differ: "+key)
            if key.startswith("sem") or key=="gt":
                if a.ndim!=2 or a.dtype!=np.uint8 or a.max()>4:raise ValueError("Invalid semantic ID map")
        for k in (1,2,3):
            if any(not np.array_equal(arrays[x]==k,arrays["sem161"]==k) for x in ("sem326","sem329")):raise ValueError("Protected class mismatch")
        gt,base,dense,current=[arrays[k] for k in ("gt","sem161","sem326","sem329")]
        changes=(current!=base);restored=(current!=dense)
        effects=np.full((989,1320,3),238,dtype=np.uint8)
        effects[changes&(base!=gt)&(current==gt)]=[36,135,102]
        effects[changes&(base==gt)&(current!=gt)]=[210,65,56]
        effects[changes&(base!=gt)&(current!=gt)]=[220,162,34]
        restore_effect=np.full_like(effects,238)
        restore_effect[restored&(current==gt)]=[36,135,102]
        restore_effect[restored&(dense==gt)]=[210,65,56]
        restore_effect[restored&(current!=gt)&(dense!=gt)]=[220,162,34]
        visuals={k:PALETTE[a] if (k.startswith("sem") or k=="gt") else a for k,a in arrays.items()}
        overlay=arrays["rgb"].copy();fg=current>0;overlay[fg]=np.rint(.45*overlay[fg]+.55*PALETTE[current[fg]]).astype(np.uint8)
        visuals.update(overlay=overlay,effects=effects,restored=restore_effect)
        folder=out/"assets/images"/id;folder.mkdir(exist_ok=True);links={}
        for key,array in visuals.items():
            path=folder/(key+".png");Image.fromarray(array).save(path);links[key]=str(path.relative_to(out))
            if key in paths:source(paths[key],id+":"+key)
        # Keep original integer IDs available independently of colorized presentation.
        for key in ("gt","sem161","sem326","sem329"):
            shutil.copyfile(paths[key],folder/(key+"-ids.png"))
        fullkeys=["reference","rgb","gt","sem161","sem326","sem329"]
        titles=[id+" reference PINHOLE",id+" frozen Scaffold RGB","GT (reused developer)","SEM161 formal","SEM326 first-source","SEM329 second-source"]
        fullpath=folder/"native-comparison.png"
        panel([Image.fromarray(visuals[k]) for k in fullkeys],titles,fullpath)
        crop=bbox(restored if restored.any() else (changes if changes.any() else ((gt==4)|(base==4))))
        # At least128 source pixels per axis, where possible; preserve the actual native crops.
        x0,y0,x1,y1=crop
        for axis,limit in ((0,1320),(1,989)):
            lo,hi=(x0,x1) if axis==0 else (y0,y1)
            if hi-lo<128:
                lo=max(0,min(limit-128,(lo+hi)//2-64));hi=min(limit,lo+128)
            if axis==0:x0,x1=lo,hi
            else:y0,y1=lo,hi
        crop=[x0,y0,x1,y1];cropkeys=["rgb","gt","sem161","sem326","sem329","restored"]
        for key in cropkeys:
            path=folder/(key+"-crop.png");Image.fromarray(visuals[key][y0:y1,x0:x1]).save(path);links[key+"_crop"]=str(path.relative_to(out))
        croppath=folder/"native-crop-comparison.png"
        panel([Image.fromarray(visuals[k][y0:y1,x0:x1]) for k in cropkeys],titles[1:]+["Veto effect: green gain / red loss"],croppath)
        samples.append({"id":id,"name":name,"note":NOTES[id],"selection":"fixed six" if id in SAMPLES[:6] else "sealed metrics diagnostic selection",
                        "assets":links,"native_panel":str(fullpath.relative_to(out)),"crop_panel":str(croppath.relative_to(out)),"crop_xyxy":crop,
                        "miou":score["per_view"][name],"changed_pixels_vs_sem161":int(changes.sum()),"restored_pixels_vs_sem326":int(restored.sum()),
                        "source_hashes":{k:digest(p) for k,p in paths.items()},"space":"1320x989 PINHOLE"})
        print("prepared images "+id,flush=True)
    videos=[];dynamic_examples=[]
    if args.phase=="final":
        media=read(args.video_dir/"media_manifest.json")
        if media["frame_count"]!=144 or len(media["videos"])!=4:raise ValueError("Video set incomplete")
        for row in media["videos"]:
            src=args.video_dir/row["path"]
            if digest(src)!=row["sha256"]:raise ValueError("Video hash differs")
            dest=out/"assets/videos"/src.name;shutil.copyfile(src,dest)
            poster=args.video_dir/row["poster"];pdest=out/"assets/videos"/poster.name;shutil.copyfile(poster,pdest)
            videos.append({**row,"path":str(dest.relative_to(out)),"poster":str(pdest.relative_to(out))})
        shutil.copyfile(args.video_dir/"media_manifest.json",out/"data/video-manifest.json")
        source(args.video_dir/"result.json","VIS006_render_audit")
        source(args.video_dir/"media_manifest.json","VIS006_media_audit")
        render=read(args.video_dir/"result.json");byframe={r["name"]:r for r in render["frames"]}
        dynamic_folder=out/"assets/images/dynamic";dynamic_folder.mkdir(exist_ok=True)
        still=args.video_dir/"rgb-sem329-overlay-frame0072.png"
        shutil.copyfile(still,dynamic_folder/still.name);source(still,"VIS006:frame0072_overlay")
        dynamic_examples.append({"path":str((dynamic_folder/still.name).relative_to(out)),"caption":"3.00秒 / frame0072：近岸和桥面前景存在模糊、拉伸与浮点；叠加语义仍有不完整边缘。未做清点、美化或时间平滑。","frames":[72],"experiment_id":"VIS-006-SEM329-FLYTHROUGH"})
        masks=[]
        for frame in [71,72,73]:
            record=byframe[f"frame{frame:04d}.png"]["outputs"]["sem329"];src=args.video_dir/record["path"]
            if digest(src)!=record["sha256"]:raise ValueError("Temporal example source changed")
            masks.append(load_image(src));source(src,f"VIS006:frame{frame:04d}_SEM329")
        x0,y0,x1,y1=bbox(np.logical_or.reduce([m==4 for m in masks]),12)
        temporal=dynamic_folder/"sem329-consecutive-native-crops.png"
        panel([Image.fromarray(PALETTE[m][y0:y1,x0:x1]) for m in masks],[f"SEM329 frame{f:04d} / {f/24:.4f}s" for f in [71,72,73]],temporal)
        dynamic_examples.append({"path":str(temporal.relative_to(out)),"caption":"连续三帧的基础区域：相机移动时边缘和覆盖随帧变化，可在视频半速下检查闪烁。相同屏幕裁剪并非同一三维点配准；没有GT，不能把所有帧差认定为识别错误。","frames":[71,72,73],"crop_xyxy":[x0,y0,x1,y1],"experiment_id":"VIS-006-SEM329-FLYTHROUGH"})
    oldvideos=[("Scaffold + SEM001 · 历史三维语义","results/frozen_scaffold_sem001/flythrough_v1/frozen_scaffold_sem001_side_by_side.mp4"),
               ("Standard 3DGS · 历史同轨迹重建","results/frozen_standard_rgb/flythrough_v1/frozen_standard_rgb_flythrough.mp4"),
               ("SEM123 · 历史选择性索恢复","r2/vis_005/selective_recovery/flythrough-20260903-r1/semantic_dev030_vs_sem123_leader_vs_sem123_conservative.mp4")]
    legacy=[]
    for title,rel in oldvideos:
        src=artifact/rel;dest=out/"assets/videos"/src.name;shutil.copyfile(src,dest);source(src,"historical_video")
        legacy.append({"title":title,"path":str(dest.relative_to(out)),"sha256":digest(src)})
    data={"schema_version":1,"date":"2026-09-08","title":"从重建到可验证的语义纠错","registry":regcopy,"routes":ROUTES,
          "score":score,"history":history,"rgb":rgbs,"samples":samples,"class_names":CLASSES,"class_labels":LABELS,"palette":PALETTE.tolist(),
          "videos":videos,"legacy_videos":legacy,"dynamic_examples":dynamic_examples,"source_manifest":source_manifest,
          "protocol":"相同74张历史复用开发图；1320×989 PINHOLE；含背景五类全局mIoU；219训练参考源；seed 20260805（最新三臂）。",
          "report_scope":"阶段性研究截至SEM329；VIS006仅补新视角可视化，不产生新的质量分数。"}
    data["build_provenance"]={"git_commit":subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip(),
        "uncommitted_worktree":bool(subprocess.check_output(["git","-C",str(root),"status","--porcelain"],text=True)),
        "sources":{rel:digest(root/rel) for rel in ["scripts/build_research_report.py","scripts/plot_research_report.py","scripts/report_assets/research_report.html"]}}
    data["asset_experiments"]={key:{"experiment_id":exp,**{field:byid[exp].get(field,"not_available") for field in ["config","seed","split_id","git_commit","dataset_manifest"]}} for key,exp in
        [("standard","RGB-011-R2-ALLOC"),("mip","RGB-020-FULL"),("rgb","RGB-030-FULL"),("sem001","SEM-001"),("sem111","SEM-111-PROTECTED-FUSION-FINAL"),("sem141","SEM-141-CABLE-BINARY-FINAL"),("sem151","SEM-151-FOUNDATION-GATE-FINAL"),("sem161","SEM-161-DECK-BINARY-FINAL"),("sem326","SEM-326-DENSE-FOUNDATION-REVIEW-FINAL"),("sem329","SEM-329-SECOND-SOURCE-VETO-FINAL"),("videos","VIS-006-SEM329-FLYTHROUGH")]}
    write_json(out/"data/content.json",data);write_json(out/"data/provenance.json",source_manifest)
    return data


def make_html(args,data):
    template=(args.project_root/"scripts/report_assets/research_report.html").read_text()
    encoded=json.dumps(data,ensure_ascii=False).replace("</","<\\/")
    result=template.replace("__REPORT_DATA__",encoded)
    (args.output_dir/"index.html").write_text(result,encoding="utf-8")


def make_pdf(args,d):
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.colors import HexColor,Color
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import Paragraph
    from reportlab.lib.styles import ParagraphStyle
    pdfmetrics.registerFont(TTFont("CJK","/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"))
    pdfmetrics.registerFont(TTFont("Latin","/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    W,H=841.89,595.28;M=38;ink=HexColor("#213b32");muted=HexColor("#5e6d65");accent=HexColor("#145f56")
    path=args.output_dir/"research-summary.pdf";c=canvas.Canvas(str(path),pagesize=(W,H));c.setTitle("阶段性研究可视化报告 · 截至SEM329");c.setAuthor("ICSHM2026 Project 2")
    body=ParagraphStyle("body",fontName="CJK",fontSize=10.4,leading=17,textColor=ink,wordWrap="CJK")
    small=ParagraphStyle("small",parent=body,fontSize=8.5,leading=13,textColor=muted)
    def rich(t):
        # Droid's fallback font intentionally lacks Latin glyphs: embed and use
        # DejaVu for Latin, digits and symbols rather than silently drawing blanks.
        return "".join(f'<font name="{"Latin" if ord(run[0])<0x2e80 else "CJK"}">{html.escape(run)}</font>'
                       for run in re.findall(r"[^\u2e80-\uffff]+|[\u2e80-\uffff]+",t))
    def text(t,x,y,w,size="body"):
        p=Paragraph(rich(t),small if size=="small" else body);_,h=p.wrap(w,500);p.drawOn(c,x,y-h);return y-h
    def line(t,x,y,size=25,color=ink):
        style=ParagraphStyle("line",parent=body,fontSize=size,leading=size*1.15,textColor=color)
        p=Paragraph(rich(t),style);_,h=p.wrap(W-x-M,100);p.drawOn(c,x,y+size-h)
    def title(num,t,sub):
        c.setFillColor(HexColor("#f7f7f2"));c.rect(0,0,W,H,fill=1,stroke=0)
        line("ICSHM 2026 / PROJECT 2     RESEARCH REPORT     08 SEP 2026",M,H-30,9,accent)
        line(t,M,H-68);text(sub,M,H-85,W-2*M,"small")
        c.setStrokeColor(HexColor("#d9ddd5"));c.line(M,32,W-M,32);line("本地开发证据 · 完整指标、来源和视频见同目录 index.html",M,19,8,muted);line(f"{num:02d}/08",W-M-30,19,8,muted)
    def pic(rel,x,y,w,h):
        image=Image.open(args.output_dir/rel);iw,ih=image.size;scale=min(w/iw,h/ih);nw,nh=iw*scale,ih*scale
        c.drawImage(ImageReader(image),x+(w-nw)/2,y+(h-nh)/2,width=nw,height=nh)
    def cap(t,x,y,w):text(t,x,y,w,"small")
    def legend(x,y):
        for i,(name,label,color) in enumerate(zip(CLASSES,LABELS,PALETTE)):
            xx=x+i*151;c.setFillColor(Color(*[int(v)/255 for v in color]));c.rect(xx,y-8,9,9,fill=1,stroke=0)
            text(f"{i} {label} / {name}",xx+14,y+4,135,"small")
    samples={s["id"]:s for s in d["samples"]};hero=samples["140"];s=d["score"]["summary"]
    title(1,"从重建到可验证的语义纠错","阶段性成果总览 · 研究截至SEM329 · 适合10–15分钟组内／导师讨论")
    pic(hero["assets"]["rgb"],M,234,375,234);pic(hero["assets"]["overlay"],W-M-375,234,375,234)
    cap("140 · 冻结Scaffold RGB重建",M,231,375);cap("140 · SEM329叠加（五类固定颜色，alpha=0.55）",W-M-375,231,375)
    for x,value,label in [(M,"83.705%","SEM329 · 当前三臂mIoU最高"),(M+266,"+0.318 pp","相对正式SEM161的mIoU增量"),(M+522,"89.762%","基础recall · 仍低于SEM161")]:
        line(value,x,177,29,accent);text(label,x,161,245)
    text("正式可搬迁方案为 RGB-030-FULL + SEM161（mIoU 83.387017%）；SEM329是优先研究候选，提交包尚未集成。它提升基础IoU与precision，但相对SEM161基础recall下降1.424762个百分点。",M,122,W-2*M)
    text(d["protocol"],M,65,W-2*M,"small");c.showPage()
    title(2,"任务、数据与最终推理链","从官方相机出发，得到新视角RGB和五类语义；二维专家在冻结重建结果上工作。")
    pic(hero["assets"]["reference"],M,258,365,205);pic(hero["assets"]["gt"],W-M-365,258,365,205)
    cap("140 · 真实参考图（PINHOLE，供本报告比较）",M,253,365);cap("140 · 官方标注生成的PINHOLE语义GT",W-M-365,253,365);legend(M,215)
    text("数据含400张RGB，其中300张带标签。语义划分：219训练、74开发、7隔离、100无标注。RGB共同主比较实际采用303训练／74验证；“400-RGB方案”不表示400张全部用于训练。",M,184,W-2*M)
    text("官方COLMAP相机 → 冻结Scaffold RGB + SEM001三维语义 → 渲染域二维证据 → SEM141拉索 → SEM151基础审核 → SEM161桥面恢复 → SEM326首源复核 → SEM329第二源否决",M,117,W-2*M)
    text("最新复核仅使用固定219训练参考源。报告静态对比为PINHOLE；新视角视频按官方SIMPLE_RADIAL畸变相机输出。目标真实RGB与GT不进入新视角推理。",M,65,W-2*M,"small");c.showPage()
    title(3,"为什么选择Scaffold作为RGB骨干","同一空间划分、74视角、30k迭代；PSNR和SSIM越高越好，LPIPS-Alex越低越好。")
    pic("assets/charts/rgb-backbones.png",M,268,W-2*M,218)
    sw=(W-2*M-24)/3
    for i,(key,label) in enumerate([("standard","Standard"),("mip","Mip-Splatting"),("rgb","Scaffold-GS")]):
        pic(hero["assets"][key],M+i*(sw+12),85,sw,170);cap("140 · "+label,M+i*(sw+12),82,sw)
    text("同条件骨干比较支持Scaffold。后续RGB101虽有更高全局PSNR，却损伤困难视角，未替换冻结骨干。数值优势与局部图像代价需要同时判断。",M,61,W-2*M,"small");c.showPage()
    title(4,"语义进展：主要收益来自哪里","从静态三维基底到渲染域与类别专家，再引入稠密参考源纠错。")
    pic("assets/charts/semantic-progress.png",M,240,W-2*M,245)
    items=[("SEM001 → SEM111","渲染域二维证据与结构保护融合，显著减少静态索误报。"),("SEM141 → SEM151 → SEM161","原生索分支、基础单向审核、桥面单向恢复，组成正式可复现配方。"),("SEM326 → SEM329","首源删除错误基础；第二可靠源支持基础时恢复部分旧删除。")]
    for i,(head,desc) in enumerate(items):
        y=222-i*54;line(head,M,y,12,accent);text(desc,M+230,y+5,W-2*M-230)
    text("历史主线同属74开发图，但训练配方与seed存在差异；它是组合演进记录，不是贯穿全程的单变量消融。只有最新三臂使用相同封存评分口径作配对比较。",M,64,W-2*M,"small");c.showPage()
    title(5,"尝试地图：收益、代价与停止原因","包括未保留方案、来源阻塞与正面局部消融。完整实验记录可在网页逐项展开。")
    y=471
    for i,r in enumerate(d["routes"]):
        x=M+(i%3)*262;yy=y-(i//3)*137
        line(r["name"],x,yy,13,accent)
        text(r["attempt"],x,yy-15,243,"small");text(r["result"],x,yy-57,243,"small")
    text("判断原则：匹配控制上的改善、超过冻结正式方案、工程可运行、隐藏集泛化是不同层面的证据。DINOv3权限阻塞不代表其方法质量失败。",M,63,W-2*M,"small");c.showPage()
    title(6,"最新三臂：更高IoU，也有召回代价","相同74张开发图：SEM161正式基线、SEM326单参考源、SEM329第二源否决。")
    pic("assets/charts/latest-comparison.png",M,245,W-2*M,241)
    headers=["方案","mIoU (%)","基础IoU (%)","precision (%)","recall (%)"]
    xs=[M,M+182,M+324,M+478,M+640]
    for x,h in zip(xs,headers):text(h,x,233,142,"small")
    for j,(arm,label) in enumerate([( "baseline","SEM161 正式"),("dense","SEM326 首源"),("candidate","SEM329 第二源")]):
        m=s[arm];f=m["per_class"]["foundation"];vals=[label,f'{100*m["mIoU_with_background"]:.6f}',f'{100*f["iou"]:.6f}',f'{100*f["precision"]:.6f}',f'{100*f["recall"]:.6f}']
        for x,v in zip(xs,vals):text(v,x,207-j*29,142)
    text("SEM329相对SEM161：mIoU +0.317985pp，基础IoU +1.584083pp，基础recall -1.424762pp。相对SEM326再救回739个真基础像素，同时返回672个背景像素；净正确像素仅增加67。",M,104,W-2*M)
    text("相对SEM161：7590个完整纠正、2383个真基础误删、80个错误类别互换。错误变为另一种错误，不计作完整纠正。",M,59,W-2*M,"small");c.showPage()
    title(7,"看清局部：最好结果与困难样例","原生裁剪采用相同坐标；这里放大展示基础细节，网页同时保留全图与可切换版本。")
    sample=samples["183"];keys=["gt_crop","sem161_crop","sem326_crop","sem329_crop"]
    for i,(key,label) in enumerate(zip(keys,["GT","SEM161","SEM326","SEM329"])):
        x=M+i*194;pic(sample["assets"][key],x,288,182,173);cap("183 · "+label,x,282,182)
    text(NOTES["183"],M,251,W-2*M)
    pic("assets/charts/per-view.png",M,66,W-2*M,159)
    text("补充坏例：145为相对SEM326最大退化，088为相对SEM161最大退化；061属于固定最差8张，第二源无新增收益。seg-010基础recall仍为83.635%。",M,60,W-2*M,"small");c.showPage()
    title(8,"动态结果、交付方式与下一步","144个真实新视角 · 24fps · 6秒 · 三方案共用同一相机轨迹；无目标GT的可视化不产生mIoU。")
    if d["videos"]:
        for i,v in enumerate(d["videos"]):
            x=M+(i%2)*392;y=283-(i//2)*165;pic(v["poster"],x,y,372,139);cap(v["title"],x,y-4,372)
    text("打开同目录index.html即可离线播放、暂停、定位或以0.5倍速检查。高清图片含原生PNG及同坐标局部；网页提供完整实验来源和历史视频。",M,99,W-2*M)
    text("下一步：在训练侧区分第二源覆盖不足与标签冲突；方法选定后另做全量拟合与提交包复现。当前保留SEM329研究起点、SEM326精度折中与SEM161正式回退。",M,58,W-2*M,"small");c.showPage()
    c.save()


def package(args):
    out=args.output_dir
    files={str(p.relative_to(out)):{"sha256":digest(p),"bytes":p.stat().st_size} for p in sorted(out.rglob("*")) if p.is_file() and p.name!="package-manifest.json"}
    write_json(out/"package-manifest.json",{"schema_version":1,"files":files,"scope":"Local presentation assets only; no checkpoints, full dataset, caches or secrets."})
    archive=out.parent/(out.name+".zip")
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():z.write(p,Path(out.name)/p.relative_to(out))
    print(json.dumps({"report":str(out),"archive":str(archive),"files":len(files)+1,"archive_bytes":archive.stat().st_size}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root",type=Path,default=ROOT)
    for key in ("artifact-root","data-root","output-dir","plot-python"):p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--video-dir",type=Path)
    p.add_argument("--phase",choices=("prepare","final"),default="final");args=p.parse_args()
    args.project_root=args.project_root.resolve();args.artifact_root=args.artifact_root.resolve();args.data_root=args.data_root.resolve();args.output_dir=args.output_dir.resolve();args.video_dir=(args.video_dir or args.project_root/"outputs/vis006/full-20260908-r1").resolve()
    d=gather(args)
    subprocess.run([str(args.plot_python),str(args.project_root/"scripts/plot_research_report.py"),"--content",str(args.output_dir/"data/content.json"),"--output-dir",str(args.output_dir/"assets/charts")],check=True,env=dict(os.environ,MPLCONFIGDIR=str(args.output_dir.parent/"report_plot_cache")))
    make_html(args,d)
    if args.phase=="final":make_pdf(args,d);package(args)


if __name__=="__main__":main()
