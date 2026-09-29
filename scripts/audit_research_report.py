#!/usr/bin/env python3
"""Read-only checks of the report, native samples, decoded videos and freeze."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
from pypdf import PdfReader


def read(p):return json.loads(p.read_text(encoding="utf-8"))
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(2**20),b""):h.update(block)
    return h.hexdigest()
def pixels(p):
    with Image.open(p) as im:return np.asarray(im).copy()
def require(ok,message):
    if not ok:raise ValueError(message)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("report","project-root","artifact-root","video-dir","output"):p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args();d=read(a.report/"data/content.json");manifest=read(a.report/"package-manifest.json")
    for rel,r in manifest["files"].items():
        require(sha(a.report/rel)==r["sha256"],"Packaged file differs: "+rel)
    require(d["class_names"]==["background","deck","stay_cable","tower","foundation"],"Class name mismatch")
    pdf=PdfReader(a.report/"research-summary.pdf");require(len(pdf.pages)==8,"PDF must have8 pages")
    alltext="\n".join(page.extract_text() for page in pdf.pages)
    for value in ("83.387017","83.705003","1.424762","stay_cable","SEM329","144"):
        require(value in alltext,"PDF text missing: "+value)
    palette=np.array(d["palette"],dtype=np.uint8);samples=[]
    for sample in d["samples"]:
        im={key:pixels(a.report/rel) for key,rel in sample["assets"].items()}
        labels={key:pixels(a.report/"assets/images"/sample["id"]/(key+"-ids.png")) for key in ["gt","sem161","sem326","sem329"]}
        for key,lab in labels.items():require(np.array_equal(palette[lab],im[key]),"Semantic palette differs")
        for key in ["reference","rgb","standard","mip","gt","sem161","sem326","sem329"]:require(im[key].shape==(989,1320,3),"Pixel space differs")
        x0,y0,x1,y1=sample["crop_xyxy"]
        for key in ["rgb","gt","sem161","sem326","sem329","restored"]:
            require(np.array_equal(im[key][y0:y1,x0:x1],im[key+"_crop"]),"Crop coordinates differ")
        for k in [1,2,3]:
            for arm in ["sem326","sem329"]:require(np.array_equal(labels["sem161"]==k,labels[arm]==k),"Protected class differs")
        samples.append(sample["id"])
    require(samples==["054","081","140","159","160","183","180","145","088","061"],"Sample selection differs")
    video_results=[]
    for video in d["videos"]:
        src=a.report/video["path"];require(sha(src)==video["sha256"],"MP4 differs")
        info=read_probe(src);stream=info["streams"][0]
        require(int(stream["nb_read_frames"])==144 and stream["r_frame_rate"]=="24/1" and float(info["format"]["duration"])==6,"Decoded MP4 duration or frame count differs")
        # Decode the first, middle and final encoded frames to confirm their order
        # against the exact pre-codec presentation PNGs, allowing H.264 loss.
        raw=subprocess.check_output(["ffmpeg","-v","error","-i",str(src),"-vf","select='eq(n,0)+eq(n,72)+eq(n,143)'","-vsync","0","-f","rawvideo","-pix_fmt","rgb24","pipe:1"])
        frames=np.frombuffer(raw,np.uint8).reshape(3,video["height"],video["width"],3);errors=[]
        for frame,decoded in zip([0,72,143],frames):
            expected=pixels(a.video_dir/(src.stem+f"-frame{frame:04d}.png"))
            error=float(np.abs(expected.astype(np.int16)-decoded.astype(np.int16)).mean());require(error<5,"Encoded representative frame differs / out of order");errors.append(error)
        video_results.append({"path":video["path"],"decoded_frames":144,"duration_seconds":6,"representative_mean_absolute_errors":errors})
    freeze=read(a.project_root/"experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json")["artifacts"]
    for item in freeze:require(sha(a.artifact_root/item["path"])==item["sha256"],"Frozen artifact changed: "+item["role"])
    result={"status":"passed_report_integrity","files_verified":len(manifest["files"]),"pdf_pages":8,"native_samples":samples,"videos":video_results,"frozen_artifacts_verified":len(freeze),"class2":"stay_cable","new_quality_score":False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(result))


def read_probe(src):
    return json.loads(subprocess.check_output(["ffprobe","-v","error","-count_frames","-show_entries","stream=nb_read_frames,r_frame_rate,width,height","-show_entries","format=duration","-of","json",str(src)],text=True))


if __name__=="__main__":main()
