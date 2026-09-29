#!/usr/bin/env python3
"""Export static, source-labelled scientific plots from the report content manifest."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#7c8993", "#5294ad", "#145f56"]


def main():
    p=argparse.ArgumentParser();p.add_argument("--content",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args();d=json.loads(args.content.read_text());args.output_dir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,"axes.spines.right":False,
                         "axes.edgecolor":"#c9cfcb","text.color":"#21342f","axes.labelcolor":"#21342f","figure.facecolor":"#ffffff",
                         "savefig.facecolor":"#ffffff","svg.fonttype":"none"})
    def save(fig,name,caption):
        fig.text(.015,.015,caption,fontsize=8,color="#596760")
        fig.tight_layout(rect=(0,.06,1,1))
        for ext in ("svg","png"):fig.savefig(args.output_dir/(name+"."+ext),dpi=190)
        plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(11.8,3.5))
    for ax,key,label,lim in zip(axs,["PSNR_dB","SSIM","LPIPS_Alex"],["PSNR (dB), higher is better","SSIM (0-1), higher is better","LPIPS-Alex (0-1), lower is better"],[(0,25),(0,1),(0,.6)]):
        vals=[r["metrics"][key]["mean"] for r in d["rgb"]];bars=ax.bar([r["short"] for r in d["rgb"]],vals,color=COLORS,width=.57)
        ax.set(title=label,ylim=lim,xlabel="RGB backbone",ylabel=label.split(",")[0]);ax.bar_label(bars,fmt="%.3f",padding=4,fontsize=10)
    save(fig,"rgb-backbones","Source: RGB-011-R2-ALLOC / RGB-020-FULL / RGB-030-FULL; rgb_r1_spatial_v1, 74 views; training seed 0.")
    fig,ax=plt.subplots(figsize=(11.8,3.7));h=d["history"];x=np.arange(len(h));ys=[r["miou"]*100 for r in h]
    ax.plot(x,ys,"o-",color=COLORS[2],linewidth=2,markersize=7)
    for xx,yy in zip(x,ys):ax.annotate(f"{yy:.3f}%",(xx,yy),xytext=(0,12),textcoords="offset points",ha="center")
    ax.set(xticks=x,xticklabels=[r["short"] for r in h],ylim=(55,91),ylabel="Five-class mIoU including background (%)",xlabel="Historical combined recipe (not a single-variable ablation)",title="Semantic development: frozen 3D labels to source-supported review")
    ax.grid(axis="y",alpha=.2)
    save(fig,"semantic-progress","Source: experiment registry + SEM329 sealed score; same74 development views; recipes and seeds differ across milestones.")
    fig,axs=plt.subplots(1,2,figsize=(11.8,3.9),gridspec_kw={"width_ratios":[1.8,1]})
    labels=d["class_names"];x=np.arange(5)
    for i,(key,name) in enumerate(zip(["baseline","dense","candidate"],["SEM161","SEM326","SEM329"])):
        axs[0].bar(x+(i-1)*.22,[d["score"]["summary"][key]["per_class"][c]["iou"]*100 for c in labels],width=.22,label=name,color=COLORS[i])
    axs[0].set(xticks=x,xticklabels=labels,ylim=(0,105),ylabel="IoU (%)",xlabel="Class",title="Three fixed arms: all five classes");axs[0].tick_params(axis="x",labelsize=9);axs[0].legend(ncol=3,fontsize=9)
    for i,(key,name) in enumerate(zip(["baseline","dense","candidate"],["SEM161","SEM326","SEM329"])):
        m=d["score"]["summary"][key]["per_class"]["foundation"];axs[1].scatter(m["recall"]*100,m["precision"]*100,s=65,color=COLORS[i])
        delta=[(5,-14),(5,8),(5,-15)][i];axs[1].annotate(name,(m["recall"]*100,m["precision"]*100),xytext=delta,textcoords="offset points",fontsize=10)
    axs[1].set(xlim=(88.8,92),ylim=(77.5,82.7),xlabel="Foundation recall (%)",ylabel="Foundation precision (%)",title="Precision / recall tradeoff");axs[1].grid(alpha=.18)
    save(fig,"latest-comparison","Source: SEM329 developer-score-20260908-r1; same74 PINHOLE views, 1320 x 989; seed 20260805; reused development data.")
    fig,ax=plt.subplots(figsize=(11.8,3.8));names=sorted(d["score"]["per_view"]);x=np.arange(len(names))
    for key,name,color in [("baseline","SEM329 - SEM161",COLORS[2]),("dense","SEM329 - SEM326",COLORS[1])]:
        y=[100*(d["score"]["per_view"][n]["candidate"]-d["score"]["per_view"][n][key]) for n in names]
        ax.plot(x,y,".-",label=name,color=color,linewidth=1,markersize=4)
    ax.axhline(0,color="#5a6561",linewidth=.7);ax.set(xticks=x[::5],xticklabels=[n[:-4] for n in names[::5]],xlabel="Development camera / image ID (ordered)",ylabel="Per-view mIoU difference (percentage points)",title="Aggregate improvement is not uniform across views");ax.legend(loc="upper left",ncol=2,fontsize=9)
    save(fig,"per-view","Source: SEM329 sealed per_view results, 74 historically reused views; no re-ranking of fixed challenge/worst subsets.")
    fig,ax=plt.subplots(figsize=(11.8,3.5));classes=["background","deck","stay_cable","tower","foundation"]
    rows=d["score"]["summary"]
    names=["Corrected background","Harmed foundation","Still wrong (other class)"];values=[rows["remaining_deletions_by_gt_class"][0],rows["remaining_deletions_by_gt_class"][4],sum(rows["remaining_deletions_by_gt_class"][1:4])]
    bars=ax.barh(names,values,color=["#247c62","#b85b4f","#c89c42"]);ax.bar_label(bars,padding=5,fmt="%d");ax.invert_yaxis();ax.set(xlim=(0,max(values)*1.15),xlabel="Changed pixels across the same74 views (count)",ylabel="Effect relative to SEM161",title="SEM329: benefits and costs of its 10,053 changed pixels")
    save(fig,"pixel-effects","Source: SEM329 sealed change accounting. Net correct pixels +5,207 vs SEM161; +67 vs SEM326; wrong-to-wrong is not corrected.")


if __name__=="__main__":main()
