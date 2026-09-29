#!/usr/bin/env python3
"""Independent integrity, portability and numerical audit for REPORT004."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
import numpy as np
from PIL import Image


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.ids=set()
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if 'id'in d:self.ids.add(d['id'])
        for k in ['src','href','poster']:
            if k in d:self.links.append((tag,k,d[k]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-root',type=Path);parser.add_argument('--decode-videos',action='store_true')
    a=parser.parse_args();root=a.package.resolve();checks=[];failures=[]
    def check(ok,label):
        checks.append(label)
        if not ok:failures.append(label)
    manifest=read(root/'package_manifest.json')['files'];actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='package_manifest.json'}
    check(actual==set(manifest),'manifest covers exact package file set')
    for rel,entry in manifest.items():
        p=root/rel;check(p.is_file() and p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],'file hash '+rel)
    index=read(root/'data/experiment_index.json');check(len(index)==351,'351 historical registry entries')
    check(len({e['experiment_id'] for e in index})==351,'unique experiment IDs')
    check(all(e['experiment_id']!='REPORT-004-PAPER-HANDOFF' for e in index),'handoff excluded from historical denominator')
    figs=read(root/'data/figures.json');videos=read(root/'data/videos.json')
    check([f['id'] for f in figs]==[f'F{i:02}' for i in range(1,11)],'ten figure themes')
    check(len(videos)==8 and len({v['id'][:3] for v in videos})==6,'eight videos in six themes')
    check(all(f['caption_zh'] and f['caption_en'] and f['refs'] and f['role'] for f in figs),'complete figure captions and roles')
    check(all(v['caption_zh'] and v['caption_en'] and v['legend'] and v['role'] for v in videos),'complete video captions and legends')
    guide=read(root/'data/author_guide.json')
    check(len(guide['sections'])==12,'twelve author understanding chapters')
    guide_md=(root/'author_guide.md').read_text();guide_html=(root/'author_guide.html').read_text()
    check(len(guide['glossary'])==16,'sixteen author glossary entries')
    check('author_guide.html' in (root/'index.html').read_text(),'author guide discoverable on main page')
    for s in guide['sections']:
        check(all(p in guide_md for p in s['paragraphs']),'complete author text '+s['id'])
        check(s['checkpoint']['answer'] in guide_md and s['checkpoint']['answer'] in guide_html,'self-check answer '+s['id'])
        check(bool(s['refs']) and bool(s['takeaway']),'author claim has evidence and takeaway '+s['id'])
    for p in root.rglob('*.html'):
        parsed=Links();parsed.feed(p.read_text())
        for tag,attr,link in parsed.links:
            url=urlsplit(link)
            if url.scheme in ['http','https','mailto']:
                check(tag=='a' and attr=='href','no remote page dependency '+str(p.relative_to(root)))
                continue
            target=(p.parent/unquote(url.path)).resolve() if url.path else p
            check(target==root or root in target.parents,'local link stays inside package '+link)
            check(target.exists(),'local link exists '+str(p.relative_to(root))+' '+link)
            if url.fragment and target.suffix=='.html':
                pp=Links();pp.feed(target.read_text());check(url.fragment in pp.ids,'anchor '+link)
    for p in root.rglob('*.md'):
        text=p.read_text()
        # Scientific snapshots can quote historical Markdown as literal text;
        # validate only actual links remaining after the builder's rewriting.
        for label,link in re.findall(r'(?<!!)\[([^\]]+)\]\(([^)]+)\)',text):
            url=urlsplit(link.strip('<>'))
            if url.scheme or not url.path:continue
            check((p.parent/unquote(url.path)).exists(),'Markdown target '+str(p.relative_to(root))+' '+link)
    palette=np.asarray(read(root/'data/build.json')['palette'],dtype=np.uint8)
    for asset in read(root/'data/asset_manifest.json')['assets']:
        p=root/asset['path']
        check(sha(p)==asset['sha256'],'asset digest '+asset['path'])
        if p.suffix.lower() in ['.png','.jpg']:
            with Image.open(p) as im:im.verify()
        if asset.get('semantic'):
            arr=np.asarray(Image.open(p));allowed=np.zeros(arr.shape[:2],bool)
            for color in palette:allowed|=np.all(arr[:,:,:3]==color,axis=2)
            check(bool(allowed.all()),'exact shared palette '+asset['path'])
            if a.source_root:
                src=a.source_root/asset['sources'][0]['path'];original=np.asarray(Image.open(src))
                expected=palette[original] if original.ndim==2 else original[:,:,:3]
                check(np.array_equal(expected,arr),'semantic pixels preserved '+asset['path'])
    for v in videos:
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,codec_tag_string,pix_fmt,nb_frames,width,height','-of','json',str(root/v['path'])],text=True))['streams'][0]
        check(info['codec_name']=='h264' and info['pix_fmt']=='yuv420p','portable video codec '+v['id'])
        check(int(info['nb_frames'])==v['frame_count'],'video frames '+v['id'])
        check(v['frame_count']==(100 if v['id']=='V05' else 144),'expected semantic/camera sequence '+v['id'])
        if a.decode_videos:
            result=subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(root/v['path']),'-f','null','-'],capture_output=True)
            check(result.returncode==0 and not result.stderr,'complete decode '+v['id'])
    data=read(root/'data/registry_snapshot.json');byid={e['experiment_id']:e for e in data}
    def experiment(prefix):return next(e for k,e in byid.items() if k==prefix or k.startswith(prefix+'-'))
    def packaged_result(src):return read(root/'evidence/results'/src.replace('/','__'))
    old=packaged_result('outputs/research_report_20260908/data/content.json')['history']
    delta=(old[4]['miou']-old[0]['miou'])*100
    check(abs(delta-21.776713102733807)<1e-9,'historical mIoU gain recomputed')
    for code in ['436','437']:
        d=packaged_result(f'outputs/quality/SEM-{code}/r2/inference/result.json')
        for step in ['6000','8000','10000','12000']:
            m=d['endpoints'][step]['metrics']; stored=experiment('SEM-'+code)['metrics'][f'training_miou_{step}']
            check(abs(m['mIoU_with_background']-stored)<1e-12,'native inference vs registry '+code+' '+step)
    for code in ['011','013']:
        d=packaged_result(f'outputs/quality/QUAL-{code}/r1/result.json')
        for arm in ['absgs','standard']:
            for region in ['holdout','region-1','region-2','region-3']:
                for metric in ['five_class_miou','foreground_miou','stay_cable_iou','boundary_f1']:
                    average=sum(x[arm][region][metric] for x in d['paired'].values())/2
                    check(abs(average-d['two_seed_means'][arm][region][metric])<1e-12,'two-seed arithmetic '+code+' '+arm+' '+region+' '+metric)
    cap=packaged_result('outputs/quality/QUAL-026/r1/result.json')['comparison_15M_minus_1M']
    check(abs(cap['holdout']['PSNR']-(-.8670078678309849))<1e-12,'capacity holdout delta source')
    page=(root/'index.html').read_text();check('尚待训练' in page and 'RGB166' in page,'pending RGB166 semantic status explicit')
    check('95.678649%' in page and '训练' in page and '非盲测' in page,'training-fit and development distinctions present')
    for p in root.rglob('*'):
        if p.is_file() and p.suffix in ['.json','.md','.yaml','.html','.txt','.js','.css']:
            check(not re.search(r'/(?:home|Users)/[^/\s"<>]+',p.read_text()),'no personal absolute paths '+str(p.relative_to(root)))
    if a.source_root:
        for rel,expected in read(root/'data/source_hashes.json').items():
            check((a.source_root/rel).is_file() and sha(a.source_root/rel)==expected,'source unchanged '+rel)
    result={'status':'passed' if not failures else 'failed','checks':len(checks),'failures':failures,'files':len(manifest),'full_video_decode':a.decode_videos,'source_hash_check':bool(a.source_root)}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))
    raise SystemExit(bool(failures))


if __name__=='__main__':main()
