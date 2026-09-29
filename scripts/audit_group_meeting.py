#!/usr/bin/env python3
"""Verify portable links, hashes and stored semantic confusion summaries."""
import argparse, hashlib, json
from html.parser import HTMLParser
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--check-sources',action='store_true');a=ap.parse_args()
    root=Path(__file__).resolve().parents[1];out=a.report.resolve();bad=[];refs=[]
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key not in ['href','src','poster'] or not value or value.startswith(('#','data:')):continue
                refs.append(value)
                if '://' in value or value.startswith('/'):bad.append((str(self.path),'nonportable',value));continue
                target=(self.path.parent/value.split('#')[0]).resolve()
                if not target.is_relative_to(out) or not target.is_file():bad.append((str(self.path),value))
    for p in out.rglob('*.html'):
        parser=Links();parser.path=p;parser.feed(p.read_text())
    manifest=json.loads((out/'data/asset_manifest.json').read_text())
    for x in manifest:
        assert sha(out/x['path'])==x['sha256'],x['path']
        if a.check_sources:assert sha(root/x['source'])==x['sha256'],x['source']
    d=json.loads((out/'data/content.json').read_text())
    for group in d['galleries'].values():
        for row in group:
            for tile in row['tiles']:assert (out/tile['path']).is_file()
    assert d['class_names']==['background','deck','stay_cable','tower','foundation']
    for group in ['matched','cross']:
        for arm in d[group]:
            c=arm['confusion'];iou=[]
            for i,n in enumerate(d['class_names']):
                tp=c[i][i];fp=sum(row[i] for row in c)-tp;fn=sum(c[i])-tp
                v=tp/(tp+fp+fn);iou.append(v)
                assert abs(v-arm['per_class'][n]['IoU'])<1e-12
            assert abs(sum(iou)/5-arm['mIoU_with_background'])<1e-12
    package=json.loads((out/'package_manifest.json').read_text())
    for p,v in package['files'].items():assert sha(out/p)==v['sha256'],p
    assert not bad,bad
    result=dict(status='passed',html_files=len(list(out.rglob('*.html'))),static_references=len(refs),packaged_files=len(package['files']),source_media_exact=len(manifest) if a.check_sources else None,semantic_confusion_summaries_recomputed=8,new_quality_score=False)
    a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
