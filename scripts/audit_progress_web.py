#!/usr/bin/env python3
"""Validate REPORT002 portability, source hashes, exact metric mappings and assets."""
import argparse
import json
from html.parser import HTMLParser
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        self.links.extend(v for k,v in attrs if k in ['href','src','poster'] and v)


def read(p):return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--result',type=Path,required=True);a=p.parse_args()
    root=a.report.resolve();checks={};manifest=read(root/'package_manifest.json')['files']
    for rel,item in manifest.items():assert sha256(root/rel)==item['sha256'],rel
    checks['package_files']=len(manifest)
    count=0
    for f in root.rglob('*.html'):
        parser=Links();parser.feed(f.read_text())
        assert '/home/' not in f.read_text(),str(f)
        for link in parser.links:
            url=urlsplit(link)
            if url.scheme=='data' or not url.path:continue
            assert not url.scheme and not url.netloc,link
            target=(f.parent/unquote(url.path)).resolve()
            assert target.is_relative_to(root) and target.is_file(),link
            count+=1
    checks['local_html_links']=count
    data=read(root/'data/content.json');assert len(data['stages'])==6
    def references(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k in ['video','poster','path','crop','link'] and isinstance(v,str):
                    assert (root/v).is_file() and (root/v).resolve().is_relative_to(root),v
                else:references(v)
        elif isinstance(value,list):
            for v in value:references(v)
    references(data)
    for item in read(root/'data/asset_manifest.json'):
        assert sha256(root/item['path'])==item['sha256']
        inputs=item.get('input_hashes',{})
        if 'source' in item:inputs[item['source']]=item['source_sha256']
        for s in item.get('sources',[]):inputs[s['path']]=s['sha256']
        for rel,h in inputs.items():
            source=a.artifact_root/rel.removeprefix('$ARTIFACT_ROOT/') if rel.startswith('$ARTIFACT_ROOT/') else ROOT/rel
            assert sha256(source)==h,rel
    checks['source_assets_exact']=True
    b2=read(root/'data/score-r2-receipt.json')['metrics']
    assert data['history'][2]['miou']==b2['mIoU_with_background']
    assert abs(data['history'][2]['miou']-.8091772380534016)>.01
    for row in data['history']+data['full']:
        assert 0<row['miou']<1
        assert all(0<=row[k]<=1 for k in ['deck','stay_cable','tower','foundation'])
    registry=read(ROOT/'experiments/registry.yaml')['experiments']
    for pre,index in [('SEM-378',1),('SEM-380',2),('SEM-381',3),('SEM-382',4),('SEM-383',5)]:
        r=next(r for r in registry if r['experiment_id'].startswith(pre+'-'));m=r['metrics']
        assert data['full'][index]['miou']==m['mIoU_with_background']
        for k in ['deck','stay_cable','tower','foundation']:assert data['full'][index][k]==m['per_class_iou'][k]
    checks['raw_b2_and_full_metric_identity']=True
    assert [v['name'] for v in data['views']]==['159.png','160.png','146.png','057.png']
    for stage in ['render','b2-r2','m2f']:
        receipt=read(root/'data'/f'{stage}-receipt.json');assert receipt['status']=='completed' and receipt['trained_steps']==0 and receipt['frozen18_exact']
    checks['frozen_no_training_receipts']=True
    checks['difficult_views_and_dynamic_paths']=True
    result=dict(status='passed',checks=checks)
    a.result.parent.mkdir(parents=True,exist_ok=True);a.result.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
