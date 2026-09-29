#!/usr/bin/env python3
"""Export an audited public snapshot; never push, train, infer or rescore."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote, urlsplit
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'outputs/github_public_20260929'
REPOSITORY = WORK / 'repository'
SITE = WORK / 'site'
HANDOFF = ROOT / 'outputs/paper_handoff_20260929'
QA = ROOT / 'outputs/github_public_20260929_qa'
REPO = 'https://github.com/samtu648-hub/icshm2026-semantic-gs'
SITE_URL = 'https://samtu648-hub.github.io/icshm2026-semantic-gs'
TAG = 'paper-handoff-20260929-v1'
ASSET = 'paper-handoff-20260929-public.zip'
LINK = re.compile(r'(!?)\[([^\]]+)\]\(([^)]+)\)')
SOURCE_MAP: dict[str, dict] = {}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def clean(text: str) -> str:
    text = text.replace(str(ROOT), '$PROJECT_ROOT')
    text = re.sub(r'/(?:home|Users)/[^/\s"<>`]+/outputs/icshm2026', '$ICSHM_OUTPUT_ROOT', text)
    return re.sub(r"/(?:home|Users)/[^/\s\"'<>`]+", '$USER_HOME', text)


def write(relative: str, text: str, sources: list[Path], processing: str) -> None:
    p = REPOSITORY / relative
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    SOURCE_MAP[relative] = {'sources': [{'path': str(s.relative_to(ROOT)), 'sha256': sha(s)} for s in sources], 'processing': processing}


def template(name: str) -> str:
    return (ROOT/'scripts/report_assets'/name).read_text().replace('@@REPO@@', REPO).replace('@@SITE@@', SITE_URL).replace('@@TAG@@', TAG)


def portable_python(text: str) -> tuple[str, bool]:
    pattern = re.compile(r'Path\(([\"\'])(/(?:home|Users)/[^\"\']+/outputs/icshm2026)\1\)')
    if not pattern.search(text):
        return text, False
    lines = []
    for line in text.splitlines(keepends=True):
        replacement = 'parse_artifact_root(__doc__)' if re.search(r'^\s*artifact_root\s*=\s*Path', line) else 'artifact_root_default()'
        lines.append(pattern.sub(replacement, line))
    text = ''.join(lines)
    tree = ast.parse(text)
    insertion = 0
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            insertion = node.end_lineno
        elif isinstance(node, ast.ImportFrom) and node.module == '__future__':
            insertion = node.end_lineno
        else:
            break
    lines = text.splitlines(keepends=True)
    lines.insert(insertion, '\nfrom _artifact_paths import artifact_root_default, parse_artifact_root\n')
    result = ''.join(lines)
    ast.parse(result)
    return result, True


def refresh_manifest(repo: Path = REPOSITORY) -> None:
    path = repo/'publication/source_manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text())
    else:
        manifest = {'schema_version': 1, 'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    'source_working_tree': 'uncommitted research state, not identical to source_commit',
                    'historical_results_cutoff': '2026-09-23', 'publication_record': 'REPORT-005-GITHUB-PUBLIC-RELEASE', 'files': {}}
    records = {}
    for p in sorted(repo.rglob('*')):
        rel = p.relative_to(repo).as_posix()
        if not p.is_file() or '.git' in p.relative_to(repo).parts or '__pycache__' in p.parts or rel == 'publication/source_manifest.json':
            continue
        old = SOURCE_MAP.get(rel, manifest['files'].get(rel, {}))
        processing = old.get('processing', 'publication metadata or generated verification record')
        if old.get('public_sha256') and old['public_sha256'] != sha(p):
            processing += '; publication-only revision after initial export'
        records[rel] = {'sources': old.get('sources', []), 'processing': processing,
                        'public_sha256': sha(p), 'bytes': p.stat().st_size}
    manifest['files'] = records
    save(path, manifest)


def package_site() -> dict:
    assets = json.loads((SITE/'data/asset_manifest.json').read_text())
    for entry in assets['assets']:
        p = SITE/entry['path']
        new_hash = sha(p)
        if new_hash != entry['sha256']:
            entry['previous_handoff_sha256'] = entry['sha256']
            entry['transform'] += '; public navigation/text sanitation only'
            entry['sha256'] = new_hash
            entry['bytes'] = p.stat().st_size
    save(SITE/'data/asset_manifest.json', assets)
    files = {p.relative_to(SITE).as_posix(): {'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(SITE.rglob('*')) if p.is_file() and p.name != 'package_manifest.json'}
    save(SITE/'package_manifest.json', {'schema_version':1,'experiment_id':'REPORT-005-GITHUB-PUBLIC-RELEASE','files':files})
    archive = WORK/ASSET
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(SITE.rglob('*')):
            if p.is_file(): z.write(p, p.relative_to(SITE))
    result = {'repository':REPO,'site_url':SITE_URL,'release_tag':TAG,'asset':ASSET,'sha256':sha(archive),'bytes':archive.stat().st_size,
              'site_files':len(files),'evidence_cutoff':'2026-09-23','media_source':'REPORT004 author-guide v2','training':False,'inference':False}
    save(REPOSITORY/'publication/release.json',result)
    (WORK/(ASSET+'.sha256')).write_text(result['sha256']+'  '+ASSET+'\n')
    return result


def create_workflow(actions: dict) -> None:
    action = lambda name: 'actions/'+name+'@'+actions[name]['sha']
    content = '''name: Publish research visualization
on:
  release:
    types: [published]
  workflow_dispatch:
permissions:
  contents: read
concurrency:
  group: github-pages
  cancel-in-progress: false
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: CHECKOUT
      - name: Download the pinned release asset and verify every file
        env:
          GH_TOKEN: ${{ github.token }}
        run: python3 scripts/fetch_public_site.py --output _site
      - uses: CONFIGURE
      - uses: UPLOAD
        with:
          path: _site
  deploy:
    needs: build
    runs-on: ubuntu-latest
    permissions:
      pages: write
      id-token: write
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy verified artifact
        id: deployment
        uses: DEPLOY
'''.replace('CHECKOUT',action('checkout')).replace('CONFIGURE',action('configure-pages')).replace('UPLOAD',action('upload-pages-artifact')).replace('DEPLOY',action('deploy-pages'))
    write('.github/workflows/publish-pages.yml', content, [ROOT/'scripts/prepare_public_release.py'], 'official Pages workflow; action commits pinned')


def main() -> None:
    if (REPOSITORY/'.git').exists():
        raise SystemExit('Refusing to overwrite an initialized publication checkout; use a new staging directory or targeted edits.')
    REPOSITORY.mkdir(parents=True,exist_ok=True)
    SITE.mkdir(parents=True,exist_ok=True)
    shutil.copytree(HANDOFF,SITE,dirs_exist_ok=True)
    # Historical receipts remain explicitly historical; this publication needs its own checks.
    shutil.copyfile(SITE/'data/verification.json',SITE/'data/handoff_verification.json')
    save(SITE/'data/verification.json',{'publication_record':'REPORT-005-GITHUB-PUBLIC-RELEASE','status':'not_yet_audited'})
    save(SITE/'data/publication.json',{'repository':REPO,'release':REPO+'/releases/tag/'+TAG,'site_url':SITE_URL,
        'scientific_evidence_cutoff':'2026-09-23','teammate_commit':'dceb0e67faa6b4c0fd6fba754f15c713c5cbf721',
        'scope':'Selected images and videos approved for public release; no raw dataset or model weights.'})
    for filename in ['index.html','author_guide.html']:
        p=SITE/filename;text=p.read_text()
        text=text.replace('<nav aria-label=',f'<a href="{REPO}">GitHub 项目与源码</a><nav aria-label=',1)
        text=text.replace('REPORT-004-PAPER-HANDOFF · 本地离线材料 · 无训练／新推理／上传','REPORT-005 · 公开写作资料 · 无新训练或推理')
        p.write_text(text)
    # Map packaged evidence back to source paths for useful historical links.
    links={}
    for asset in json.loads((HANDOFF/'data/asset_manifest.json').read_text())['assets']:
        for source in asset.get('sources',[]):links.setdefault(source['path'],SITE_URL+'/'+asset['path'])
    names=[n for n in subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT,text=True).split('\0') if n]
    excluded=[];patched=[]
    for name in sorted(set(names)):
        src=ROOT/name
        if not src.is_file():continue
        if src.is_symlink():raise RuntimeError('Unexpected source symlink '+name)
        if name.endswith(('.doc','.docx','.pdf')):
            excluded.append({'path':name,'reason':'third-party supplied document; source notice retained'});continue
        if name=='.env' or name.startswith(('.git/','outputs/')) and name!='outputs/README.md':
            raise RuntimeError('Unexpected private/output candidate '+name)
        try:text=src.read_text()
        except UnicodeError:raise RuntimeError('Unreviewed binary candidate '+name)
        changed=False
        if name.endswith('.py'):text,changed=portable_python(text)
        if changed:patched.append(name)
        cleaned=clean(text)
        write(name,cleaned,[src],'artifact-root CLI/env portability; text paths sanitized' if changed else ('personal path sanitation' if cleaned!=text else 'byte-equivalent text copy'))
    for target,source in [('README.md','public_readme.md.in'),('docs/paper/team_comparison.md','public_team_comparison.md.in'),('docs/public_reproduction.md','public_reproduction.md.in')]:
        write(target,template(source),[ROOT/'scripts/report_assets'/source],'public reader-oriented documentation from curated template')
    write('scripts/_artifact_paths.py',(ROOT/'scripts/report_assets/public_artifact_paths.py').read_text(),[ROOT/'scripts/report_assets/public_artifact_paths.py'],'portable path helper')
    ignore=REPOSITORY/'.gitignore'
    ignore.write_text(ignore.read_text()+'\n# Local Pages extraction cache\n/_site/\n')
    SOURCE_MAP['.gitignore']['processing']+='; ignore local Pages extraction cache'
    optional_tests=[('tests/test_sem387_protocol.py','test_parent_plan_normalization_does_not_change_trace','outputs/quality/SEM-384/r1/sampling-plan.json'),
        ('tests/test_spatial_confirmation_baseline.py','test_prepared_scene_is_training_only','outputs/quality/RGB-152/scene/preparation_manifest.json'),
        ('tests/test_spatial_confirmation_split.py','test_registered_roles_are_complete_and_disjoint','outputs/quality/SEM-395/r2/prepare/confirmation-split-candidate.json')]
    for filename,method,artifact in optional_tests:
        p=REPOSITORY/filename
        marker='    def '+method+'(self):'
        replacement=f'    @unittest.skipUnless((ROOT / "{artifact}").is_file(), "optional saved run artifact is not distributed")\n'+marker
        text=p.read_text();assert marker in text
        p.write_text(text.replace(marker,replacement,1))
        SOURCE_MAP[filename]['processing']+='; missing non-distributed run artifact explicitly skips the relevant record check'
    write('NOTICE.md','# 版权与第三方来源\n\n本仓库公开项目研究代码与经确认的精选结果，供阅读、核对和协作。公开可访问不代表全部内容被统一授予MIT、Apache或其它新的许可。未标注独立许可的原创部分，本次未额外授予通用再分发许可；如需授权，请通过仓库Issues联系维护者。\n\n第三方算法、源码补丁、预训练权重及数据分别受原许可约束，参见[第三方来源](third_party/README.md)。保留相应版权与许可说明，不能用本仓库说明覆盖它们。\n\n本仓库不随源码分发比赛原始数据、标注、模型权重或官方Word模板。Release和在线页面中的图、视频为已确认可公开的精选展示材料；其它资产不因本次公开而获得新的授权。\n',[], 'publication rights boundary; no invented blanket license')
    write('docs/paper/README.md','# 论文写作入口\n\n先读[作者必读](author_guide.md)，再看[论文底稿](paper_notes.md)、[结果表](results_summary.md)、[中英图注](captions.md)与[两方案对照](team_comparison.md)。\n\n官方Word模板由团队在竞赛渠道获取，本次不再分发；模板原名称和摘要保留于历史记录。\n',[], 'public document entry; external template omitted')
    for p in (HANDOFF/'figures').iterdir():
        if p.is_file():
            dest=REPOSITORY/'docs/paper/figures'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
            SOURCE_MAP[dest.relative_to(REPOSITORY).as_posix()]={'sources':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)}],'processing':'byte-exact selected scientific figure'}
    for filename in ['author_guide.md','development_history.md','paper_notes.md','results_summary.md','captions.md','source_notes.md']:
        source=HANDOFF/filename
        def handoff_link(match):
            image,label,target=match.groups();url=urlsplit(target)
            if url.scheme:return match.group(0)
            if target.startswith('figures/'):return match.group(0)
            if url.path.startswith('evidence/reports/'):
                eid=Path(url.path).stem;dest=REPOSITORY/'experiments/runs'/eid/'report.md'
                if dest.exists():return f'{image}[{label}](../../experiments/runs/{eid}/report.md)'
            if url.path in ['author_guide.md','development_history.md','paper_notes.md','results_summary.md','captions.md','source_notes.md']:
                return match.group(0)
            return f'{image}[{label}]({SITE_URL}/{target})'
        write('docs/paper/'+filename,LINK.sub(handoff_link,source.read_text()),[source],'GitHub-readable text; figures local, media and packaged evidence linked online')
    # Resolve local Markdown in the snapshot only; record unavailable runtime assets honestly.
    for p in REPOSITORY.rglob('*.md'):
        relative=p.relative_to(REPOSITORY).as_posix()
        def local_link(match):
            image,label,target=match.groups();url=urlsplit(target.strip('<>'))
            if url.scheme or not url.path:return match.group(0)
            resolved=(p.parent/unquote(url.path)).resolve()
            if resolved.exists():return match.group(0)
            try:source_rel=resolved.relative_to(REPOSITORY).as_posix()
            except ValueError:source_rel=''
            if source_rel.startswith('outputs/paper_handoff_20260929/'):
                return f'{image}[{label}]({SITE_URL}/{source_rel.split("outputs/paper_handoff_20260929/",1)[1]}'+('#'+url.fragment if url.fragment else '')+')'
            if source_rel in links:return f'{image}[{label}]({links[source_rel]})'
            return f'{label}（来源记录：`{clean(target)}`；未随公开包提供）'
        text=p.read_text();revised=LINK.sub(local_link,text)
        if revised!=text:
            p.write_text(revised);SOURCE_MAP[relative]['processing']+='; portable links or explicit missing-runtime annotations'
    save(REPOSITORY/'publication/export_notes.json',{'excluded':excluded,'artifact_root_scripts':patched,
        'original_repository_history_published':False,'raw_data_published':False,'selected_media_authorized':True})
    create_workflow(json.loads((QA/'actions.json').read_text()))
    release=package_site()
    refresh_manifest()
    save(QA/'preparation.json',release|{'code_files':len(SOURCE_MAP),'patched_scripts':len(patched),'excluded_files':len(excluded)})
    print(json.dumps(release|{'code_files':len(SOURCE_MAP),'patched_scripts':len(patched)},ensure_ascii=False))


if __name__=='__main__':main()
