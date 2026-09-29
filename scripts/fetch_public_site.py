#!/usr/bin/env python3
"""Download the pinned public Release and verify it before Pages deployment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def extract_verified_archive(archive: Path, output: Path, expected_sha256: str) -> int:
    if output.exists():
        raise FileExistsError(output)
    if digest(archive) != expected_sha256:
        raise ValueError('Release archive SHA256 mismatch')
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive path')
        if sum(i.file_size for i in infos) > 1024**3:
            raise ValueError('Unexpected unpacked size')
        for i in infos:
            p = PurePosixPath(i.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in i.filename or ':' in i.filename or stat.S_ISLNK(i.external_attr >> 16):
                raise ValueError('Unsafe archive path')
        manifest = json.loads(z.read('package_manifest.json'))['files']
        actual = {i.filename for i in infos if not i.is_dir()} - {'package_manifest.json'}
        if actual != set(manifest):
            raise ValueError('Archive file set differs from manifest')
        for name, record in manifest.items():
            data = z.read(name)
            if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
                raise ValueError('Packaged file integrity mismatch: '+name)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Integrity and paths are fully checked before extracting anything.
        z.extractall(output)
    return len(manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path, help='Optional already downloaded archive')
    args = parser.parse_args()
    meta = json.loads((ROOT/'publication/release.json').read_text())
    with tempfile.TemporaryDirectory(prefix='icshm-public-site-') as directory:
        if args.archive:
            archive = args.archive
        else:
            subprocess.run(['gh','release','download',meta['release_tag'],'--repo',meta['repository'].removeprefix('https://github.com/'),
                            '--pattern',meta['asset'],'--dir',directory], check=True)
            archive = Path(directory)/meta['asset']
        count = extract_verified_archive(archive,args.output,meta['sha256'])
    print(json.dumps({'status':'verified','files':count,'output':str(args.output)}))


if __name__ == '__main__':
    main()
