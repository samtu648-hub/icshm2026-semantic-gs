from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from scripts.fetch_public_site import extract_verified_archive
from scripts.prepare_public_release import clean

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT/'scripts/_artifact_paths.py'
if not HELPER.exists():
    HELPER = ROOT/'scripts/report_assets/public_artifact_paths.py'
spec = importlib.util.spec_from_file_location('publication_artifact_paths', HELPER)
paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paths)


class PublicReleaseTests(unittest.TestCase):
    def test_path_sanitizer_preserves_quoted_home_prefix_check(self):
        source="assert '/home/' not in text\n"
        self.assertEqual(clean(source),source)

    def test_paths_default_and_env_and_cli(self):
        with mock.patch.object(paths,'ROOT',Path('/tmp/project')), mock.patch.dict('os.environ',{},clear=True):
            self.assertEqual(paths.artifact_root_default(),Path('/tmp/project/outputs'))
            with mock.patch.dict('os.environ',{'ICSHM_OUTPUT_ROOT':'relative artifacts'}):
                self.assertEqual(paths.artifact_root_default(),Path('/tmp/project/relative artifacts'))
                with mock.patch('sys.argv',['program','--artifact-root','/tmp/explicit root']):
                    self.assertEqual(paths.parse_artifact_root(),Path('/tmp/explicit root'))
            with mock.patch.dict('os.environ',{'ICSHM_OUTPUT_ROOT':'/tmp/environment root'}):
                with mock.patch('sys.argv',['program']):
                    self.assertEqual(paths.parse_artifact_root(),Path('/tmp/environment root'))

    def archive(self, path, name='index.html', corrupt=False):
        data=b'<html>verified site</html>'
        record={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        with zipfile.ZipFile(path,'w') as z:
            z.writestr(name,b'corrupt' if corrupt else data)
            z.writestr('package_manifest.json',json.dumps({'files':{name:record}}))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_verified_archive_and_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/'site.zip';digest=self.archive(archive)
            self.assertEqual(extract_verified_archive(archive,root/'中文 space',digest),1)
            with self.assertRaises(FileExistsError):extract_verified_archive(archive,root/'中文 space',digest)

    def test_integrity_rejection_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/'site.zip';digest=self.archive(archive,corrupt=True)
            with self.assertRaises(ValueError):extract_verified_archive(archive,root/'out','0'*64)
            with self.assertRaises(ValueError):extract_verified_archive(archive,root/'out',digest)
            self.assertFalse((root/'out').exists())

    def test_archive_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/'site.zip';digest=self.archive(archive,name='../escape.html')
            with self.assertRaises(ValueError):extract_verified_archive(archive,root/'out',digest)
            self.assertFalse((root/'escape.html').exists())


if __name__ == '__main__':unittest.main()
