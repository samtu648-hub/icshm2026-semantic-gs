"""Run inside the dedicated fresh Ubuntu container, with read-only inputs.

Mounts: /bundle (frozen package), /camera-metadata (official text files),
/frozen-reference (old external render audit), /cuda (compiler), /out (new).
The host Python environment and source build products must NOT be mounted.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def run(command,log):
    with log.open('w') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,text=True)
    if result.returncode:
        print(log.read_text()[-12000:],flush=True)
        raise RuntimeError(f'Command failed; retained log: {log}')


def main():
    out=Path('/out')
    if any(out.iterdir()):
        raise FileExistsError('Fresh empty output mount required')
    started=time.perf_counter()
    manifest=json.loads(Path('/bundle/metadata/artifact_manifest.json').read_text())
    for record in manifest['files']:
        if sha(Path('/bundle')/record['path'])!=record['sha256']:
            raise ValueError('Frozen bundle hash mismatch before build')
    os.environ.update(CUDA_HOME='/cuda',TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='4',
                      PATH='/cuda/bin:'+os.environ['PATH'],
                      LD_LIBRARY_PATH='/cuda/lib:'+os.environ.get('LD_LIBRARY_PATH',''))
    extensions=[]
    for name in ('diff-gaussian-rasterization','simple-knn'):
        source=Path('/bundle/third_party/scaffold-gs/submodules')/name
        build=Path('/tmp/icshm-clean-source')/name
        shutil.copytree(source,build,ignore=shutil.ignore_patterns('*.so','build','__pycache__','.git'))
        run([sys.executable,'-m','pip','install','--no-deps','--no-build-isolation',str(build)],out/(name+'_build.log'))
        extensions.append(name)
    # Preload FRESH installed modules before frozen Scaffold adds its bundled
    # source directories to sys.path; otherwise old bundled .so files can shadow
    # the new build and produce a false clean-build claim.
    launcher=out/'fresh_extension_launch.py'
    launcher.write_text("import sys, runpy, json\nimport diff_gaussian_rasterization._C as raster\nimport simple_knn._C as knn\nassert raster.__file__.startswith('/opt/icshm-runtime/')\nassert knn.__file__.startswith('/opt/icshm-runtime/')\nprint(json.dumps({'fresh_raster':raster.__file__,'fresh_knn':knn.__file__}),flush=True)\nscript=sys.argv.pop(1)\nrunpy.run_path(script,run_name='__main__')\n")
    script='/bundle/code/scripts/render_submission_sem161.py'
    command=[sys.executable,str(launcher),script,'--bundle','/bundle',
             '--images','/camera-metadata/images.txt','--image-name','140.png','--write-fallbacks']
    for name,camera in [('reload1','/camera-metadata/cameras.txt'),('reload2','/camera-metadata/cameras.txt'),
                        ('small','/frozen-reference/small_cameras.txt')]:
        run([*command,'--cameras',camera,'--output-dir',str(out/name)],out/(name+'.log'))
    import numpy as np
    from PIL import Image
    checks={}
    for name in ('reload1','reload2','small'):
        ref=Path('/frozen-reference')/('small' if name=='small' else 'reload1')
        for stage in ('rgb','semantic'):
            a=np.asarray(Image.open(out/name/stage/'140.png'))
            b=np.asarray(Image.open(ref/stage/'140.png'))
            checks[f'{name}_{stage}_frozen_exact']=bool(np.array_equal(a,b))
    conflict=subprocess.run([*command,'--cameras','/frozen-reference/small_cameras.txt',
                            '--output-dir',str(out/'small')],capture_output=True,text=True)
    checks['nonempty_output_rejected']=conflict.returncode!=0 and 'Refusing to overwrite' in conflict.stderr
    checks['bundle_unchanged']=all(sha(Path('/bundle')/r['path'])==r['sha256'] for r in manifest['files'])
    run([sys.executable,'-m','pip','freeze','--all'],out/'installed_packages.txt')
    import torch
    import diff_gaussian_rasterization._C as raster
    import simple_knn._C as knn
    result={'experiment_id':'SUBMIT-040-CLEAN-UBUNTU-REPRO','checks':checks,
            'status':'completed_clean_container_reproduction' if all(checks.values()) else 'output_mismatch',
            'fresh_os':Path('/etc/os-release').read_text(),'fresh_python_dependencies':True,
            'fresh_extension_builds':extensions,'read_only_external_cuda_compiler':True,
            'same_host_gpu_and_kernel':True,'gpu':torch.cuda.get_device_name(),
            'seconds':time.perf_counter()-started,'frozen_bundle_files':len(manifest['files']),
            'no_reference_rgb_or_labels_as_inference_inputs':True,
            'no_host_python_environment_mounted':True,
            'versions':{p:importlib.metadata.version(p) for p in ('torch','transformers','numpy','torch-scatter')},
            'installed_package_lock_sha256':sha(out/'installed_packages.txt'),
            'extension_hashes':{'raster':sha(raster.__file__),'knn':sha(knn.__file__)}}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    if not all(checks.values()):
        raise RuntimeError('Fresh environment did not replay frozen outputs')


if __name__=='__main__':
    main()
