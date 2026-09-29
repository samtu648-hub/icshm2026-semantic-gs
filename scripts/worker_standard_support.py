#!/usr/bin/env python3
"""Apply process-local split hooks while retaining the audited native worker."""
import argparse
import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('worker_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source = ROOT / config['source']
    sys.path[:0] = [str(source), str(source/'submodules/diff-gaussian-rasterization'),
                   str(source/'submodules/simple-knn'), str(ROOT)]
    from scene import GaussianModel
    from src.standard_support_densification import install
    observe = install(GaussianModel, config['support'], args.output)
    original = runpy.run_path
    def patched(path, *a, **kw):
        if Path(path).resolve() == (source/'train.py').resolve():
            text = Path(path).read_text()
            needle = '        loss.backward()\n'
            if text.count(needle) != 1:
                raise RuntimeError('Native training hook location changed')
            text = text.replace(needle, needle + '        support_observe(iteration, viewpoint_cam, loss, gaussians)\n')
            (args.output/'executed_train.py').write_text(text)
            scope = dict(__name__='__main__', __file__=str(path), support_observe=observe)
            exec(compile(text, str(path), 'exec'), scope)
            return scope
        return original(path, *a, **kw)
    runpy.run_path = patched
    worker_args = args.worker_args[1:] if args.worker_args[:1] == ['--'] else args.worker_args
    sys.argv = [str(ROOT/'scripts/backbone_all400_worker.py'), *worker_args]
    original(sys.argv[0], run_name='__main__')

if __name__ == '__main__':
    main()
