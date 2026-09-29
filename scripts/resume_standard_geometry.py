#!/usr/bin/env python3
"""Resume a retained state to a new directory with the original actual iteration budget."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quality_protocol import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--artifact-root',type=Path,required=True);a=p.parse_args()
    previous=json.loads((a.run/'started.json').read_text());latest=json.loads((a.run/'recovery/latest.json').read_text())
    config=ROOT/f"experiments/runs/{previous['experiment_id']}/config.yaml"
    assert sha256(config)==previous['config_sha256']
    for name,h in previous['code_sha256'].items():assert sha256(ROOT/name)==h,'Current source differs from retained run; do not silently resume'
    cmd=[sys.executable,str(ROOT/'scripts/run_standard_geometry.py'),'--artifact-root',str(a.artifact_root),'--config',str(config),
         '--output',str(a.output),'--iterations',str(previous['actual_iterations']),'--resume-state',latest['path']]
    subprocess.run(cmd,check=True)


if __name__=='__main__':main()
