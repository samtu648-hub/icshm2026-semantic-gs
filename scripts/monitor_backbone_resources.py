#!/usr/bin/env python3
"""Observe the active comparison without altering GPU state."""
import json,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];base=ROOT/'outputs/quality/backbone_comparison';out=base/'resource_timeline.jsonl'
with out.open('x') as f:
 while True:
  status=base/'queue-r1/status.json';phase=json.loads(status.read_text())['status'] if status.exists() else 'starting'
  gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw,clocks.sm,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
  memory={line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','MemTotal:','SwapFree:','SwapTotal:'))}
  f.write(json.dumps(dict(unix=time.time(),phase=phase,gpu=gpu.stdout.strip(),gpu_returncode=gpu.returncode,free_bytes=shutil.disk_usage(base).free,system_memory=memory))+'\n');f.flush()
  if (base/'queue-r1/result.json').exists():break
  time.sleep(15)
