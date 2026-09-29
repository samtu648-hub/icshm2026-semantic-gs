#!/usr/bin/env bash
set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Run this on the Ubuntu GPU server after the NVIDIA driver is installed." >&2
  exit 1
fi

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

if command -v python3 >/dev/null 2>&1; then
  python3 - <<'PY'
try:
    import torch
except ImportError:
    print("PyTorch is not installed yet; this is expected before the RGB backbone environment is frozen.")
else:
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA: {torch.version.cuda}")
PY
fi

