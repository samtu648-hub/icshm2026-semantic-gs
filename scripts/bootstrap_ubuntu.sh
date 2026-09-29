#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:---check-only}"

case "$mode" in
  --check-only)
    ;;
  --create-dirs)
    mkdir -p "$repo_root/outputs" "$repo_root/experiments/runs"
    ;;
  *)
    echo "Usage: $0 [--check-only|--create-dirs]" >&2
    exit 2
    ;;
esac

for command in git bash python3; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Missing required command: $command" >&2
    exit 1
  fi
done

echo "Repository: $repo_root"
echo "Git: $(git --version)"
echo "Bash: ${BASH_VERSION}"
echo "Python: $(python3 --version 2>&1)"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA driver detected."
else
  echo "NVIDIA driver not detected in this environment; run scripts/check_gpu.sh on the GPU server."
fi

echo "No system packages, drivers or CUDA components were modified."
echo "Environment versions will be locked after RGB backbone freeze R2."

