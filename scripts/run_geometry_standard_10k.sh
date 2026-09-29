#!/usr/bin/env bash
# Run one fresh continuous Standard 3DGS 0->10k arm for GEO-027.
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 EXPERIMENT_ID PREPARED_SCENE OUTPUT_ROOT" >&2
  exit 2
fi

experiment_id="$1"
scene="$2"
output_root="$3"

case "$experiment_id" in
  GEO-027-STD-CTRL-10K|GEO-027-STD-VGGT-10K) ;;
  *) echo "unregistered GEO-027 experiment id: $experiment_id" >&2; exit 2 ;;
esac

: "${R1_ENV_ROOT:?set R1_ENV_ROOT to the frozen R1 Python environment}"
: "${R1_CUDA_HOME:?set R1_CUDA_HOME to the frozen CUDA toolkit}"
: "${R1_THIRD_PARTY_ROOT:?set R1_THIRD_PARTY_ROOT to pinned third-party roots}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_root="$R1_THIRD_PARTY_ROOT/standard-3dgs"
model_path="$output_root/$experiment_id"
records=(1000 3000 5000 10000)
config="$repo_root/experiments/runs/$experiment_id/config.yaml"

if [[ ! -f "$source_root/train.py" || ! -f "$source_root/render.py" ]]; then
  echo "Pinned Standard 3DGS source is unavailable: $source_root" >&2
  exit 1
fi
if [[ ! -d "$scene/images" || ! -f "$scene/sparse/0/test.txt" || ! -f "$scene/sparse/0/points3D.ply" ]]; then
  echo "Prepared scene is incomplete: $scene" >&2
  exit 1
fi
if [[ -e "$model_path" ]]; then
  echo "Refusing to overwrite existing run output: $model_path" >&2
  exit 1
fi
mkdir -p "$output_root"
"$repo_root/scripts/verify_geo027_standard_source.py" \
  --source "$source_root" --config "$config"

# Share the same exclusive GPU admission lock as the other repository jobs.
mkdir -p "$repo_root/outputs/quality"
exec 9>"$repo_root/outputs/quality/gpu.lock"
echo "Waiting for the shared GPU lock"
flock 9

gpu_row="$(nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free \
  --format=csv,noheader,nounits)"
compute_apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits)"
allowed_todesk_memory_mib=0
unexpected_compute_apps=""
while IFS=',' read -r pid process_name used_memory; do
  pid="${pid#"${pid%%[![:space:]]*}"}"
  pid="${pid%"${pid##*[![:space:]]}"}"
  process_name="${process_name#"${process_name%%[![:space:]]*}"}"
  process_name="${process_name%"${process_name##*[![:space:]]}"}"
  used_memory="${used_memory#"${used_memory%%[![:space:]]*}"}"
  used_memory="${used_memory%"${used_memory##*[![:space:]]}"}"
  [[ -z "$pid" ]] && continue
  if [[ "${process_name##*/}" == "ToDesk_Session" && "$used_memory" =~ ^[0-9]+$ ]]; then
    allowed_todesk_memory_mib=$((allowed_todesk_memory_mib + used_memory))
  else
    unexpected_compute_apps+="${pid}, ${process_name}, ${used_memory}"$'\n'
  fi
done <<<"$compute_apps"
if [[ -n "$unexpected_compute_apps" ]]; then
  echo "Refusing to start while an unapproved compute process is using the GPU:" >&2
  printf '%s' "$unexpected_compute_apps" >&2
  exit 1
fi
if (( allowed_todesk_memory_mib > 1024 )); then
  echo "ToDesk_Session GPU memory exceeds admission limit: ${allowed_todesk_memory_mib} MiB > 1024 MiB" >&2
  exit 1
fi
if (( allowed_todesk_memory_mib > 0 )); then
  echo "Admitting ToDesk_Session desktop process at ${allowed_todesk_memory_mib} MiB"
fi
gpu_free_mib="$(awk -F, '{gsub(/ /, "", $4); print $4}' <<<"$gpu_row")"
min_free_mib="${GEO027_MIN_FREE_MIB:-10000}"
if (( gpu_free_mib < min_free_mib )); then
  echo "Insufficient free GPU memory: ${gpu_free_mib} MiB < ${min_free_mib} MiB" >&2
  exit 1
fi
disk_free_kib="$(df -Pk "$output_root" | awk 'NR==2 {print $4}')"
if (( disk_free_kib < 104857600 )); then
  echo "Less than 100 GiB remains on the output filesystem" >&2
  exit 1
fi

export CUDA_HOME="$R1_CUDA_HOME"
export PATH="$R1_ENV_ROOT/bin:$R1_CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$R1_CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$source_root/submodules/diff-gaussian-rasterization:$source_root/submodules/simple-knn:$source_root"
export TORCH_CUDA_ARCH_LIST="12.0"
export CC="/usr/bin/gcc"
export CXX="/usr/bin/g++"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_HOME="${R1_TORCH_HOME:-$output_root/../weights/torch}"
mkdir -p "$TORCH_HOME"

"$R1_ENV_ROOT/bin/python" -c \
  'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

start_epoch="$(date --iso-8601=seconds)"
start_seconds="$(date +%s)"
"$R1_ENV_ROOT/bin/python" "$source_root/train.py" \
  --source_path "$scene" --model_path "$model_path" --eval --resolution 1 \
  --iterations 10000 --test_iterations -1 --save_iterations "${records[@]}" \
  --checkpoint_iterations "${records[@]}" --quiet

"$repo_root/scripts/evaluate_geometry_standard_checkpoints.sh" "$model_path" "$scene"

end_epoch="$(date --iso-8601=seconds)"
end_seconds="$(date +%s)"
"$R1_ENV_ROOT/bin/python" "$repo_root/scripts/write_geometry_run_metadata.py" \
  --output "$model_path/run_metadata.json" --experiment-id "$experiment_id" \
  --start-time "$start_epoch" --end-time "$end_epoch" \
  --start-seconds "$start_seconds" --end-seconds "$end_seconds"
