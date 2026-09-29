#!/usr/bin/env bash
# Continue the authorized R1 full runs serially after an already-running predecessor.
# It stops on any non-zero wrapper exit, missing fixed-metric summary, or low disk space.
set -euo pipefail

: "${R1_ENV_ROOT:?}"
: "${R1_CUDA_HOME:?}"
: "${R1_THIRD_PARTY_ROOT:?}"
: "${R1_OUTPUT_ROOT:?}"
: "${ICSHM_DATA_ROOT:?}"

predecessor_pid="$1"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_root="$R1_OUTPUT_ROOT/r1/logs"
mkdir -p "$log_root"

while kill -0 "$predecessor_pid" 2>/dev/null; do
  sleep 30
done

if [ ! -f "$R1_OUTPUT_ROOT/r1/runs/RGB-011-FULL/r1_metrics_summary.json" ]; then
  echo "RGB-011-FULL did not produce its fixed-metric summary; queue stopped." >&2
  exit 1
fi

run_one() {
  local experiment_id="$1"
  local candidate="$2"
  local scene="$3"
  local run_dir="$R1_OUTPUT_ROOT/r1/runs/$experiment_id"
  local log="$log_root/$experiment_id.log"
  local gpu_csv="$log_root/$experiment_id-gpu-samples.csv"
  local wrapper_pid sampler_pid

  if [ -e "$run_dir" ]; then
    echo "Refusing to overwrite existing output: $run_dir" >&2
    return 1
  fi
  if [ "$(df --output=avail -k "$R1_OUTPUT_ROOT" | tail -n 1 | tr -d ' ')" -lt 104857600 ]; then
    echo "Less than 100 GiB available at $R1_OUTPUT_ROOT; queue stopped." >&2
    return 1
  fi

  export R1_SKIP_LPIPS=1
  /usr/bin/time -v "$repo_root/scripts/run_r1_rgb_candidate.sh" "$experiment_id" "$candidate" "$scene" 30000 > "$log" 2>&1 &
  wrapper_pid=$!
  (
    printf 'timestamp,utilization_gpu_percent,memory_used_mib,memory_total_mib\n' > "$gpu_csv"
    while kill -0 "$wrapper_pid" 2>/dev/null; do
      nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits >> "$gpu_csv"
      sleep 10
    done
  ) &
  sampler_pid=$!
  wait "$wrapper_pid"
  wait "$sampler_pid" || true
  [ -f "$run_dir/r1_metrics_summary.json" ]
}

run_one RGB-020-FULL mip-splatting "$R1_OUTPUT_ROOT/rgb_r1_spatial_v1/scene_400_eval"
run_one RGB-030-FULL scaffold-gs "$R1_OUTPUT_ROOT/rgb_r1_spatial_v1/scene_400_eval"
