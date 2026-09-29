#!/usr/bin/env bash
# Render and score GEO-027 Standard checkpoints with the fixed R1 evaluator.
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 MODEL_PATH PREPARED_SCENE" >&2
  exit 2
fi

model_path="$1"
scene="$2"
: "${R1_ENV_ROOT:?set R1_ENV_ROOT to the frozen R1 Python environment}"
: "${R1_CUDA_HOME:?set R1_CUDA_HOME to the frozen CUDA toolkit}"
: "${R1_THIRD_PARTY_ROOT:?set R1_THIRD_PARTY_ROOT to pinned third-party roots}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_root="$R1_THIRD_PARTY_ROOT/standard-3dgs"
split="$repo_root/experiments/splits/rgb_r1_spatial_v1.json"
records=(1000 3000 5000 10000)

export CUDA_HOME="$R1_CUDA_HOME"
export PATH="$R1_ENV_ROOT/bin:$R1_CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$R1_CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$source_root/submodules/diff-gaussian-rasterization:$source_root/submodules/simple-knn:$source_root"
export TORCH_CUDA_ARCH_LIST="12.0"
export TORCH_HOME="${R1_TORCH_HOME:-$(dirname "$model_path")/../weights/torch}"
mkdir -p "$TORCH_HOME"

expected_views="$(wc -l < "$scene/sparse/0/test.txt")"
for iteration in "${records[@]}"; do
  checkpoint="$model_path/chkpnt${iteration}.pth"
  native="$model_path/test/ours_${iteration}/renders"
  prediction="$model_path/r1_validation_renders/iteration_${iteration}"
  summary="$model_path/r1_metrics_summary_iteration_${iteration}.json"
  per_view="$model_path/r1_metrics_per_view_iteration_${iteration}.csv"
  mapping="$model_path/r1_render_mapping_iteration_${iteration}.json"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    exit 1
  fi
  if [[ ! -d "$native" ]]; then
    "$R1_ENV_ROOT/bin/python" "$source_root/render.py" \
      --source_path "$scene" --model_path "$model_path" --iteration "$iteration" \
      --skip_train --quiet
  fi
  observed_views="$(find "$native" -maxdepth 1 -type f -name '*.png' | wc -l)"
  if [[ "$observed_views" != "$expected_views" ]]; then
    echo "Incomplete Standard renders ($observed_views/$expected_views): $native" >&2
    exit 1
  fi
  if [[ -e "$prediction" || -e "$summary" || -e "$per_view" || -e "$mapping" ]]; then
    echo "Refusing to overwrite evaluation output for iteration $iteration" >&2
    exit 1
  fi
  "$R1_ENV_ROOT/bin/python" "$repo_root/scripts/organize_r1_renders.py" \
    --split "$split" --source-dir "$native" --output-dir "$prediction" \
    --mapping-json "$mapping"
  "$R1_ENV_ROOT/bin/python" "$repo_root/scripts/evaluate_rgb.py" \
    --split "$split" --reference-dir "$scene/images" --prediction-dir "$prediction" \
    --per-view-csv "$per_view" --summary-json "$summary" --lpips-device cuda
done
