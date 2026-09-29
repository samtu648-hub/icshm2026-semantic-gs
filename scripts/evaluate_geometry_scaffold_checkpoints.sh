#!/usr/bin/env bash
# Render and score existing Scaffold checkpoints with the frozen R1 evaluator.
# This is deliberately separate from Scaffold's optional upstream VGG metric:
# GeometryPrior comparisons use the pre-registered LPIPS-Alex implementation.
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
: "${ICSHM_DATA_ROOT:?set ICSHM_DATA_ROOT to the Contest Dataset root}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_root="$R1_THIRD_PARTY_ROOT/scaffold-gs"
split="$repo_root/experiments/splits/rgb_r1_spatial_v1.json"
if [[ -n "${GEOMETRY_RECORD_ITERATIONS:-}" ]]; then
  read -r -a records <<< "$GEOMETRY_RECORD_ITERATIONS"
else
  records=(1000 3000 5000 10000)
fi
if [[ "${#records[@]}" -eq 0 ]]; then
  echo "GEOMETRY_RECORD_ITERATIONS must contain at least one iteration" >&2
  exit 2
fi

if [[ ! -d "$model_path" || ! -d "$scene/images" || ! -f "$scene/sparse/0/test.txt" ]]; then
  echo "Missing model output or prepared scene: $model_path ; $scene" >&2
  exit 1
fi

export CUDA_HOME="$R1_CUDA_HOME"
export PATH="$R1_ENV_ROOT/bin:$R1_CUDA_HOME/bin:$PATH"
torch_lib="$($R1_ENV_ROOT/bin/python -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent / "lib")')"
torch_cuda_runtime="$($R1_ENV_ROOT/bin/python -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parents[1] / "nvidia/cuda_runtime/lib")' 2>/dev/null || true)"
export LD_LIBRARY_PATH="$torch_lib:$torch_cuda_runtime:$R1_CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
# Use the extensions installed for the active R1 Python ABI rather than stale
# source-tree artifacts from another Python minor version.
export PYTHONPATH="${R1_IMPORT_SHIM:-}:$source_root"
export TORCH_CUDA_ARCH_LIST="12.0"
export CC="/usr/bin/gcc"
export CXX="/usr/bin/g++"
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
  if [[ -d "$native" ]]; then
    observed_views="$(find "$native" -maxdepth 1 -type f -name '*.png' | wc -l)"
    if [[ "$observed_views" != "$expected_views" ]]; then
      echo "Refusing to overwrite incomplete native render directory ($observed_views/$expected_views): $native" >&2
      exit 1
    fi
  else
    "$R1_ENV_ROOT/bin/python" "$source_root/render.py" \
      --source_path "$scene" --model_path "$model_path" --iteration "$iteration" --skip_train --quiet
  fi
  if [[ -e "$prediction" || -e "$summary" || -e "$per_view" || -e "$mapping" ]]; then
    echo "Refusing to overwrite fixed evaluation output for iteration $iteration" >&2
    exit 1
  fi
  "$R1_ENV_ROOT/bin/python" "$repo_root/scripts/organize_r1_renders.py" \
    --split "$split" --source-dir "$native" --output-dir "$prediction" --mapping-json "$mapping"
  "$R1_ENV_ROOT/bin/python" "$repo_root/scripts/evaluate_rgb.py" \
    --split "$split" --reference-dir "$scene/images" --prediction-dir "$prediction" \
    --per-view-csv "$per_view" --summary-json "$summary" --lpips-device cuda
done
