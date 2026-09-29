#!/usr/bin/env bash
# Run one registered R1 RGB screening/full candidate with its frozen explicit validation split.
set -euo pipefail

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  echo "usage: $0 EXPERIMENT_ID {standard-3dgs|mip-splatting|scaffold-gs} PREPARED_SCENE ITERATIONS [SCAFFOLD_PRUNE_UNTIL]" >&2
  exit 2
fi

experiment_id="$1"
candidate="$2"
scene="$3"
iterations="$4"
scaffold_prune_until="${5:-}"

if [ -n "$scaffold_prune_until" ] && [ "$candidate" != "scaffold-gs" ]; then
  echo "SCAFFOLD_PRUNE_UNTIL is valid only for scaffold-gs" >&2
  exit 2
fi

: "${R1_ENV_ROOT:?set R1_ENV_ROOT to the isolated Python environment}"
: "${R1_CUDA_HOME:?set R1_CUDA_HOME to the isolated CUDA toolkit}"
: "${R1_THIRD_PARTY_ROOT:?set R1_THIRD_PARTY_ROOT to the Git-ignored pinned source root}"
: "${R1_OUTPUT_ROOT:?set R1_OUTPUT_ROOT to a Git-ignored output root}"
: "${ICSHM_DATA_ROOT:?set ICSHM_DATA_ROOT to the Contest Dataset root}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_root="$R1_THIRD_PARTY_ROOT/$candidate"
model_path="$R1_OUTPUT_ROOT/r1/runs/$experiment_id"
split="$repo_root/experiments/splits/rgb_r1_spatial_v1.json"

if [ ! -d "$source_root" ] || [ ! -f "$source_root/train.py" ]; then
  echo "Pinned candidate source is unavailable: $source_root" >&2
  exit 1
fi
if [ ! -d "$scene/images" ] || [ ! -f "$scene/sparse/0/test.txt" ]; then
  echo "Prepared scene must contain images and sparse/0/test.txt: $scene" >&2
  exit 1
fi
if [ -e "$model_path" ]; then
  echo "Refusing to overwrite experiment output: $model_path" >&2
  exit 1
fi

export CUDA_HOME="$R1_CUDA_HOME"
export PATH="$R1_ENV_ROOT/bin:$R1_CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$R1_CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$source_root/submodules/diff-gaussian-rasterization:$source_root/submodules/simple-knn:$source_root"
export TORCH_CUDA_ARCH_LIST="12.0"
export CC="/usr/bin/gcc"
export CXX="/usr/bin/g++"
# Keep pretrained metric weights outside Git and independent from any other project cache.
# R1_SKIP_LPIPS=1 is strictly a temporary non-ranking escape hatch while this cache is absent.
export TORCH_HOME="${R1_TORCH_HOME:-$R1_OUTPUT_ROOT/r1/weights/torch}"
mkdir -p "$TORCH_HOME"

common=(--source_path "$scene" --model_path "$model_path" --eval --resolution 1 --iterations "$iterations" \
  --test_iterations "$iterations" --save_iterations "$iterations" --checkpoint_iterations "$iterations" --quiet)

case "$candidate" in
  standard-3dgs)
    "$R1_ENV_ROOT/bin/python" "$source_root/train.py" "${common[@]}"
    "$R1_ENV_ROOT/bin/python" "$source_root/render.py" --source_path "$scene" --model_path "$model_path" --iteration "$iterations" --skip_train --quiet
    native_renders="$model_path/test/ours_${iterations}/renders"
    ;;
  mip-splatting)
    "$R1_ENV_ROOT/bin/python" "$source_root/train.py" "${common[@]}"
    "$R1_ENV_ROOT/bin/python" "$source_root/render.py" --source_path "$scene" --model_path "$model_path" --iteration "$iterations" --skip_train --quiet
    native_renders="$model_path/test/ours_${iterations}/test_preds_1"
    ;;
  scaffold-gs)
    scaffold_args=()
    if [ "${R1_SKIP_LPIPS:-0}" = "1" ]; then
      scaffold_args+=(--skip_metric_eval)
    fi
    if [ -n "$scaffold_prune_until" ]; then
      scaffold_args+=(--prune_until "$scaffold_prune_until")
    fi
    "$R1_ENV_ROOT/bin/python" "$source_root/train.py" "${common[@]}" "${scaffold_args[@]}"
    native_renders="$model_path/test/ours_${iterations}/renders"
    ;;
  *)
    echo "Unsupported candidate: $candidate" >&2
    exit 2
    ;;
esac

prediction_dir="$model_path/r1_validation_renders"
"$R1_ENV_ROOT/bin/python" "$repo_root/scripts/organize_r1_renders.py" \
  --split "$split" --source-dir "$native_renders" --output-dir "$prediction_dir" \
  --mapping-json "$model_path/r1_render_mapping.json"
lpips_args=(--lpips-device cuda)
if [ "${R1_SKIP_LPIPS:-0}" = "1" ]; then
  # This is allowed only for non-ranking execution smoke tests when the frozen AlexNet
  # weight is not cached yet.  The generated summary intentionally omits LPIPS.
  lpips_args=(--skip-lpips)
fi
"$R1_ENV_ROOT/bin/python" "$repo_root/scripts/evaluate_rgb.py" \
  --split "$split" --reference-dir "$scene/images" --prediction-dir "$prediction_dir" \
  --per-view-csv "$model_path/r1_metrics_per_view.csv" --summary-json "$model_path/r1_metrics_summary.json" \
  "${lpips_args[@]}"
