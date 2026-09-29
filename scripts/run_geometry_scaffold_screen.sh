#!/usr/bin/env bash
# Run one fresh, bounded Scaffold screening experiment from iteration 0.
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 EXPERIMENT_ID PREPARED_SCENE OUTPUT_ROOT ITERATIONS" >&2
  exit 2
fi

experiment_id="$1"
scene="$2"
output_root="$3"
iterations="$4"

if [[ "$iterations" != "3000" ]]; then
  echo "This pre-registered screen only permits a 3000-iteration budget" >&2
  exit 2
fi

: "${R1_ENV_ROOT:?set R1_ENV_ROOT to the frozen R1 Python environment}"
: "${R1_CUDA_HOME:?set R1_CUDA_HOME to the frozen CUDA toolkit}"
: "${R1_THIRD_PARTY_ROOT:?set R1_THIRD_PARTY_ROOT to pinned third-party roots}"
: "${ICSHM_DATA_ROOT:?set ICSHM_DATA_ROOT to the Contest Dataset root}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_root="$R1_THIRD_PARTY_ROOT/scaffold-gs"
model_path="$output_root/$experiment_id"
records=(1000 3000)

if [[ ! -f "$source_root/train.py" || ! -f "$source_root/render.py" ]]; then
  echo "Pinned Scaffold-GS source is unavailable: $source_root" >&2
  exit 1
fi
if [[ ! -d "$scene/images" || ! -f "$scene/sparse/0/test.txt" ]]; then
  echo "Prepared scene must have images and sparse/0/test.txt: $scene" >&2
  exit 1
fi
if [[ -e "$model_path" ]]; then
  echo "Refusing to overwrite existing run output: $model_path" >&2
  exit 1
fi

export CUDA_HOME="$R1_CUDA_HOME"
export PATH="$R1_ENV_ROOT/bin:$R1_CUDA_HOME/bin:$PATH"
torch_lib="$($R1_ENV_ROOT/bin/python -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent / "lib")')"
torch_cuda_runtime="$($R1_ENV_ROOT/bin/python -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parents[1] / "nvidia/cuda_runtime/lib")' 2>/dev/null || true)"
export LD_LIBRARY_PATH="$torch_lib:$torch_cuda_runtime:$R1_CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
# Geometry runs use the extensions installed for the active R1 Python.  The
# source submodule directories may contain a stale ABI (for example cp312
# artifacts alongside a cp311 environment), so do not shadow installed
# extensions with those directories.
export PYTHONPATH="${R1_IMPORT_SHIM:-}:$source_root"
export TORCH_CUDA_ARCH_LIST="12.0"
export CC="/usr/bin/gcc"
export CXX="/usr/bin/g++"
export TORCH_HOME="${R1_TORCH_HOME:-$output_root/../weights/torch}"
mkdir -p "$TORCH_HOME"

start_epoch="$(date --iso-8601=seconds)"
start_seconds="$(date +%s)"
"$R1_ENV_ROOT/bin/python" "$source_root/train.py" \
  --source_path "$scene" --model_path "$model_path" --eval --resolution 1 --iterations "$iterations" \
  --test_iterations "${records[@]}" --save_iterations "${records[@]}" --checkpoint_iterations "${records[@]}" \
  --skip_metric_eval --quiet

GEOMETRY_RECORD_ITERATIONS="${records[*]}" "$repo_root/scripts/evaluate_geometry_scaffold_checkpoints.sh" "$model_path" "$scene"

end_epoch="$(date --iso-8601=seconds)"
end_seconds="$(date +%s)"
"$R1_ENV_ROOT/bin/python" "$repo_root/scripts/write_geometry_run_metadata.py" \
  --output "$model_path/run_metadata.json" --experiment-id "$experiment_id" \
  --start-time "$start_epoch" --end-time "$end_epoch" --start-seconds "$start_seconds" --end-seconds "$end_seconds" \
  --record-iterations "${records[@]}"
