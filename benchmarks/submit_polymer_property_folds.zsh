#!/usr/local_rwth/bin/zsh

set -euo pipefail

N_FOLDS=10
SLURM_SCRIPT="slurm_cpu.sh"
BENCHMARK_SCRIPT="benchmarks/polymer_property_prediction_benchmark.py"
OUTPUT_DIR="benchmarks/results/polymers"
FORWARD_ARGS=()

while (( $# > 0 )); do
  case "$1" in
    --n-folds)
      N_FOLDS="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      FORWARD_ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "$SLURM_SCRIPT" ]]; then
  echo "ERROR: SLURM submission script not found: $SLURM_SCRIPT" >&2
  exit 1
fi

if [[ ! -f "$BENCHMARK_SCRIPT" ]]; then
  echo "ERROR: benchmark script not found: $BENCHMARK_SCRIPT" >&2
  exit 1
fi

echo "Submitting $N_FOLDS fold jobs using $SLURM_SCRIPT"
for fold in $(seq 1 "$N_FOLDS"); do
  tag="fold_${fold}"
  cmd=(
    sbatch
    --job-name "polymer_fold_${fold}"
    "$SLURM_SCRIPT"
    "$BENCHMARK_SCRIPT"
    --n-folds "$N_FOLDS"
    --fold-index "$fold"
    --results-tag "$tag"
    "${FORWARD_ARGS[@]}"
  )
  echo "Submitting fold $fold/$N_FOLDS"
  "${cmd[@]}"
done

echo "All fold jobs submitted."
