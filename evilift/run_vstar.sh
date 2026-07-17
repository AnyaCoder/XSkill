#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
  echo "Usage: $0 {prepare|smoke|pilot|full|baseline-dev|skill-dev|select-dev|baseline-test|skill-test} [arguments]" >&2
  exit 2
fi
shift

VSTAR_SOURCE="${VSTAR_SOURCE:-$REPO_ROOT/../datasets/vstar/annotation_vstar_updated.json}"
VSTAR_IMAGE_ROOT="${VSTAR_IMAGE_ROOT:-$REPO_ROOT/../datasets}"
VSTAR_BENCHMARK_DIR="${VSTAR_BENCHMARK_DIR:-$REPO_ROOT/benchmark/VStar_Evilift}"
VSTAR_OUTPUT_ROOT="${VSTAR_OUTPUT_ROOT:-$REPO_ROOT/output/vstar_single_image}"
VSTAR_MEMORY_ROOT="${VSTAR_MEMORY_ROOT:-$REPO_ROOT/memory_bank/vstar_single_image}"
VSTAR_LOG_ROOT="${VSTAR_LOG_ROOT:-$REPO_ROOT/logs/vstar_single_image}"
VSTAR_RESUME="${VSTAR_RESUME:-0}"
DEFAULT_CONDA_BASE="$(conda info --base 2>/dev/null || true)"
VSTAR_PYTHON="${VSTAR_PYTHON:-${DEFAULT_CONDA_BASE:+$DEFAULT_CONDA_BASE/envs/implicitcue/bin/python}}"
if [[ -z "$VSTAR_PYTHON" || ! -x "$VSTAR_PYTHON" ]]; then
  echo "implicitcue Python not found; set VSTAR_PYTHON to its Python executable" >&2
  exit 1
fi

prepare_data() {
  "$VSTAR_PYTHON" -m evilift.prepare_vstar \
    --input "$VSTAR_SOURCE" \
    --output-dir "$VSTAR_BENCHMARK_DIR" \
    --seed 42 \
    --pilot-size 8
}

require_reasoning_api() {
  : "${REASONING_MODEL_NAME:?REASONING_MODEL_NAME is required}"
  : "${REASONING_API_KEY:?REASONING_API_KEY is required}"
  : "${REASONING_END_POINT:?REASONING_END_POINT is required}"

  export EXPERIENCE_MODEL_NAME="${EXPERIENCE_MODEL_NAME:-$REASONING_MODEL_NAME}"
  export EXPERIENCE_API_KEY="${EXPERIENCE_API_KEY:-$REASONING_API_KEY}"
  export EXPERIENCE_END_POINT="${EXPERIENCE_END_POINT:-$REASONING_END_POINT}"
  export ENABLE_FUNCTION_CALLING="true"
  export ENABLED_TOOLS="zoom"

  # V* is multiple choice and is scored deterministically after inference.
  unset VERIFIER_API_KEY VERIFIER_END_POINT
}

ensure_prepared() {
  if [[ ! -f "$VSTAR_BENCHMARK_DIR/manifest.json" ]]; then
    prepare_data
  fi
}

ensure_run_directory() {
  local output_dir="$1"
  if [[ -f "$output_dir/results.jsonl" && "$VSTAR_RESUME" != "1" ]]; then
    echo "Run already completed: $output_dir (set VSTAR_RESUME=1 to verify/resume it)" >&2
    exit 1
  fi
  mkdir -p "$output_dir" "$VSTAR_LOG_ROOT"
}

run_distillation() {
  local stage="$1"
  local input_file="$2"
  local rollouts="$3"
  local large_batch="$4"
  local output_dir="$VSTAR_OUTPUT_ROOT/$stage"
  local memory_dir="$VSTAR_MEMORY_ROOT/$stage"

  ensure_run_directory "$output_dir"
  if [[ "$VSTAR_RESUME" != "1" && ! -d "$output_dir/snapshots" ]]; then
    if [[ -e "$memory_dir/SKILL.md" || -e "$memory_dir/experiences.json" ]]; then
      echo "Refusing to reuse a knowledge bank without its run snapshots: $memory_dir" >&2
      exit 1
    fi
  fi
  mkdir -p "$memory_dir"

  "$VSTAR_PYTHON" -u eval/infer_api.py \
    --input-file "$input_file" \
    --image-folder "$VSTAR_IMAGE_ROOT" \
    --output-dir "$output_dir" \
    --temperature 0.6 \
    --top-p 1.0 \
    --max-completion-tokens 8192 \
    --max-turns 6 \
    --max-images 8 \
    --max-total-tokens 32768 \
    --num-workers 8 \
    --rollouts-per-sample "$rollouts" \
    --seed-base 42 \
    --system-prompt-key agent_zoom \
    --tool-config-path eval/configs/tool_configs.yaml \
    --skip-completed \
    --skill-enable \
    --skill-library "$memory_dir/SKILL.md" \
    --skill-inference \
    --no-skill-adaptation \
    --skill-refine \
    --skill-max-length 1000 \
    --experience-online-generate \
    --experience-library-update \
    --experience-library "$memory_dir/experiences.json" \
    --experience-max-ops 3 \
    --experience-large-batch "$large_batch" \
    --experience-max-items 120 \
    --experience-refine \
    2>&1 | tee "$VSTAR_LOG_ROOT/$stage.log"

  "$VSTAR_PYTHON" -m evilift.evaluate_vstar score \
    --benchmark "$input_file" \
    --results "$output_dir/results.jsonl" \
    --trajectories "$output_dir" \
    --skill "$memory_dir/SKILL.md" \
    --audit-benchmark "$VSTAR_BENCHMARK_DIR/train.json" \
    --output "$output_dir/vstar_metrics.json"

  if [[ "$stage" == "pilot" ]]; then
    "$VSTAR_PYTHON" -m evilift.evaluate_vstar pilot-gate \
      --metrics "$output_dir/vstar_metrics.json" \
      --skill "$memory_dir/SKILL.md" \
      --experiences "$memory_dir/experiences.json" \
      --expected-rollouts 16 \
      --output "$output_dir/pilot_gate.json"
  fi
}

run_evaluation() {
  local split="$1"
  local condition="$2"
  local skill_path="${3:-}"
  local tag="${4:-$condition}"
  if [[ ! "$tag" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Evaluation tag may contain only letters, digits, dot, underscore, and dash" >&2
    exit 2
  fi
  if [[ "$condition" == "skill" && ! -f "$skill_path" ]]; then
    echo "Skill file not found: $skill_path" >&2
    exit 1
  fi

  local input_file="$VSTAR_BENCHMARK_DIR/$split.json"
  local output_dir="$VSTAR_OUTPUT_ROOT/${tag}_${split}"
  ensure_run_directory "$output_dir"

  local skill_args=()
  if [[ "$condition" == "skill" ]]; then
    skill_args=(
      --skill-enable
      --skill-library "$skill_path"
      --skill-inference
      --no-skill-adaptation
    )
  fi

  "$VSTAR_PYTHON" -u eval/infer_api.py \
    --input-file "$input_file" \
    --image-folder "$VSTAR_IMAGE_ROOT" \
    --output-dir "$output_dir" \
    --temperature 0.0 \
    --top-p 1.0 \
    --max-completion-tokens 8192 \
    --max-turns 6 \
    --max-images 8 \
    --max-total-tokens 32768 \
    --num-workers 8 \
    --rollouts-per-sample 1 \
    --seed-base 42 \
    --system-prompt-key agent_zoom \
    --tool-config-path eval/configs/tool_configs.yaml \
    --skip-completed \
    "${skill_args[@]}" \
    2>&1 | tee "$VSTAR_LOG_ROOT/${tag}_${split}.log"

  local audit_args=()
  if [[ "$condition" == "skill" ]]; then
    audit_args=(
      --skill "$skill_path"
      --audit-benchmark "$VSTAR_BENCHMARK_DIR/train.json"
    )
  fi
  "$VSTAR_PYTHON" -m evilift.evaluate_vstar score \
    --benchmark "$input_file" \
    --results "$output_dir/results.jsonl" \
    --trajectories "$output_dir" \
    "${audit_args[@]}" \
    --output "$output_dir/vstar_metrics.json"
}

if [[ "$MODE" != "prepare" ]]; then
  ensure_prepared
fi

case "$MODE" in
  prepare)
    prepare_data
    ;;
  smoke)
    require_reasoning_api
    "$VSTAR_PYTHON" -m evilift.smoke_autodl \
      --image "$VSTAR_IMAGE_ROOT/vstar/direct_attributes/sa_7429.jpg" \
      --image "$VSTAR_IMAGE_ROOT/vstar/relative_position/sa_6531.jpg"
    "$VSTAR_PYTHON" -m evilift.smoke_zoom \
      --image "$VSTAR_IMAGE_ROOT/vstar/direct_attributes/sa_7429.jpg"
    ;;
  pilot)
    require_reasoning_api
    run_distillation pilot "$VSTAR_BENCHMARK_DIR/pilot.json" 2 8
    ;;
  full)
    require_reasoning_api
    run_distillation full "$VSTAR_BENCHMARK_DIR/train.json" 4 32
    ;;
  baseline-dev)
    require_reasoning_api
    run_evaluation dev baseline "" baseline
    ;;
  skill-dev)
    require_reasoning_api
    [[ $# -eq 2 ]] || { echo "Usage: $0 skill-dev SKILL_PATH TAG" >&2; exit 2; }
    run_evaluation dev skill "$1" "$2"
    ;;
  select-dev)
    [[ $# -ge 2 ]] || { echo "Usage: $0 select-dev BASELINE_METRICS CANDIDATE_METRICS..." >&2; exit 2; }
    baseline="$1"
    shift
    mkdir -p "$VSTAR_OUTPUT_ROOT"
    "$VSTAR_PYTHON" -m evilift.evaluate_vstar select \
      --baseline "$baseline" \
      --candidates "$@" \
      --output "$VSTAR_OUTPUT_ROOT/dev_selection.json"
    ;;
  baseline-test)
    require_reasoning_api
    run_evaluation test baseline "" baseline
    ;;
  skill-test)
    require_reasoning_api
    [[ $# -eq 2 ]] || { echo "Usage: $0 skill-test SKILL_PATH TAG" >&2; exit 2; }
    run_evaluation test skill "$1" "$2"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac
