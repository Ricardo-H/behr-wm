#!/bin/bash
# =============================================================================
# Behavior Consistency Evaluation Script (Metric 3)
# =============================================================================
#
# Usage:
#   1. Start the Reference Agent (judge) vLLM server (port 8000):
#        bash scripts/servers/start_reference_agent_server.sh -m Qwen/Qwen3-8B -p 8000 -gpu 0
#
#   2. Start the World Model vLLM server (port 8001):
#        bash scripts/servers/start_wm_server.sh -m <wm_model_path> -p 8001 -gpu 1
#
#   3. Run evaluation:
#        bash eval/03_behavior_consistency/run_eval_bf.sh <path_to_test.json>
# =============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$SCRIPT_DIR"

if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# === Configuration ===
TEST_FILE="${1:-}"
WM_PORT="${WM_PORT:-8001}"
JUDGE_PORT="${JUDGE_PORT:-8000}"
OUTPUT_DIR="${OUTPUT_DIR:-./eval_results/$(date +%Y-%m-%d_%H-%M-%S)}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
MAX_STEPS="${MAX_STEPS:--1}"

if [ -z "$TEST_FILE" ] || [ ! -f "$TEST_FILE" ]; then
    echo "Usage: bash run_eval_bf.sh <path_to_test_file.json>"
    echo ""
    echo "The test file is a JSON list of logged trajectories. See docs/EVALUATION.md"
    echo "for the schema; obtain the standard test split with:"
    echo "    python scripts/download_data.py"
    exit 1
fi

echo "=========================================="
echo "Behavior Consistency Evaluation (Metric 3)"
echo "=========================================="
echo "Test File:  $TEST_FILE"
echo "WM Port:    $WM_PORT"
echo "Judge Port: $JUDGE_PORT"
echo "Output Dir: $OUTPUT_DIR"
echo "=========================================="

python eval_behavior_consistency.py \
    --test-file "$TEST_FILE" \
    --wm-port "$WM_PORT" \
    --judge-port "$JUDGE_PORT" \
    --output-dir "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --max-steps "$MAX_STEPS" \
    --reward-mode exponential \
    --behavior-scale-coef 10.0

echo "=========================================="
echo "Evaluation complete. Results: $OUTPUT_DIR"
echo "=========================================="
