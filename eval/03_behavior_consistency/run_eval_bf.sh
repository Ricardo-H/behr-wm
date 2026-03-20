#!/bin/bash
# =============================================================================
# Behavior Consistency Evaluation Script
# =============================================================================
# 
# 使用方法:
# 1. 先启动裁判模型 vLLM 服务 (端口 8000):
#    bash scripts/servers/start_reference_agent_server.sh
#
# 2. 启动世界模型 vLLM 服务 (端口 8001):
#    bash scripts/servers/start_wm_server.sh
# 
# 3. 运行评估:
#    bash run_eval_bf.sh
# =============================================================================

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$SCRIPT_DIR"

# Activate environment
if [ -f "$PROJECT_ROOT/uv_webshop/bin/activate" ]; then
    source "$PROJECT_ROOT/uv_webshop/bin/activate"
fi

# === Configuration ===
TEST_FILE="$PROJECT_ROOT/data/llama_factory/webshop_test_109.json"
WM_PORT=8001
REF_AGENT_PORT=8000
OUTPUT_DIR="./eval_results/$(date +%Y-%m-%d_%H-%M-%S)"
MAX_SAMPLES=-1  # -1 for all
MAX_STEPS=-1    # -1 for all

# Check test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo "Error: Test file not found: $TEST_FILE"
    echo "Please run 'python scripts/download_data.py' first."
    exit 1
fi

# === Run Evaluation ===
echo "=========================================="
echo "Behavior Consistency Evaluation"
echo "=========================================="
echo "Test File:  $TEST_FILE"
echo "WM Port:    $WM_PORT"
echo "Reference Agent Port: $REF_AGENT_PORT"
echo "Output Dir: $OUTPUT_DIR"
echo "=========================================="

python eval_behavior_consistency.py \
    --test-file "$TEST_FILE" \
    --wm-port $WM_PORT \
    --ref-agent-port $REF_AGENT_PORT \
    --output-dir "$OUTPUT_DIR" \
    --max-samples $MAX_SAMPLES \
    --max-steps-per-sample $MAX_STEPS \
    --reward-mode exponential \
    --behavior-scale-coef 10.0

echo "=========================================="
echo "Evaluation Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
