#!/bin/bash
# =============================================================================
# Task Success Rate Evaluation - World Model
# =============================================================================
#
# Usage:
#   bash run_wm.sh [options]
#
# This script evaluates agent performance in the world model.
# =============================================================================

set -x

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# === Configuration ===
# $8: 任务名 (webshop / textworld / alfworld / sciworld, 默认 webshop)
TASK="${8:-webshop}"
# $1: 输出目录名 (自定义，如 baseline-gpt-4o / grpo-gpt-5)
AGENT_MODEL_PATH="${1:-Qwen3-8B}"
# $2: API Key (EMPTY / EMPTY=本地vLLM)
API_KEY="${2:-EMPTY}"
# $3: API Base URL (本地 vLLM 地址，OpenAI API 忽略此参数)
API_BASE_URL="${3:-http://localhost:8000/v1}"
# $4: Agent 模型名 (OpenAI API模型名如 gpt-4o-20241120，本地默认 vllm_agent)
# $5: 并发数 (本地默认64，OpenAI API默认2)
# $6: 温度 (默认0，确定性输出)
if [ "$API_KEY" = "EMPTY" ]; then
    AGENT_MODEL="vllm_agent"
    DEFAULT_CONCURRENCY=64             # 本地 vLLM: 2卡A100推荐100，4卡64
else
    AGENT_MODEL="${4:-gpt-4o-20241120}"
    DEFAULT_CONCURRENCY=2              # OpenAI API S0 限流，推荐2-4
fi
MAX_CONCURRENCY="${5:-$DEFAULT_CONCURRENCY}"
TEMPERATURE="${6:-0}"
# $7: 是否开启思考模式 (true/false, 默认false)
# 开启: Qwen3 Best Practices (temp=0.6, top_p=0.95, top_k=20, min_p=0, max_tokens=32768)
# 关闭: 确定性输出 (temp=0, top_p=1, max_tokens=4096)
ENABLE_THINKING="${7:-false}"
WM_PORT="${WM_PORT:-8001}"               # 可通过环境变量覆盖
WM_NAME="llm_world_model"               # vLLM WM 服务模型名
MAX_STEPS=50
OUTPUT_ROOT="$PROJECT_ROOT/outputs/task_success_rate"

# Test data - use init_contexts for agent/wm system prompts (based on TASK)
AGENT_INSTRUCT_FILE="$PROJECT_ROOT/data/init_contexts/$TASK/agent_instruct_test.json"
WM_INSTRUCT_FILE="$PROJECT_ROOT/data/init_contexts/$TASK/wm_instruct_test.json"

# Output directory - use actual model path for naming
OUTPUT_DIR="$OUTPUT_ROOT/wm/$TASK/$(basename $AGENT_MODEL_PATH)"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Task Success Rate - World Model"
echo "=========================================="
echo "Task:              $TASK"
echo "Agent Model:       $AGENT_MODEL"
echo "Output Dir Name:   $(basename $AGENT_MODEL_PATH)"
echo "API Key:           ${API_KEY:0:10}..."
echo "Temperature:       $TEMPERATURE"
echo "Thinking Mode:     $ENABLE_THINKING"
echo "Max Concurrency:   $MAX_CONCURRENCY"
echo "WM Service:        $WM_NAME"
echo "WM Port:           $WM_PORT"
echo "Output:            $OUTPUT_DIR"
echo "=========================================="

# Check if WM server is running
if ! curl -s "http://localhost:$WM_PORT/v1/models" > /dev/null; then
    echo "Error: World Model server not running on port $WM_PORT"
    echo "Please start it first: bash scripts/servers/start_wm_server.sh"
    exit 1
fi

# === Run Evaluation ===
python "$SCRIPT_DIR/interact_with_wm.py" \
    --task "$TASK" \
    --model "$AGENT_MODEL" \
    --api-key "$API_KEY" \
    --api-base-url "$API_BASE_URL" \
    --wm-port $WM_PORT \
    --wm-name "$WM_NAME" \
    --max-steps $MAX_STEPS \
    --max-concurrency $MAX_CONCURRENCY \
    --agent-instruct-file "$AGENT_INSTRUCT_FILE" \
    --wm-instruct-file "$WM_INSTRUCT_FILE" \
    --temperature $TEMPERATURE \
    $([ "$ENABLE_THINKING" = "true" ] && echo "--enable-thinking") \
    --output-root "$OUTPUT_DIR"

echo "=========================================="
echo "Evaluation Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
