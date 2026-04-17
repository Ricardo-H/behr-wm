#!/bin/bash
# =============================================================================
# Task Success Rate Evaluation - Real Environment
# =============================================================================
#
# Usage:
#   bash run_real.sh [options]
#
# This script evaluates agent performance in the real WebShop environment.
# =============================================================================

set -x

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# source "$PROJECT_ROOT/.venv/bin/activate"

# === Configuration ===
TASK="webshop"
# $1: 输出目录名 (自定义，如 baseline-gpt-4o / grpo-gpt-5)
MODEL_PATH="${1:-Qwen3-8B}"
# $2: Agent 模型名 (OpenAI API模型名如 gpt-4o-20241120，本地默认 vllm_agent)
MODEL="${2:-vllm_agent}"
# $3: API Key (EMPTY / EMPTY=本地vLLM)
API_KEY="${3:-EMPTY}"
# $4: API Base URL (本地 vLLM 地址，OpenAI API 忽略此参数)
API_BASE_URL="${4:-http://localhost:8000/v1}"
# $5: 温度 (默认0，确定性输出)
TEMPERATURE="${5:-0}"
# $6: 并发数 (默认8)
MAX_CONCURRENCY="${6:-8}"
# $7: 最大输出token数 (默认4096，Qwen3思考模型推荒32768)
MAX_TOKENS="${7:-4096}"
# $8: top_p (默认1，Qwen3推荐0.95)
TOP_P="${8:-1}"
MAX_ROUND=20
NUM_EXAMPLES=-1
OUTPUT_ROOT="$PROJECT_ROOT/outputs/task_success_rate"

# Test data
INFERENCE_FILE="$PROJECT_ROOT/data/eval/webshop_test.json"

# Output directory - use actual model path for naming
OUTPUT_DIR="$OUTPUT_ROOT/real/$TASK/$(basename $MODEL_PATH)"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Task Success Rate - Real Environment"
echo "=========================================="
echo "Task:              $TASK"
echo "Agent Model:       $MODEL"
echo "Output Dir Name:   $(basename $MODEL_PATH)"
echo "API Key:           ${API_KEY:0:10}..."
echo "Temperature:       $TEMPERATURE"
echo "Max Concurrency:   $MAX_CONCURRENCY"
echo "Max Tokens:        $MAX_TOKENS"
echo "Output:            $OUTPUT_DIR"
echo "=========================================="

# === Start Environment Server ===
ENV_PORT=$((30000 + RANDOM % (99999-30000+1)))

# Activate WebShop environment
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

webshop --host 0.0.0.0 --port $ENV_PORT >/tmp/webshop_server_${ENV_PORT}.log 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null || true" EXIT INT TERM

echo "Launching WebShop server... (pid=$SERVER_PID, port=$ENV_PORT)"
sleep 10
echo "WebShop server is running on port $ENV_PORT"

# === Run Evaluation ===
python "$SCRIPT_DIR/interact_with_real.py" \
    --api_key "$API_KEY" \
    --base_url "$API_BASE_URL" \
    --model "$MODEL" \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --inference_file "$INFERENCE_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --max_round $MAX_ROUND \
    --max_concurrency $MAX_CONCURRENCY \
    --max_tokens $MAX_TOKENS \
    --num_examples $NUM_EXAMPLES \
    --task_name "$TASK" \
    --env_server_base "http://localhost:$ENV_PORT"

echo "=========================================="
echo "Evaluation Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
