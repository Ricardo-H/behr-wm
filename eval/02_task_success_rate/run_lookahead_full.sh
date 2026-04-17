#!/bin/bash
# =============================================================================
# TextWorld Lookahead Full Experiment
# =============================================================================
# Runs 3 conditions: No Planning, Lookahead+SFT-WM, Lookahead+GRPO-WM
# All 200 test samples, Qwen3-8B agent (no thinking, temp=0)
# =============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/.venv/bin/activate"

# Config
AGENT_PORT=8010  # Qwen3-8B
ENV_PORT=8024    # TextWorld real env server
N_SAMPLES=200
MAX_STEPS=50
LOOKAHEAD_K=5

AGENT_INSTRUCT="$PROJECT_ROOT/data/init_contexts/textworld/agent_instruct_test.json"
WM_INSTRUCT="$PROJECT_ROOT/data/init_contexts/textworld/wm_instruct_test.json"
INFERENCE_FILE="$PROJECT_ROOT/data/eval/textworld_test.json"
OUTPUT_BASE="$PROJECT_ROOT/outputs/lookahead_textworld"

echo "=============================================="
echo "TextWorld Lookahead Full Experiment"
echo "$(date)"
echo "=============================================="
echo "Agent: Qwen3-8B (port $AGENT_PORT)"
echo "Env: TextWorld (port $ENV_PORT)"
echo "Samples: $N_SAMPLES, Max Steps: $MAX_STEPS, K: $LOOKAHEAD_K"
echo "=============================================="

# -------- Condition 1: No Planning --------
echo ""
echo ">>>>>>>>>> [1/3] No Planning Baseline <<<<<<<<<<"
echo "Start: $(date)"
NO_PLAN_DIR="$OUTPUT_BASE/no_planning_8b"
mkdir -p "$NO_PLAN_DIR"

python "$SCRIPT_DIR/interact_with_real.py" \
    --api_key EMPTY \
    --base_url "http://localhost:$AGENT_PORT/v1" \
    --model vllm_agent_8b \
    --task_name textworld \
    --env_server_base "http://localhost:$ENV_PORT" \
    --inference_file "$INFERENCE_FILE" \
    --output_dir "$NO_PLAN_DIR" \
    --max_round $MAX_STEPS \
    --temperature 0 \
    --max_concurrency 5 \
    --num_examples $N_SAMPLES

echo "[1/3] No Planning complete at $(date)"
echo ""

# -------- Condition 2: Lookahead + SFT-WM (port 8001) --------
echo ">>>>>>>>>> [2/3] Lookahead + SFT-WM <<<<<<<<<<"
echo "Start: $(date)"
LA_SFT_DIR="$OUTPUT_BASE/lookahead_sft_wm"
mkdir -p "$LA_SFT_DIR"

python "$SCRIPT_DIR/interact_with_lookahead_textworld.py" \
    --agent-model vllm_agent_8b \
    --api-key EMPTY \
    --api-base-url "http://localhost:$AGENT_PORT/v1" \
    --wm-port 8001 \
    --wm-name llm_world_model \
    --env-port $ENV_PORT \
    --agent-instruct-file "$AGENT_INSTRUCT" \
    --wm-instruct-file "$WM_INSTRUCT" \
    --output-root "$LA_SFT_DIR" \
    --max-steps $MAX_STEPS \
    --n-samples $N_SAMPLES \
    --temperature 0 \
    --lookahead-k $LOOKAHEAD_K

echo "[2/3] Lookahead + SFT-WM complete at $(date)"
echo ""

# -------- Condition 3: Lookahead + GRPO-220 WM (port 8003) --------
echo ">>>>>>>>>> [3/3] Lookahead + GRPO-220 WM <<<<<<<<<<"
echo "Start: $(date)"
LA_GRPO_DIR="$OUTPUT_BASE/lookahead_grpo220_wm"
mkdir -p "$LA_GRPO_DIR"

python "$SCRIPT_DIR/interact_with_lookahead_textworld.py" \
    --agent-model vllm_agent_8b \
    --api-key EMPTY \
    --api-base-url "http://localhost:$AGENT_PORT/v1" \
    --wm-port 8003 \
    --wm-name llm_world_model \
    --env-port $ENV_PORT \
    --agent-instruct-file "$AGENT_INSTRUCT" \
    --wm-instruct-file "$WM_INSTRUCT" \
    --output-root "$LA_GRPO_DIR" \
    --max-steps $MAX_STEPS \
    --n-samples $N_SAMPLES \
    --temperature 0 \
    --lookahead-k $LOOKAHEAD_K

echo "[3/3] Lookahead + GRPO-220 WM complete at $(date)"
echo ""

# -------- Summary --------
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE - $(date)"
echo "=============================================="
for DIR in "$NO_PLAN_DIR" "$LA_SFT_DIR" "$LA_GRPO_DIR"; do
    METRICS="$DIR/_metrics.json"
    if [ -f "$METRICS" ]; then
        echo "  $(basename $DIR): $(python3 -c "import json; d=json.load(open('$METRICS')); print(f'SR={d.get(\"accuracy\",0):.2f}%')")"
    else
        echo "  $(basename $DIR): (no metrics)"
    fi
done
