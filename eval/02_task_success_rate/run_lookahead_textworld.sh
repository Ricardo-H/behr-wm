#!/bin/bash
# =============================================================================
# TextWorld Lookahead Experiment
# =============================================================================
#
# Compares: No Planning vs Lookahead(SFT-WM) vs Lookahead(GRPO-WM)
# Agent: Qwen3-8B (no thinking, temp=0)
# WMs: SFT baseline (port 8001), GRPO step_220 (port 8003, best CR)
#
# Prerequisites:
#   - WM servers already running on ports 8001 (SFT), 8003 (GRPO-220)
#   - GPU 6 or 7 free for Qwen3-8B agent
#
# Usage:
#   bash eval/02_task_success_rate/run_lookahead_textworld.sh
# =============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

# Activate environment
source "$PROJECT_ROOT/.venv/bin/activate"

# === Configuration ===
AGENT_MODEL_NAME="Qwen/Qwen3-8B"
AGENT_PORT=8010
AGENT_GPU=6  # Use GPU 6 (currently only placeholder)
ENV_PORT=8024
N_SAMPLES=200  # All 200 test samples
MAX_STEPS=50   # TextWorld max steps

AGENT_INSTRUCT="$PROJECT_ROOT/data/init_contexts/textworld/agent_instruct_test.json"
WM_INSTRUCT="$PROJECT_ROOT/data/init_contexts/textworld/wm_instruct_test.json"

OUTPUT_BASE="$PROJECT_ROOT/outputs/lookahead_textworld"

echo "=============================================="
echo "TextWorld Lookahead Experiment"
echo "=============================================="
echo "Agent:        $AGENT_MODEL_NAME (GPU $AGENT_GPU, port $AGENT_PORT)"
echo "Env Port:     $ENV_PORT"
echo "Samples:      $N_SAMPLES"
echo "Max Steps:    $MAX_STEPS"
echo "Output:       $OUTPUT_BASE"
echo "=============================================="

# === Step 1: Start Qwen3-8B Agent (if not already running) ===
if curl -s "http://localhost:$AGENT_PORT/v1/models" > /dev/null 2>&1; then
    echo "[Agent] Already running on port $AGENT_PORT"
else
    echo "[Agent] Starting Qwen3-8B on GPU $AGENT_GPU..."
    CUDA_VISIBLE_DEVICES=$AGENT_GPU python3 -m vllm.entrypoints.openai.api_server \
        --model "$AGENT_MODEL_NAME" \
        --port $AGENT_PORT \
        --served-model-name vllm_agent_8b \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.90 \
        --dtype bfloat16 \
        --max-model-len 32768 \
        --trust-remote-code \
        --disable-log-requests \
        --enable-prefix-caching \
        > /tmp/agent_8b_${AGENT_PORT}.log 2>&1 &
    AGENT_PID=$!
    echo "[Agent] PID=$AGENT_PID, waiting for startup..."
    
    MAX_WAIT=180
    WAITED=0
    while ! curl -s "http://localhost:$AGENT_PORT/v1/models" > /dev/null 2>&1; do
        if ! kill -0 $AGENT_PID 2>/dev/null; then
            echo "[Agent] FAILED! Check /tmp/agent_8b_${AGENT_PORT}.log"
            tail -20 /tmp/agent_8b_${AGENT_PORT}.log
            exit 1
        fi
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo "[Agent] Timeout after ${MAX_WAIT}s"
            exit 1
        fi
        sleep 5
        WAITED=$((WAITED + 5))
    done
    echo "[Agent] Ready! (took ~${WAITED}s)"
fi

# === Step 2: Start TextWorld Real Environment Server ===
if curl -s "http://localhost:$ENV_PORT" > /dev/null 2>&1; then
    echo "[Env] TextWorld server already running on port $ENV_PORT"
else
    echo "[Env] Starting TextWorld server on port $ENV_PORT..."
    python -m agentenv.envs.textworld_server \
        --host 0.0.0.0 --port $ENV_PORT \
        --games_dir "$PROJECT_ROOT/data/textworld/games" \
        > /tmp/textworld_server_${ENV_PORT}.log 2>&1 &
    ENV_PID=$!
    
    MAX_WAIT=60
    WAITED=0
    while ! curl -s "http://localhost:$ENV_PORT" > /dev/null 2>&1; do
        if ! kill -0 $ENV_PID 2>/dev/null; then
            echo "[Env] FAILED! Check /tmp/textworld_server_${ENV_PORT}.log"
            tail -20 /tmp/textworld_server_${ENV_PORT}.log
            exit 1
        fi
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo "[Env] Timeout after ${MAX_WAIT}s"
            exit 1
        fi
        sleep 3
        WAITED=$((WAITED + 3))
    done
    echo "[Env] Ready! (took ~${WAITED}s)"
fi

echo ""
echo "===================== All services ready ====================="
echo ""

# === Step 3: Run No Planning baseline (Agent → Real Env directly) ===
echo ">>> [1/3] Running No Planning baseline..."
NO_PLAN_DIR="$OUTPUT_BASE/no_planning_8b"

python "$SCRIPT_DIR/interact_with_real.py" \
    --api_key EMPTY \
    --base_url "http://localhost:$AGENT_PORT/v1" \
    --model vllm_agent_8b \
    --task_name textworld \
    --env_server_base "http://localhost:$ENV_PORT" \
    --inference_file "$PROJECT_ROOT/data/eval/textworld_test.json" \
    --output_dir "$NO_PLAN_DIR" \
    --max_round $MAX_STEPS \
    --temperature 0 \
    --max_concurrency 1 \
    --num_examples $N_SAMPLES

echo ">>> No Planning complete!"
echo ""

# === Step 4: Run Lookahead with SFT baseline WM (port 8001) ===
echo ">>> [2/3] Running Lookahead with SFT-WM (port 8001)..."
LA_SFT_DIR="$OUTPUT_BASE/lookahead_sft_wm"

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
    --temperature 0

echo ">>> Lookahead (SFT-WM) complete!"
echo ""

# === Step 5: Run Lookahead with GRPO step_220 WM (port 8003, best CR) ===
echo ">>> [3/3] Running Lookahead with GRPO-220 WM (port 8003)..."
LA_GRPO_DIR="$OUTPUT_BASE/lookahead_grpo220_wm"

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
    --temperature 0

echo ">>> Lookahead (GRPO-220 WM) complete!"
echo ""

# === Summary ===
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=============================================="
echo ""
echo "Results:"
for DIR in "$NO_PLAN_DIR" "$LA_SFT_DIR" "$LA_GRPO_DIR"; do
    METRICS="$DIR/_metrics.json"
    if [ -f "$METRICS" ]; then
        echo "  $(basename $DIR): $(cat $METRICS | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"SR={d.get(\"accuracy\",0):.2f}% ({d.get(\"success\",d.get(\"total_success\",0))}/{d.get(\"total\",d.get(\"processed_items\",0))})")')"
    else
        echo "  $(basename $DIR): (metrics not found)"
    fi
done
echo ""
echo "Output: $OUTPUT_BASE"
