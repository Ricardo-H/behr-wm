#!/bin/bash
#
# Queue evaluation for World Model Task Success Rate
#
# Usage:
#   bash run_wm_queue.sh
#
# Prerequisites:
#   - Agent (Qwen3-8B, nothink) already running on GPU 0,1,2,3, port 8000
#   - GPUs 4,5,6,7 available for WM deployment
#
# GPU Allocation (8x A100):
#   Agent: GPU 0,1,2,3 (TP=4, Qwen3-8B, 32 heads, 32%4=0 ✓)
#   WM:    GPU 4,5,6,7 (TP=4, Qwen2.5-7B, 28 heads, 28%4=0 ✓)
#
# Nothink Mode:
#   - interact_with_wm.py: enable_thinking=False (via extra_body)
#   - interact_with_real.py (agent.py): enable_thinking=False (via extra_body)
#
# Evaluates for each WM checkpoint:
#   1. WM evaluation (2.2) - Agent in World Model
#   2. W2R replay  (2.3)  - Replay WM actions in Real Environment
#
# Models:
#   - Baseline: anonymous/SFT-WM-WebShop-Qwen2.5-7B
#   - HuggingFace collection anonymous/ws-wm-0221:
#     step-120, step-160, step-200, step-240, step-280, step-300
#

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Activate environment
cd "$PROJECT_ROOT"
if [ -f "$PROJECT_ROOT/uv_webshop/bin/activate" ]; then
    echo "Activating uv_webshop environment..."
    source "$PROJECT_ROOT/uv_webshop/bin/activate"
fi

# Configuration
TASK="webshop"
WM_PORT=8001
# Qwen2.5-7B: 28 attention heads → TP must divide 28 (1,2,4,7,14,28)
# 4 GPUs → TP=4 (28%4=0 ✓)
WM_GPUS="4,5,6,7"
AGENT_NAME="A_qwen3next-80b-a3b-thinking"

# HuggingFace model prefix and steps to evaluate
HF_PREFIX="anonymous/ws-wm-0221"
STEPS=(120 160 200 240 280 300)

# Baseline Configuration
BASELINE_MODEL="anonymous/SFT-WM-WebShop-Qwen2.5-7B"
BASELINE_OUTPUT="${AGENT_NAME}_baseline"

# W2R replay toggle (set to false to skip W2R after each WM eval)
RUN_W2R=true

echo "=========================================="
echo "Queue Evaluation - World Model Task Success Rate"
echo "=========================================="
echo "Task:       $TASK"
echo "HF Prefix:  $HF_PREFIX"
echo "WM Port:    $WM_PORT"
echo "WM GPUs:    $WM_GPUS (TP=4)"
echo "Agent:      $AGENT_NAME (port 8000, GPU 0,1,2,3, TP=4)"
echo "Baseline:   $BASELINE_MODEL"
echo "Steps:      ${STEPS[*]}"
echo "Run W2R:    $RUN_W2R"
echo "Nothink:    ✅ (enable_thinking=False in both WM & Real)"
echo "=========================================="

# Check if Agent is running on port 8000
echo "Checking Agent server on port 8000..."
if ! curl -s "http://localhost:8000/v1/models" > /dev/null 2>&1; then
    echo "Error: Agent server not running on port 8000"
    echo "Please start Qwen3-8B agent first (nothink, 4 GPUs):"
    echo "  bash scripts/servers/start_agent_server.sh -m Qwen/Qwen3-8B -p 8000 -gpu 0,1,2,3"
    exit 1
fi
echo "✅ Agent server is running"

# Function to stop WM server robustly using nvidia-smi
stop_wm_server() {
    echo "Stopping WM server..."
    
    # 1. Kill by Port (Head process)
    local port_pids=$(ss -tlnp "sport = :$WM_PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u)
    if [ -n "$port_pids" ]; then
        echo "Found head process on port $WM_PORT: $port_pids. Killing..."
        echo "$port_pids" | xargs -r kill -9 2>/dev/null || true
    fi

    # 2. Kill by GPU (Worker processes) - DIRECT KILL via nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        echo "Checking GPUs $WM_GPUS for lingering processes..."
        local gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "$WM_GPUS" 2>/dev/null)
        gpu_pids=$(echo "$gpu_pids" | tr -d '\r' | sort -u | tr '\n' ' ')
        
        if [ -n "$gpu_pids" ]; then
            echo "Found lingering GPU processes on IDs $WM_GPUS: $gpu_pids"
            echo "Force killing GPU processes..."
            echo "$gpu_pids" | xargs -r kill -9 2>/dev/null || true
        else
            echo "No lingering processes found on GPUs $WM_GPUS."
        fi
    fi
    
    # 3. Wait for port release
    local wait_count=0
    while ss -tlnp "sport = :$WM_PORT" 2>/dev/null | grep -q "LISTEN"; do
        echo "Waiting for port $WM_PORT to release..."
        sleep 1
        wait_count=$((wait_count + 1))
        if [ $wait_count -ge 10 ]; then
             ss -tlnp "sport = :$WM_PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u | xargs -r kill -9 2>/dev/null || true
        fi
    done
    
    # 4. GPU Memory Cooldown
    echo "Resources released. Cooling down for 3s..."
    sleep 3
}

# Function to start WM server and wait for it
start_wm_server() {
    local model_path=$1
    
    echo "Starting WM server with model: $model_path"
    echo "GPUs: $WM_GPUS, Port: $WM_PORT"
    
    cd "$PROJECT_ROOT"
    nohup bash "$PROJECT_ROOT/scripts/servers/start_wm_server.sh" \
        -m "$model_path" \
        -p "$WM_PORT" \
        -gpu "$WM_GPUS" \
        > "/tmp/wm_server_${WM_PORT}.log" 2>&1 &
    
    echo "Waiting for WM server to start..."
    local max_wait=600  # Increased to 10 mins for HF download if needed
    local waited=0
    while ! curl -s "http://localhost:$WM_PORT/v1/models" > /dev/null 2>&1; do
        sleep 5
        waited=$((waited + 5))
        if [ $waited -ge $max_wait ]; then
            echo "Error: WM server failed to start within ${max_wait}s"
            echo "Check log: /tmp/wm_server_${WM_PORT}.log"
            tail -20 "/tmp/wm_server_${WM_PORT}.log" 2>/dev/null || true
            return 1
        fi
        
        if ! ss -tlnp "sport = :$WM_PORT" 2>/dev/null | grep -q "LISTEN"; then
             if grep -q -E "Error|Exception|Traceback" "/tmp/wm_server_${WM_PORT}.log" 2>/dev/null; then
                 echo "Warning: Server process might have crashed. Log snippet:"
                 tail -5 "/tmp/wm_server_${WM_PORT}.log"
             fi
        fi
        
        echo -n "."
    done
    echo ""
    echo "✅ WM server is ready"
}

# Generic function to run a single evaluation task (WM + optional W2R)
run_eval_task() {
    local model_path=$1
    local output_name=$2
    
    echo ""
    echo "=========================================="
    echo "Evaluating: ${output_name}"
    echo "WM Model: $model_path"
    echo "=========================================="
    
    # Only check directory if it looks like a local path (starts with / or .)
    # Skip check for HF IDs (like anonymous/...)
    if [[ "$model_path" == /* ]] || [[ "$model_path" == ./* ]]; then
        if [ ! -d "$model_path" ]; then
            echo "Warning: Model directory not found: $model_path, skipping..."
            return 1
        fi
    fi
    
    # Cleanup and Start WM server
    stop_wm_server
    if ! start_wm_server "$model_path"; then
        echo "Error: Failed to start WM server for $output_name"
        stop_wm_server
        return 1
    fi
    
    # --- Step 2.2: WM Evaluation ---
    echo "Running WM evaluation..."
    if bash "$SCRIPT_DIR/run_wm.sh" "$output_name"; then
         echo "✅ WM Completed: ${output_name}"
    else
         echo "❌ WM Evaluation failed: ${output_name}"
    fi
    
    # Cleanup WM server (free GPU memory for other tasks)
    stop_wm_server
    
    # --- Step 2.3: W2R Replay ---
    if [ "$RUN_W2R" = true ]; then
        local wm_output_dir="$PROJECT_ROOT/outputs/task_success_rate/wm/$TASK/${output_name}"
        if [ -d "$wm_output_dir" ]; then
            echo "Running W2R replay for: ${output_name}..."
            if bash "$SCRIPT_DIR/run_wm2real.sh" "$wm_output_dir"; then
                echo "✅ W2R Completed: ${output_name}"
            else
                echo "❌ W2R Replay failed: ${output_name}"
            fi
        else
            echo "⚠️ WM output dir not found, skipping W2R: $wm_output_dir"
        fi
    fi
}

# ---------------------------------------------------------
# 1. Evaluate Baseline FIRST (skip if SKIP_BASELINE=true)
# ---------------------------------------------------------
SKIP_BASELINE="${SKIP_BASELINE:-false}"
if [ "$SKIP_BASELINE" = true ]; then
    echo "⏭️ Skipping baseline evaluation (SKIP_BASELINE=true)"
else
    run_eval_task "$BASELINE_MODEL" "$BASELINE_OUTPUT"
fi

# ---------------------------------------------------------
# 2. Evaluate HuggingFace Checkpoint Steps
# ---------------------------------------------------------
for STEP in "${STEPS[@]}"; do
    MODEL_ID="${HF_PREFIX}-step-${STEP}"
    OUTPUT_NAME="${AGENT_NAME}_ws-wm-0221-step-${STEP}"
    run_eval_task "$MODEL_ID" "$OUTPUT_NAME"
done

echo ""
echo "=========================================="
echo "🎉 All evaluations completed!"
echo "WM results:  $PROJECT_ROOT/outputs/task_success_rate/wm/$TASK/"
echo "W2R results: $PROJECT_ROOT/outputs/task_success_rate/wm/$TASK/*/valid_on_real_env/"
echo "=========================================="

# ---------------------------------------------------------
# 3. Restart GPU placeholder to keep GPUs occupied
# ---------------------------------------------------------
echo "Restarting GPU placeholder script on all 8 GPUs..."
# Kill any existing placeholder first
pkill -f "gpu_placeholder.py" 2>/dev/null || true
sleep 2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python "$PROJECT_ROOT/scripts/gpu_placeholder.py" \
    > /tmp/gpu_placeholder.log 2>&1 &
echo "✅ GPU placeholder started (pid=$!, log: /tmp/gpu_placeholder.log)"