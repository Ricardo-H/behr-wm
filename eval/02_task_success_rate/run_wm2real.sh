#!/bin/bash
# =============================================================================
# Task Success Rate Evaluation - WM to Real (Replay)
# =============================================================================
#
# Usage:
#   TASK=webshop   bash run_wm2real.sh <wm_output_dir>
#   TASK=textworld bash run_wm2real.sh <wm_output_dir>
#
# Supported tasks: webshop, textworld, alfworld, sciworld
# If TASK is not set, it defaults to "webshop".
#
# This script replays actions from world model in real environment.
# =============================================================================

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Activate environment
if [ -f "$PROJECT_ROOT/uv_webshop/bin/activate" ]; then
    source "$PROJECT_ROOT/uv_webshop/bin/activate"
fi

# === Configuration ===
TASK="${TASK:-webshop}"
WM_OUTPUT_DIR="${1:-$PROJECT_ROOT/outputs/task_success_rate/wm/$TASK}"
MAX_WORKERS=50

echo "=========================================="
echo "Task Success Rate - WM to Real (Replay)"
echo "=========================================="
echo "Task:           $TASK"
echo "WM Output Dir:  $WM_OUTPUT_DIR"
echo "=========================================="

# === Start Environment Server ===
ENV_PORT=$((30000 + RANDOM % (99999-30000+1)))

start_env_server() {
    case "$TASK" in
        webshop)
            webshop --host 0.0.0.0 --port $ENV_PORT >/tmp/${TASK}_server_${ENV_PORT}.log 2>&1 &
            SERVER_PID=$!
            ;;
        textworld)
            python -m agentenv.envs.textworld_server --host 0.0.0.0 --port $ENV_PORT \
                --games_dir "$PROJECT_ROOT/data/textworld/games" \
                >/tmp/${TASK}_server_${ENV_PORT}.log 2>&1 &
            SERVER_PID=$!
            ;;
        alfworld|alfworld_valid_seen|alfworld_valid_unseen)
            python -m agentenv.envs.alfworld_server --host 0.0.0.0 --port $ENV_PORT \
                >/tmp/${TASK}_server_${ENV_PORT}.log 2>&1 &
            SERVER_PID=$!
            ;;
        sciworld)
            python -m agentenv.envs.sciworld_server --host 0.0.0.0 --port $ENV_PORT \
                >/tmp/${TASK}_server_${ENV_PORT}.log 2>&1 &
            SERVER_PID=$!
            ;;
        *)
            echo "Error: Unknown task '$TASK'. Supported: webshop, textworld, alfworld, sciworld"
            exit 1
            ;;
    esac
}

start_env_server
trap "kill $SERVER_PID 2>/dev/null || true" EXIT INT TERM

echo "Launching $TASK server... (pid=$SERVER_PID, port=$ENV_PORT)"

# Wait for server to be ready (up to 120s)
MAX_WAIT=120
WAITED=0
echo -n "Waiting for $TASK server to be ready"
while ! curl -s "http://localhost:$ENV_PORT" > /dev/null 2>&1; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo ""
        echo "Error: $TASK server process died. Check /tmp/${TASK}_server_${ENV_PORT}.log"
        tail -20 /tmp/${TASK}_server_${ENV_PORT}.log
        exit 1
    fi
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo ""
        echo "Error: $TASK server not ready after ${MAX_WAIT}s. Check /tmp/${TASK}_server_${ENV_PORT}.log"
        tail -20 /tmp/${TASK}_server_${ENV_PORT}.log
        exit 1
    fi
    echo -n "."
    sleep 5
    WAITED=$((WAITED + 5))
done
echo ""
echo "$TASK server is running on port $ENV_PORT (took ~${WAITED}s)"

# === Run Replay ===
python "$SCRIPT_DIR/cal_wm2real.py" \
    --task "$TASK" \
    --test_file_root "$WM_OUTPUT_DIR" \
    --port $ENV_PORT \
    --max_workers $MAX_WORKERS

echo "=========================================="
echo "Evaluation Complete!"
echo "Results saved to: $WM_OUTPUT_DIR/valid_on_real_env/"
echo "=========================================="
