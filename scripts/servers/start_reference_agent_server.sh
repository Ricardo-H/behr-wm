#!/bin/bash
# =============================================================================
# Reference Agent vLLM Server (for GRPO Training & Behavior Consistency Evaluation)
# =============================================================================
#
# Usage:
#   bash start_reference_agent_server.sh -m <model> [-p <port>] [-gpu <gpus>] [-l <max_len>]
#   bash start_reference_agent_server.sh -m <model> [-p <port>] [-gpu <gpus>] [-l <max_len>] [--shared] [--util <0-1>]
#
# Options:
#   -m,   --model    Model path or HuggingFace ID (required)
#   -p,   --port     Server port (default: 8000)
#   -gpu, --gpu      GPU IDs, comma-separated (default: auto-detect)
#   -l,   --len      Max model length (default: 36864)
#         --shared   Shared-GPU mode for training (adds --enforce-eager)
#         --util     vLLM gpu_memory_utilization (default: 0.15)
#   -h,   --help     Show this help message
#
# Examples:
#   bash start_reference_agent_server.sh -m Qwen/Qwen3-30B-A3B -gpu 3
#   bash start_reference_agent_server.sh -m Qwen/Qwen3-8B -p 8000 -gpu 0,1
#   bash start_reference_agent_server.sh --model Qwen/Qwen3-30B-A3B --gpu 2,3 --len 12288
#
# Note:
#   - Tensor parallel size is auto-calculated from GPU count
#   - Qwen3 MoE models (30B-A3B) require VLLM_USE_V1=1
# =============================================================================

set -e

# Show help
show_help() {
    sed -n '2,/^# =====/p' "$0" | head -n -1 | sed 's/^# //' | sed 's/^#//'
    exit 0
}

# Default values
MODEL=""
PORT="8000"
GPUS=""
MAX_MODEL_LEN="32768"
HOST="0.0.0.0"
GPU_MEM_UTIL="0.05"
SHARED_MODE="false"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -gpu|--gpu)
            GPUS="$2"
            shift 2
            ;;
        -l|--len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --shared)
            SHARED_MODE="true"
            shift
            ;;
        --util)
            GPU_MEM_UTIL="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# shared mode 只添加 --enforce-eager，不再修改 gpu_memory_utilization
# 默认 0.15 已经足够处理长 context（~90K chars）

# Check required arguments
if [ -z "$MODEL" ]; then
    echo "Error: Model path is required"
    echo "Usage: bash start_reference_agent_server.sh -m <model> [-p <port>] [-gpu <gpus>] [-l <max_len>]"
    echo "Use -h or --help for more information"
    exit 1
fi

# Activate virtual environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"
source "${PROJECT_ROOT}/.venv/bin/activate"

# GPU configuration
if [ -z "$GPUS" ]; then
    # Auto-detect: use first 2 GPUs if available, otherwise first 1
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    if [ $NUM_GPUS -gt 1 ]; then
        GPUS="0,1"
        TP_SIZE=2
    else
        GPUS="0"
        TP_SIZE=1
    fi
else
    # Count GPUs from comma-separated list
    TP_SIZE=$(echo "$GPUS" | tr ',' '\n' | wc -l)
fi

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES="$GPUS"

# HuggingFace cache
export HF_HOME="${HF_HOME:-~/.cache/huggingface}"

# Enable V1 engine for Qwen3 MoE models
if [[ "$MODEL" == *"30B-A3B"* ]] || [[ "$MODEL" == *"MoE"* ]]; then
    export VLLM_USE_V1=1
    echo "Detected MoE model, enabling VLLM_USE_V1=1"
fi

echo "=========================================="
echo "Starting Reference Agent Server"
echo "=========================================="
echo "Model:           $MODEL"
echo "Port:            $PORT"
echo "GPUs:            $GPUS (CUDA_VISIBLE_DEVICES)"
echo "Tensor Parallel: $TP_SIZE"
echo "Max Model Len:   ${MAX_MODEL_LEN:-unlimited}"
echo "Shared Mode:     $SHARED_MODE"
echo "GPU Mem Util:    $GPU_MEM_UTIL"
echo "=========================================="

# Build vllm command
VLLM_CMD="vllm serve \"$MODEL\" \
    --port $PORT \
    --host $HOST \
    --served-model-name vllm_reference_agent \
    --tensor-parallel-size $TP_SIZE \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --dtype bfloat16 \
    --trust-remote-code \
    --enable-prefix-caching \
    --disable-log-requests \
    --max-num-seqs 16"

# Shared mode: reduce memory overhead / fragmentation
if [[ "$SHARED_MODE" == "true" ]]; then
    VLLM_CMD="$VLLM_CMD --enforce-eager"
fi

# Add max-model-len only if specified
if [ -n "$MAX_MODEL_LEN" ]; then
    VLLM_CMD="$VLLM_CMD --max-model-len $MAX_MODEL_LEN"
fi

eval $VLLM_CMD
