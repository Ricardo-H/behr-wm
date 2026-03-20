#!/bin/bash
# =============================================================================
# Agent Model vLLM Server
# =============================================================================
#
# Usage:
#   bash start_agent_server.sh -m <model> [-p <port>] [-gpu <gpus>] [-l <max_len>]
#
# Options:
#   -m,   --model    Model path or HuggingFace ID (required)
#   -p,   --port     Server port (default: 8000)
#   -gpu, --gpu      GPU IDs, comma-separated (default: auto-detect)
#   -l,   --len      Max model length (default: unlimited)
#   -h,   --help     Show this help message
#
# Examples:
#   bash start_agent_server.sh -m Qwen/Qwen3-8B
#   bash start_agent_server.sh -m Qwen/Qwen3-8B -p 8000 -gpu 0,1
#   bash start_agent_server.sh -m Qwen/Qwen3-8B -gpu 4,5,6,7 -l 32768
#   bash start_agent_server.sh --model Qwen/Qwen3-8B --port 8000 --gpu 2,3
#
# Note:
#   - Tensor parallel size is auto-calculated from GPU count
#   - If no GPU specified, auto-detect available GPUs
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
# 【修改点1】默认设为 65536 (64k)
MAX_MODEL_LEN="65536"  
HOST="0.0.0.0"

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

# Check required arguments
if [ -z "$MODEL" ]; then
    echo "Error: Model path is required"
    echo "Usage: bash start_agent_server.sh -m <model> [-p <port>] [-gpu <gpus>] [-l <max_len>]"
    echo "Use -h or --help for more information"
    exit 1
fi

source uv_webshop/bin/activate

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

echo "=========================================="
echo "Starting Agent Model Server"
echo "=========================================="
echo "Model:           $MODEL"
echo "Port:            $PORT"
echo "GPUs:            $GPUS (CUDA_VISIBLE_DEVICES)"
echo "Tensor Parallel: $TP_SIZE"
echo "Max Model Len:   ${MAX_MODEL_LEN:-unlimited}"
echo "=========================================="

# Build vllm command
VLLM_CMD="vllm serve \"$MODEL\" \
    --port $PORT \
    --host $HOST \
    --served-model-name vllm_agent \
    --tensor-parallel-size $TP_SIZE \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    --trust-remote-code \
    --disable-log-requests \
    --enable-prefix-caching"

# 【修改点2】智能添加长文本支持 (YaRN)
if [ -n "$MAX_MODEL_LEN" ]; then
    VLLM_CMD="$VLLM_CMD --max-model-len $MAX_MODEL_LEN"

    # 逻辑：如果长度超过模型原生支持的 max_position_embeddings，则需要启用 YaRN
    # Qwen3 原生支持 40960, Qwen2.5 原生支持 32768
    if [ "$MAX_MODEL_LEN" -gt 32768 ]; then
        # 检查模型是否已原生支持该长度 (通过 max_position_embeddings)
        NATIVE_MAX=$(python3 -c "
from transformers import AutoConfig
try:
    cfg = AutoConfig.from_pretrained('$MODEL', trust_remote_code=True)
    print(getattr(cfg, 'max_position_embeddings', 32768))
except:
    print(32768)
" 2>/dev/null || echo "32768")
        
        if [ "$MAX_MODEL_LEN" -gt "$NATIVE_MAX" ]; then
            echo "Detected long context ($MAX_MODEL_LEN > native $NATIVE_MAX). Enabling YaRN scaling..."
            FACTOR=2.0
            if [ "$MAX_MODEL_LEN" -gt 65536 ]; then
                FACTOR=4.0
            fi
            ROPE_JSON="{\"rope_type\":\"yarn\",\"factor\":$FACTOR,\"original_max_position_embeddings\":32768}"
            VLLM_CMD="$VLLM_CMD --rope-scaling '$ROPE_JSON'"
        else
            echo "Model natively supports $MAX_MODEL_LEN tokens (max_position_embeddings=$NATIVE_MAX). Skipping YaRN."
        fi
    fi
fi

echo "Executing: $VLLM_CMD"
eval $VLLM_CMD
