#!/bin/bash
# =============================================================================
# World Model vLLM Server (Qwen2.5 YaRN Support)
# =============================================================================
#
# Usage:
#   bash start_wm_server.sh -m <model> [-p <port>] [-gpu <gpus>] [-l <max_len>]
#   bash scripts/servers/start_wm_server.sh -m anonymous/BehR-WM-WebShop-Qwen2.5-7B -p 8001 -gpu 2,3
# =============================================================================

set -e

# Show help
show_help() {
    sed -n '2,/^# =====/p' "$0" | head -n -1 | sed 's/^# //' | sed 's/^#//'
    exit 0
}

# Default values
MODEL=""
PORT="8001"
GPUS=""
# 【修改点1】默认设置为 65536 (64k)
MAX_MODEL_LEN="65536"
GPU_MEM_UTIL="0.90"
MAX_NUM_SEQS=""  # Auto-scale based on TP size if not set
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
        -mem|--gpu-mem-util)
            GPU_MEM_UTIL="$2"
            shift 2
            ;;
        -s|--max-num-seqs)
            MAX_NUM_SEQS="$2"
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
export HF_HOME="${HF_HOME:-~/.cache/huggingface}"

echo "=========================================="
echo "Starting World Model Server"
echo "=========================================="
echo "Model:           $MODEL"
echo "Port:            $PORT"
echo "GPUs:            $GPUS"
echo "Tensor Parallel: $TP_SIZE"
echo "Max Model Len:   ${MAX_MODEL_LEN:-unlimited}"
# Auto-scale max-num-seqs based on TP size if not explicitly set
if [ -z "$MAX_NUM_SEQS" ]; then
    MAX_NUM_SEQS=$((64 * TP_SIZE / 2))  # Scale: TP=2→64, TP=4→128
    [ "$MAX_NUM_SEQS" -lt 64 ] && MAX_NUM_SEQS=64
fi
echo "GPU Mem Util:    $GPU_MEM_UTIL"
echo "Max Num Seqs:    $MAX_NUM_SEQS"
echo "=========================================="

# Build vllm command
# 【修改点2】删除了原脚本中硬编码的 --max-model-len 131072，防止冲突
VLLM_CMD="vllm serve \"$MODEL\" \
    --port $PORT \
    --host $HOST \
    --served-model-name llm_world_model \
    --tensor-parallel-size $TP_SIZE \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --dtype bfloat16 \
    --trust-remote-code \
    --disable-log-requests \
    --max-num-seqs $MAX_NUM_SEQS"

# 【修改点3】动态添加长度限制和 RoPE Scaling (YaRN)
if [ -n "$MAX_MODEL_LEN" ]; then
    VLLM_CMD="$VLLM_CMD --max-model-len $MAX_MODEL_LEN"

    # Qwen2.5 官方逻辑：超过 32768 时开启 YaRN
    if [ "$MAX_MODEL_LEN" -gt 32768 ]; then
        echo "Detected long context ($MAX_MODEL_LEN > 32768). Enabling YaRN scaling..."
        
        # 计算 Factor
        # 目标 64k -> Factor 2.0
        # 目标 128k -> Factor 4.0
        FACTOR=2.0
        if [ "$MAX_MODEL_LEN" -gt 65536 ]; then
            FACTOR=4.0
        fi
        
        # 构造 JSON (根据 Qwen2.5 文档，使用 "type": "yarn")
        ROPE_JSON="{\"type\":\"yarn\",\"factor\":$FACTOR,\"original_max_position_embeddings\":32768}"
        
        VLLM_CMD="$VLLM_CMD --rope-scaling '$ROPE_JSON'"
    fi
fi

echo "Executing: $VLLM_CMD"
eval $VLLM_CMD
