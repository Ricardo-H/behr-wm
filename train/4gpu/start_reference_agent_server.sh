#!/bin/bash
#
# 启动 Qwen3-8B Reference Agent Server (TP=4, 与训练共享)
#
# 与 4 卡 GRPO 训练共存，每卡占用 ~10% 显存
#
# 使用方法:
#   bash start_reference_agent_server.sh
#
# 停止服务:
#   pkill -f "vllm serve.*Qwen3-8B"
#

set -x

# 激活环境 (子目录需要 /../.. 到项目根目录)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"
source "${PROJECT_ROOT}/uv_webshop/bin/activate"

# Reference Agent 模型配置
REF_AGENT_MODEL="Qwen/Qwen3-8B"
PORT=8000
TP_SIZE=4

# 使用 4 卡
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "=========================================="
echo "Starting Reference Agent Server (4-GPU Shared)"
echo "Model: $REF_AGENT_MODEL"
echo "Port: $PORT"
echo "Tensor Parallel: $TP_SIZE"
echo "=========================================="

# 检查端口是否已被占用
if lsof -i :$PORT > /dev/null 2>&1; then
    echo "Port $PORT is already in use. Checking if it's the reference agent server..."
    curl -s http://localhost:$PORT/health && echo "Reference Agent server is already running." && exit 0
    echo "Port occupied by another service. Please free it first."
    exit 1
fi

# 启动 vLLM 服务
# - tensor_parallel_size=4: 分布在 4 张卡
# - gpu_memory_utilization=0.10: 每卡用 10% 显存 (~8GB)，与训练共存
# - enforce-eager: 禁用 CUDA Graph，减少显存预分配
python -m vllm.entrypoints.openai.api_server \
    --model ${REF_AGENT_MODEL} \
    --tensor-parallel-size ${TP_SIZE} \
    --gpu-memory-utilization 0.12 \
    --enforce-eager \
    --max-num-seqs 128 \
    --max-model-len 32768 \
    --port ${PORT} \
    --trust-remote-code \
    --enable-prefix-caching \
    --disable-log-requests \
    2>&1 | tee "${PROJECT_ROOT}/reference_agent_server.log"
