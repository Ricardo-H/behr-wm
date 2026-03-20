#!/bin/bash
#
# Pivot-GRPO Training Script - 8 GPU 共享模式
#
# 架构: 8 卡 FSDP 训练 + vLLM TP=2 Rollout (共享显存)
# 
# 核心特点:
# 1. 8 卡全部参与 FSDP 训练 (Actor/Ref)
# 2. vLLM rollout 使用 TP=2，每 2 卡为一组
# 3. 训练和 Rollout 共享显存，交替运行
# 4. Reference Agent 复用训练卡 (TP=8, 每卡~5%显存)
#
# 适用场景:
# - 长上下文 (8k avg, 18k max)
# - 需要充分利用 8 卡训练吞吐
#
# 使用方法:
#   1. 先启动 Reference Agent: bash start_reference_agent_server.sh (可选)
#   2. 再启动训练: bash run_grpo.sh
#

set -x

# 激活虚拟环境 (子目录需要 /../.. 到项目根目录)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
FE_WORLD_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"
source "${FE_WORLD_ROOT}/uv_webshop/bin/activate"

# ============ 数据配置 ============
TRAIN_DATA="${FE_WORLD_ROOT}/data/train/hard_samples/train_stratified.parquet"
VAL_DATA="${FE_WORLD_ROOT}/data/train/hard_samples/test.parquet"

echo "=========================================="
echo "8-GPU GRPO Training (Shared Mode)"
echo "Train Data: $TRAIN_DATA"
echo "Val Data:   $VAL_DATA"
echo "=========================================="

if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Training data not found: $TRAIN_DATA"
    exit 1
fi

# ============ 模型配置 ============
WORLD_MODEL="anonymous/SFT-WM-WebShop-Qwen2.5-7B"
REWARD_FN_PATH="${FE_WORLD_ROOT}/train/reward_function.py"

# 输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M)
DATE_PREFIX=$(date +%m%d)  # 用于 HuggingFace Collection 命名 (如 0205)
OUTPUT_DIR="${FE_WORLD_ROOT}/outputs/checkpoints/8gpu_${TIMESTAMP}"
TOTAL_EPOCHS=5

# ============ HuggingFace 上传配置 ============
# 设置 HF_UPLOAD=true 来启用自动上传
# 注意：模型和 Collection 默认上传为公开仓库
HF_UPLOAD=${HF_UPLOAD:-true}
HF_USER=${HF_USER:-"anonymous"}
HF_COLLECTION_NAME=${HF_COLLECTION_NAME:-"ws-wm"}
HF_POLL_INTERVAL=${HF_POLL_INTERVAL:-60}  # 轮询间隔（秒）

# ============ 日志和 Checkpoint 保存间隔 ============
# 可通过环境变量覆盖: SAVE_FREQ=10 bash run_grpo.sh
# 注意: WandB metrics 每一步都会记录
# 此参数控制: checkpoint 保存 + checkpoint_monitor 输出
SAVE_STEPS=${SAVE_FREQ:-5}  # 默认每 5 步保存一次

# ============ 环境变量 ============
export RAY_DEDUP_LOGS=0
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN

# ============ 8卡配置 ============
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
N_GPUS=8


# Reference Agent 配置
REF_AGENT_URL="http://localhost:8000"
USE_FULL_REF_AGENT=true

# 检查 Reference Agent Server
echo "Checking Reference Agent Server at $REF_AGENT_URL ..."
if ! curl -s --connect-timeout 5 ${REF_AGENT_URL}/health > /dev/null 2>&1; then
    echo "WARNING: Reference Agent Server not running, using similarity fallback"
    USE_FULL_REF_AGENT=false
fi

echo "=========================================="
echo "Architecture: 8 GPU FSDP + vLLM TP=2 Rollout"
echo "Reference Agent Enabled: $USE_FULL_REF_AGENT"
echo "=========================================="

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/logs"

# ============ 启动 Checkpoint 自动上传器 ============
UPLOADER_PID=""
if [ "${HF_UPLOAD}" = "true" ]; then
    echo "=========================================="
    echo "Starting Checkpoint Auto-Uploader"
    echo "  HF User: ${HF_USER}"
    echo "  Collection: ${HF_USER}/${HF_COLLECTION_NAME}-${DATE_PREFIX}"
    echo "  Poll Interval: ${HF_POLL_INTERVAL}s"
    echo "  Visibility: Public"
    echo "=========================================="

    python3 "${FE_WORLD_ROOT}/train/checkpoint_uploader.py" \
        --checkpoint_dir "${OUTPUT_DIR}" \
        --hf_user "${HF_USER}" \
        --date_prefix "${DATE_PREFIX}" \
        --collection_name "${HF_COLLECTION_NAME}" \
        --poll_interval "${HF_POLL_INTERVAL}" \
        2>&1 | tee "${OUTPUT_DIR}/logs/uploader.log" &
    UPLOADER_PID=$!
    echo "Uploader started with PID: ${UPLOADER_PID}"
fi

# 清理函数：训练结束时停止上传器
cleanup() {
    if [ -n "${UPLOADER_PID}" ] && kill -0 "${UPLOADER_PID}" 2>/dev/null; then
        echo "Stopping uploader (PID: ${UPLOADER_PID})..."
        kill "${UPLOADER_PID}" 2>/dev/null
        wait "${UPLOADER_PID}" 2>/dev/null
        echo "Uploader stopped."
    fi
}
trap cleanup EXIT

# ============ 计算训练参数 ============
# 8卡，TP=2 意味着实际 DP=4，每个 DP rank 处理 batch_size/4 样本
TRAIN_BATCH_SIZE=8  # 小规模验证模式，验证通过后改为 32
NUM_SAMPLES=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('${TRAIN_DATA}')))")
STEPS_PER_EPOCH=$((NUM_SAMPLES / TRAIN_BATCH_SIZE))

# 使用配置的保存间隔（不超过每个 epoch 的步数）
ACTUAL_SAVE_FREQ=$((STEPS_PER_EPOCH < SAVE_STEPS ? STEPS_PER_EPOCH : SAVE_STEPS))

echo "=========================================="
echo "Training Configuration:"
echo "  Samples: $NUM_SAMPLES"
echo "  Batch Size: $TRAIN_BATCH_SIZE"
echo "  Steps per Epoch: $STEPS_PER_EPOCH"
echo "  Save Frequency: every $ACTUAL_SAVE_FREQ steps"
echo "  Total Epochs: $TOTAL_EPOCHS"
echo "  Rollout: vLLM TP=2 (long context optimized)"
echo "  Max Context: 14k input + 2k output = 16k tokens"
echo "=========================================="

# ============ 运行训练 ============
# 核心参数说明 (vLLM V1 引擎 + 16K 上下文):
#
# 1. tensor_model_parallel_size=2
#    - vLLM rollout 跨 2 卡分布 KV Cache
#
# 2. gpu_memory_utilization=0.5
#    - FSDP actor/ref models occupy ~37 GiB per GPU, leaving ~42 GiB free
#    - 0.5 * 79.25 = 39.6 GiB, fits within available free memory
#
# 3. max_model_len=16384 (14K prompt + 2K response)
#    - 提高上下文支持，覆盖更多训练样本
#
# 4. max_num_seqs=12
#    - 提高并发，加速 rollout
#
# 5. max_num_batched_tokens=16384
#    - 配合更大上下文的批处理
#
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=14336 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    +data.num_proc=64 \
    data.truncation='left' \
    actor_rollout_ref.model.path=${WORLD_MODEL} \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    ++actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    ++actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.max_num_seqs=12 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=${REWARD_FN_PATH} \
    custom_reward_function.name=compute_score \
    ++custom_reward_function.reward_kwargs.use_full_reference_agent=${USE_FULL_REF_AGENT} \
    ++custom_reward_function.reward_kwargs.use_http_reference_agent=True \
    ++custom_reward_function.reward_kwargs.reference_agent_api_url="${REF_AGENT_URL}" \
    ++custom_reward_function.reward_kwargs.api_timeout=600.0 \
    ++custom_reward_function.reward_kwargs.reward_mode="exponential" \
    ++custom_reward_function.reward_kwargs.behavior_scale_coef=1.5 \
    ++custom_reward_function.reward_kwargs.format_penalty=-2.0 \
    ++custom_reward_function.reward_kwargs.facts_weight=3 \
    ++custom_reward_function.reward_kwargs.behavior_weight=1 \
    ++custom_reward_function.reward_kwargs.length_penalty_weight=1.0 \
    ++custom_reward_function.reward_kwargs.length_min_ratio=0.7 \
    ++custom_reward_function.reward_kwargs.length_max_ratio=1.3 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='pivot-grpo-webshop' \
    trainer.experiment_name="wm-qwen2.5-7b-8gpu-${TIMESTAMP}" \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=${ACTUAL_SAVE_FREQ} \
    trainer.max_actor_ckpt_to_keep=${TOTAL_EPOCHS} \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir=${OUTPUT_DIR} \
    2>&1 | tee "${OUTPUT_DIR}/logs/train_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "=========================================="
echo "Training completed!"
echo "Checkpoints: ${OUTPUT_DIR}"
if [ "${HF_UPLOAD}" = "true" ]; then
    echo ""
    echo "Waiting for uploader to process remaining checkpoints..."
    # 给上传器额外时间处理最后的 checkpoints
    sleep 10
    # 手动触发最后一次处理
    python3 "${FE_WORLD_ROOT}/train/checkpoint_uploader.py" \
        --checkpoint_dir "${OUTPUT_DIR}" \
        --hf_user "${HF_USER}" \
        --date_prefix "${DATE_PREFIX}" \
        --collection_name "${HF_COLLECTION_NAME}" \
        --poll_interval 0  # 一次性运行模式
    echo ""
    echo "Models uploaded to: https://huggingface.co/${HF_USER}"
    echo "Collection: https://huggingface.co/collections/${HF_USER}/${HF_COLLECTION_NAME}-${DATE_PREFIX}"
fi
echo "=========================================="
