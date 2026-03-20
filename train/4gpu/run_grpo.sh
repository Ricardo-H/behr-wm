#!/bin/bash
#
# Pivot-GRPO Training Script - 4 GPU 共享模式
#
# 架构: 4 卡 FSDP 训练 + vLLM TP=2 Rollout (共享显存)
#
# 核心特点:
# 1. 4 卡全部参与 FSDP 训练 (Actor/Ref)
# 2. vLLM rollout 使用 TP=2，每 2 卡为一组
# 3. 训练和 Rollout 共享显存，交替运行
# 4. Reference Agent 复用训练卡 (TP=4, 每卡~10%显存)
#
# 适用场景:
# - 长上下文 (8k avg, 18k max)
# - 4 卡 A100 环境
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
TRAIN_DATA="${FE_WORLD_ROOT}/data/train/hard_samples/train.parquet"
VAL_DATA="${FE_WORLD_ROOT}/data/train/hard_samples/test.parquet"

echo "=========================================="
echo "4-GPU GRPO Training (Shared Mode)"
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
OUTPUT_DIR="${FE_WORLD_ROOT}/outputs/checkpoints/4gpu_${TIMESTAMP}"
TOTAL_EPOCHS=5

# ============ 日志和 Checkpoint 保存间隔 ============
# 可通过环境变量覆盖: SAVE_FREQ=10 bash run_grpo.sh
# 注意: WandB metrics 每一步都会记录
# 此参数控制: checkpoint 保存 + checkpoint_monitor 输出
SAVE_STEPS=${SAVE_FREQ:-10}  # 默认每 x 步保存一次

# ============ 环境变量 ============
export RAY_DEDUP_LOGS=0
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN

# ============ 4卡配置 ============
export CUDA_VISIBLE_DEVICES=0,1,2,3
N_GPUS=4


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
echo "Architecture: 4 GPU FSDP + vLLM TP=2 Rollout"
echo "Reference Agent Enabled: $USE_FULL_REF_AGENT"
echo "=========================================="

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/logs"

# ============ 计算训练参数 ============
TRAIN_BATCH_SIZE=32
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
echo "  Max Context: 28k input + 1k output = 29k tokens"
echo "=========================================="

# ============ 运行训练 ============
# 核心参数说明
# 1. tensor_model_parallel_size=8
#    - vLLM rollout 跨 4 卡分布 KV Cache，避免长上下文 OOM

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=28672 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    +data.num_proc=64 \
    data.truncation='left' \
    actor_rollout_ref.model.path=${WORLD_MODEL} \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    ++actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    ++actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.20 \
    actor_rollout_ref.rollout.max_model_len=30720 \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.max_num_batched_tokens=30720 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=1.2 \
    actor_rollout_ref.rollout.top_p=1.00 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=${REWARD_FN_PATH} \
    custom_reward_function.name=compute_score \
    ++custom_reward_function.reward_kwargs.use_full_reference_agent=${USE_FULL_REF_AGENT} \
    ++custom_reward_function.reward_kwargs.use_http_reference_agent=True \
    ++custom_reward_function.reward_kwargs.reference_agent_api_url="${REF_AGENT_URL}" \
    ++custom_reward_function.reward_kwargs.api_timeout=2400.0 \
    ++custom_reward_function.reward_kwargs.max_workers=16 \
    ++custom_reward_function.reward_kwargs.reward_mode="exponential" \
    ++custom_reward_function.reward_kwargs.behavior_scale_coef=1.0 \
    ++custom_reward_function.reward_kwargs.format_penalty=-2.0 \
    ++custom_reward_function.reward_kwargs.facts_weight=0 \
    ++custom_reward_function.reward_kwargs.behavior_weight=1 \
    ++custom_reward_function.reward_kwargs.length_penalty_weight=0 \
    ++custom_reward_function.reward_kwargs.length_min_ratio=0.7 \
    ++custom_reward_function.reward_kwargs.length_max_ratio=1.3 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='pivot-grpo-webshop' \
    trainer.experiment_name="wm-qwen2.5-7b-4gpu-${TIMESTAMP}" \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=${ACTUAL_SAVE_FREQ} \
    trainer.max_actor_ckpt_to_keep=20 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir=${OUTPUT_DIR} \
    2>&1 | tee "${OUTPUT_DIR}/logs/train_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "=========================================="
echo "Training completed!"
echo "Checkpoints: ${OUTPUT_DIR}"
echo "=========================================="
