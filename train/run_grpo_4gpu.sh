#!/bin/bash
# =============================================================================
# BehR-WM GRPO Training (default: 4× A100-80GB, WebShop)
# =============================================================================
#
# Before running:
#   1. Start the Reference Agent (judge) vLLM server on port 8000, e.g.
#        bash scripts/servers/start_reference_agent_server.sh \
#             -m Qwen/Qwen3-8B -p 8000 -gpu 0,1,2,3 --shared
#   2. Set TRAIN_DATA / VAL_DATA to your parquet splits (see docs/TRAINING.md).
#   3. Optional overrides: WORLD_MODEL, OUTPUT_DIR, JUDGE_URL, N_GPUS.
#
# For 8-GPU / TextWorld variants, see docs/TRAINING.md.
# =============================================================================

set -eu

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Activate environment if present
if [ -f "$PROJECT_ROOT/uv_webshop/bin/activate" ]; then
    source "$PROJECT_ROOT/uv_webshop/bin/activate"
fi

# ---- Required: data paths ----
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/train.parquet}"
VAL_DATA="${VAL_DATA:-${PROJECT_ROOT}/data/test.parquet}"

# ---- Model / reward ----
WORLD_MODEL="${WORLD_MODEL:-X1AOX1A/WorldModel-Webshop-Llama3.1-8B}"
REWARD_FN_PATH="${REWARD_FN_PATH:-${PROJECT_ROOT}/src/reward/behr_reward_webshop.py}"
JUDGE_URL="${JUDGE_URL:-http://localhost:8000}"

# ---- Hardware (4-GPU default) ----
N_GPUS="${N_GPUS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# ---- Output ----
TIMESTAMP=$(date +%Y%m%d_%H%M)
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/checkpoints/behr_${TIMESTAMP}}"
mkdir -p "${OUTPUT_DIR}/logs"

# ---- Pre-flight checks ----
for f in "$TRAIN_DATA" "$VAL_DATA" "$REWARD_FN_PATH"; do
    [ -f "$f" ] || { echo "ERROR: file not found: $f"; exit 1; }
done
if ! curl -s --connect-timeout 10 "${JUDGE_URL}/health" > /dev/null 2>&1; then
    echo "ERROR: Reference Agent server not reachable at ${JUDGE_URL}"
    echo "Start it first: scripts/servers/start_reference_agent_server.sh"
    exit 1
fi

echo "=========================================="
echo "BehR-WM GRPO Training (${N_GPUS}× GPU)"
echo "  Model:   $WORLD_MODEL"
echo "  Reward:  $(basename "$REWARD_FN_PATH")"
echo "  Judge:   $JUDGE_URL"
echo "  Train:   $TRAIN_DATA"
echo "  Output:  $OUTPUT_DIR"
echo "=========================================="

export RAY_DEDUP_LOGS=0
export TOKENIZERS_PARALLELISM=true

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.train_batch_size=32 \
    data.max_prompt_length=14336 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    actor_rollout_ref.model.path="${WORLD_MODEL}" \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=1.3 \
    actor_rollout_ref.rollout.top_p=1.0 \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.name=compute_score \
    ++custom_reward_function.reward_kwargs.use_http_judge=True \
    ++custom_reward_function.reward_kwargs.judge_api_url="${JUDGE_URL}" \
    ++custom_reward_function.reward_kwargs.reward_mode="cauchy" \
    ++custom_reward_function.reward_kwargs.behavior_scale_coef=1.0 \
    ++custom_reward_function.reward_kwargs.behavior_weight=0.8 \
    ++custom_reward_function.reward_kwargs.facts_weight=0.2 \
    ++custom_reward_function.reward_kwargs.format_penalty=-2.0 \
    ++custom_reward_function.reward_kwargs.max_workers=16 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='behr-wm' \
    trainer.experiment_name="behr-${TIMESTAMP}" \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.total_epochs=5 \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/logs/train_${TIMESTAMP}.log"

echo "=========================================="
echo "Training complete. Checkpoints: ${OUTPUT_DIR}"
echo "=========================================="
