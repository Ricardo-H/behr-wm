# Training Guide

This guide documents the verl GRPO configuration used to produce the BehR-WM
checkpoints reported in the paper. We deliberately **do not ship cluster-specific
launch scripts**; instead we provide the reward code plus a reference
invocation that you can adapt to your own environment.

## 1. Prerequisites

- Finish [Installation](INSTALL.md).
- Install the training framework:

  ```bash
  bash scripts/env_setup/install_verl.sh   # or: pip install verl
  ```

- Hardware: 4× A100-80GB (minimum) or 8× A100-80GB (recommended).
- Base world models used in the paper (all on HuggingFace):
  - `X1AOX1A/WorldModel-Webshop-Qwen2.5-7B`
  - `X1AOX1A/WorldModel-Webshop-Llama3.1-8B`
  - `X1AOX1A/WorldModel-Textworld-Qwen2.5-7B`
  - `X1AOX1A/WorldModel-Textworld-Llama3.1-8B`
- Reference Agent: a frozen instruction-tuned LLM exposed via OpenAI-compatible
  HTTP (we use `Qwen/Qwen3-8B` in the paper).

## 2. Architecture Overview

GRPO training co-locates three components:

```
Training node
├── FSDP actor + reference policy           (all GPUs)
├── vLLM rollout                            (shared GPUs, TP=2 for 8-GPU / TP=1 for 4-GPU)
└── Reference Agent HTTP server (judge)     (remote or same-node, queried over HTTP)
```

The BehR reward implementation in [`src/reward/`](../src/reward/) calls the
Reference Agent via HTTP to compute per-token log-probabilities of the logged
action under both the WM-predicted state and the real state.

## 3. Start the Reference Agent server

```bash
# Co-located on training GPUs (tensor-parallel, memory-shared with rollout)
bash scripts/servers/start_reference_agent_server.sh \
     -m Qwen/Qwen3-8B -p 8000 -gpu 0,1,2,3 --shared
```

Health-check:

```bash
curl http://localhost:8000/health
```

## 4. Plug BehR into verl

verl supports user-supplied reward functions through
`custom_reward_function.path` / `custom_reward_function.name`. Point it at the
environment-specific module:

| Environment | Module |
|-------------|--------|
| WebShop | [`src/reward/behr_reward_webshop.py`](../src/reward/behr_reward_webshop.py) |
| TextWorld | [`src/reward/behr_reward_textworld.py`](../src/reward/behr_reward_textworld.py) |

Both export `compute_score(data_source, solution_str, ground_truth, extra_info)`
and return a dict with a `"score"` field plus BehR diagnostics.

### Reward hyper-parameters

| Key | WebShop | TextWorld | Meaning |
|-----|---------|-----------|---------|
| `reward_mode` | `cauchy` (recommended) or `exponential` | same | Shape of the BehR transform |
| `behavior_scale_coef` ($\alpha$) | `1.0` | `1.0` | Sensitivity of BehR to $|\Delta|$ |
| `behavior_weight` | `0.8` (BehR+FactR) / `1.0` (BehR-only) | `1.0` | Weight of BehR in the total reward |
| `facts_weight` | `0.2` / `0.0` | `0.0` | Physical-facts reward (WebShop only) |
| `length_penalty_weight` | `0.0` | `0.0` | Disabled in the final paper setup |
| `format_penalty` | `-2.0` | `-1.0` | Penalty for malformed WM outputs |
| `use_http_judge` | `True` | `True` | Use the Reference Agent HTTP server |
| `judge_api_url` | `http://localhost:8000` | same | Reference Agent endpoint |
| `api_timeout` | `2400.0` | `2400.0` | Seconds |
| `max_workers` | `16` | `8` | Parallel reward-compute workers |

## 5. Reference verl invocation (WebShop, 4× A100)

The command below reproduces the BehR setup used for LLaMA-3.1-8B on WebShop.
Edit `TRAIN_DATA`, `VAL_DATA`, `WORLD_MODEL`, `REWARD_FN_PATH`, and output paths
for your environment.

```bash
TRAIN_DATA=/path/to/webshop_train.parquet
VAL_DATA=/path/to/webshop_test.parquet
WORLD_MODEL=X1AOX1A/WorldModel-Webshop-Llama3.1-8B
REWARD_FN_PATH=$(pwd)/src/reward/behr_reward_webshop.py
JUDGE_URL=http://localhost:8000
OUTPUT_DIR=./outputs/checkpoints/behr_llama_$(date +%Y%m%d_%H%M)

export CUDA_VISIBLE_DEVICES=0,1,2,3
export RAY_DEDUP_LOGS=0
export TOKENIZERS_PARALLELISM=true

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.train_batch_size=32 \
    data.max_prompt_length=14336 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    +data.num_proc=64 \
    data.truncation='left' \
    actor_rollout_ref.model.path=${WORLD_MODEL} \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
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
    actor_rollout_ref.rollout.temperature=1.3 \
    actor_rollout_ref.rollout.top_p=1.00 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=${REWARD_FN_PATH} \
    custom_reward_function.name=compute_score \
    ++custom_reward_function.reward_kwargs.use_http_judge=True \
    ++custom_reward_function.reward_kwargs.judge_api_url="${JUDGE_URL}" \
    ++custom_reward_function.reward_kwargs.api_timeout=2400.0 \
    ++custom_reward_function.reward_kwargs.max_workers=16 \
    ++custom_reward_function.reward_kwargs.reward_mode="cauchy" \
    ++custom_reward_function.reward_kwargs.behavior_scale_coef=1.0 \
    ++custom_reward_function.reward_kwargs.format_penalty=-2.0 \
    ++custom_reward_function.reward_kwargs.facts_weight=0.2 \
    ++custom_reward_function.reward_kwargs.behavior_weight=0.8 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='behr-wm-webshop' \
    trainer.experiment_name="wm-llama3.1-8b-behr-$(date +%Y%m%d_%H%M)" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.max_actor_ckpt_to_keep=20 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_epochs=5 \
    trainer.default_local_dir=${OUTPUT_DIR}
```

### 8-GPU adjustments

Keep all reward-side parameters identical; change only the capacity knobs:

```
actor_rollout_ref.rollout.tensor_model_parallel_size=2
trainer.n_gpus_per_node=8
data.train_batch_size=128
actor_rollout_ref.rollout.n=8
```

### TextWorld adjustments

Swap the reward module and drop the physical-facts term:

```
custom_reward_function.path=$(pwd)/src/reward/behr_reward_textworld.py
++custom_reward_function.reward_kwargs.behavior_weight=1.0
++custom_reward_function.reward_kwargs.facts_weight=0.0
++custom_reward_function.reward_kwargs.format_penalty=-1.0
++custom_reward_function.reward_kwargs.max_workers=8
```

## 6. Data format

Training consumes parquet files whose rows expose the fields documented in
[`src/data/prepare_data.py`](../src/data/prepare_data.py) — essentially a
prompt (agent-visible context), a target next state (real observation), and
enough metadata for BehR to re-construct the logged action. Released dataset
artifacts (see [Release Timeline](../README.md#release-timeline)) ship in this
format.

## 7. Monitoring

- Reward diagnostics are logged per step under `custom_reward_function` keys —
  `score`, `behavior_reward`, `facts_reward`, `mean_log_prob_pred`,
  `mean_log_prob_real`.
- Checkpoints land in `OUTPUT_DIR`; use `scripts/servers/start_wm_server.sh`
  to serve them for evaluation (see [EVALUATION.md](EVALUATION.md)).

## 8. Reproducibility notes

The paper numbers were obtained with:
- verl as of the v0.2 release (pinned revision documented in
  `scripts/env_setup/install_verl.sh`),
- vLLM 0.6.x with `enforce_eager=True` (disables CUDA-graph capture, removes
  a class of non-determinism in rollout),
- temperature 1.3, top-p 1.0, group size 5 (4-GPU) or 8 (8-GPU),
- 5 training epochs, checkpoints saved every 20 steps.

We observed that `cauchy` reward mode converges more stably than `exponential`
at large $|\Delta|$ (better gradient preservation); `exponential` remains a
valid choice for smoother reward landscapes.
