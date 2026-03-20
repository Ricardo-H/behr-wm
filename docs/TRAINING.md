# Training Guide

## Overview

BehR-WM uses GRPO (Group Relative Policy Optimization) with Behavior Consistency Reward (BehR) to train world models for functional equivalence.

## Prerequisites

1. Complete [Installation](INSTALL.md)
2. Have access to GPUs (recommended: 4–8× A100 80GB)

## Quick Start

### 1. Start Reference Agent Server

The reference agent server computes behavior consistency rewards during training.

```bash
# Shared-GPU reference agent (recommended for training)
bash scripts/servers/start_reference_agent_server.sh -m Qwen/Qwen3-8B -p 8000 -gpu 0,1,2,3 --shared
```

### 2. Run Training

```bash
source uv_webshop/bin/activate

# 8× A100 (recommended)
bash train/8gpu/run_grpo.sh

# 4× A100
bash train/4gpu/run_grpo.sh
```

## GPU Architecture

### 8-GPU Setup (Recommended)

```
8× A100 80GB
├── FSDP Training: 8 GPUs (Actor/Ref)
├── vLLM Rollout: TP=2 (shared memory with training)
└── Reference Agent: TP=8 shared (~5% VRAM per GPU)
```

### 4-GPU Setup

```
4× A100 80GB
├── FSDP Training: 4 GPUs (Actor/Ref)
├── vLLM Rollout: TP=1 (shared memory)
└── Reference Agent: TP=4 shared (~10% VRAM per GPU)
```

## Configuration

Key training parameters (configured via Hydra overrides in training scripts):

```bash
# Context lengths
data.max_prompt_length=32768
data.max_response_length=1024

# Learning rate
actor_rollout_ref.actor.optim.lr=5e-6

# GPU / batch
trainer.n_gpus_per_node=8
data.train_batch_size=128
```

## BehR Reward

The training uses pure BehR (Behavior Consistency Reward):

$$R_{BehR} = \exp(-\alpha \cdot |mean\_log\_prob_{pred} - mean\_log\_prob_{real}|)$$

- Range: (0, 1], higher is better
- Uses Mean Log Prob to eliminate length bias
- $\alpha$ (behavior_scale_coef) controls sensitivity (default: 1.0)

## Monitoring

### Weights & Biases

```bash
export WANDB_API_KEY="your-key"
export WANDB_PROJECT="behr-wm"

# Or disable
export WANDB_MODE=disabled
```

### Key W&B Metrics

| Metric | Description |
|--------|-------------|
| `reward/behavior_reward/mean` | BehR score distribution |
| `reward/mean_diff/mean` | Mean Log Prob difference (→0 is better) |
| `reward/fallback_ratio` | Fraction falling back to similarity (→0 is better) |
| `reward/format_penalty_count` | Format violations (should be 0) |

### Checkpoints

Saved to `outputs/checkpoints/` every N steps.

## Troubleshooting

### Out of Memory

1. Reduce `max_prompt_length`
2. Reduce batch size
3. Enable gradient checkpointing

### Reference Agent Server Not Responding

```bash
curl http://localhost:8000/v1/models
```

## Next Steps

After training, evaluate your model: [EVALUATION.md](EVALUATION.md)
