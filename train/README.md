# GRPO Training Scripts

GRPO (Group Relative Policy Optimization) training scripts for world models with BehR reward. Supports 4-GPU and 8-GPU A100 setups.

## Directory Structure

```
train/
├── 4gpu/                          # 4-GPU configuration
│   ├── run_grpo.sh               # Main training script
│   └── start_reference_agent_server.sh     # Reference Agent service (TP=4)
│
├── 8gpu/                          # 8-GPU configuration (recommended)
│   ├── run_grpo.sh               # Main training script (8-GPU FSDP + TP=2 Rollout)
│   └── start_reference_agent_server.sh     # Reference Agent service (TP=8)
│
├── reward_function.py             # BehR reward function
├── checkpoint_monitor.py          # Checkpoint monitoring module
└── README.md                      # This file
```

## Setup Comparison

| Setup | Hardware | Max Context | Throughput | Use Case |
|-------|----------|-------------|------------|----------|
| **4gpu** | 4× A100 80GB | ~10k tokens | Medium | Resource-constrained |
| **8gpu** | 8× A100 80GB | ~14k tokens | High | Long trajectories |

## Quick Start

### 8-GPU Setup (Recommended)

```bash
# 1. Start Reference Agent server
bash scripts/servers/start_reference_agent_server.sh -m Qwen/Qwen3-8B -p 8000 -gpu 0,1,2,3,4,5,6,7 --shared

# 2. Run training
bash train/8gpu/run_grpo.sh
```

### 4-GPU Setup

```bash
# 1. Start Reference Agent server
bash scripts/servers/start_reference_agent_server.sh -m Qwen/Qwen3-8B -p 8000 -gpu 0,1,2,3 --shared

# 2. Run training
bash train/4gpu/run_grpo.sh
```

## BehR Reward Function

`reward_function.py` implements the Behavior Consistency Reward:

$$R_{BehR} = \exp(-\alpha \cdot |mean\_log\_prob_{pred} - mean\_log\_prob_{real}|)$$

- **R_behavior**: Agent action distribution consistency between WM-predicted and real states
- Range: (0, 1], higher indicates better behavior consistency

### W&B Monitored Metrics

| Metric | Description |
|--------|-------------|
| `reward/behavior_reward/mean,max,min` | BehR score distribution |
| `reward/mean_diff/mean,max,min` | Mean Log Prob difference (→0 is better) |
| `reward/fallback_ratio` | Fallback rate (→0 is better) |
| `reward/format_penalty_count` | Format violations (should be 0) |

## Notes

1. **Reference Agent server is optional**: Without it, training falls back to text similarity for behavior reward
2. **VRAM monitoring**: Use `watch -n 1 nvidia-smi` to monitor GPU memory
3. **OOM handling**: Reduce `TRAIN_BATCH_SIZE` or `rollout.max_num_seqs` if you encounter OOM
