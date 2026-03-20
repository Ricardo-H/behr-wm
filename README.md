# BehR-WM: Beyond State Consistency — Behavior Consistency in Text-Based World Models

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

</div>

## Overview

This repository provides the code for training and evaluating **BehR-WM** (Behavior Consistency Reward for World Models). BehR-WM optimizes text-based world models for **functional equivalence** rather than surface-level text similarity.

**Core Insight**: A world model is functionally equivalent to the real environment if an agent cannot distinguish between them through its actions. Instead of maximizing token-level Exact Match (EM), we train world models with a **Behavior Consistency Reward (BehR)** that measures behavioral indistinguishability.

**Key Metric — Consistency Ratio (CR)**:

$$CR = \frac{SR_{W2R}}{SR_{Real}}, \quad CR_{pw} = \frac{|Real_\checkmark \cap W2R_\checkmark|}{|Real_\checkmark|}$$

Where $SR_{W2R}$ is the success rate of replaying WM-generated action sequences in the real environment, and $SR_{Real}$ is the agent's success rate in the real environment. $CR \to 1.0$ indicates perfect functional equivalence.

## Features

- **BehR Training**: GRPO-based training with Behavior Consistency Reward
- **3-Metric Evaluation**: Single-step EM, Task Success Rate (CR/CR_pw), Behavior Consistency
- **Multi-Environment**: WebShop and TextWorld support
- **Multi-Agent**: Supports OpenAI API and local vLLM deployment

## Project Structure

```
behr-wm/
├── src/                           # Core source code
│   ├── agents/                    # ReAct agent implementation
│   ├── api/                       # OpenAI / vLLM client
│   ├── reward/                    # BehR reward computation
│   ├── world_model/               # World model interface
│   └── utils/                     # Common utilities
├── train/                         # GRPO training scripts
│   ├── reward_function.py         # BehR reward for training
│   ├── checkpoint_monitor.py      # Training monitoring
│   ├── 4gpu/                      # 4× A100 configuration
│   └── 8gpu/                      # 8× A100 configuration
├── eval/                          # Evaluation framework
│   ├── 01_single_step_accuracy/   # Metric 1: EM
│   ├── 02_task_success_rate/      # Metric 2: WM/W2R/Real/CR/CR_pw
│   └── 03_behavior_consistency/    # Metric 3: BehR validation
├── configs/                       # Configuration templates
├── scripts/                       # Environment setup & server scripts
├── data/                          # System prompts & data references
└── docs/                          # Detailed guides
```

## Quick Start

### 1. Environment Setup

**Requirements**: Linux, Python 3.10+, CUDA 12.x, 4× A100 80GB (recommended)

```bash
git clone https://anonymous.4open.science/r/behr-wm-787B
cd behr-wm

# Install evaluation environment (includes vLLM + WebShop)
bash scripts/env_setup/uv_webshop.sh

# (Optional) Install GRPO training framework
pip install verl

# Activate environment
source uv_webshop/bin/activate
```

### 2. Evaluation

#### Start Servers

```bash
# Start World Model server
bash scripts/servers/start_wm_server.sh -m <your_wm_model> -p 8001 -gpu 0

# Start WebShop environment
bash scripts/servers/start_webshop_env.sh 36001

# (For BehR evaluation) Start Reference Agent server
bash scripts/servers/start_reference_agent_server.sh -m Qwen/Qwen3-8B -p 8000 -gpu 1
```

#### Run Evaluations

```bash
source uv_webshop/bin/activate

# Metric 1: Single-step Exact Match
bash eval/01_single_step_accuracy/run.sh webshop <model_name> outputs/

# Metric 2: Task Success Rate (CR)
# Step 1: Agent in World Model
bash eval/02_task_success_rate/run_wm.sh
# Step 2: Replay WM actions in Real Environment
bash eval/02_task_success_rate/run_wm2real.sh outputs/task_success_rate/wm/webshop/
# Step 3: Agent in Real Environment (baseline)
bash eval/02_task_success_rate/run_real.sh
# Step 4: Compute CR and CR_pw
python eval/02_task_success_rate/analyze_pairwise_cr.py \
    --real-dir outputs/task_success_rate/real/webshop/<experiment> \
    --w2r-dir outputs/task_success_rate/w2r/webshop/<experiment>

# Metric 3: Behavior Consistency
bash eval/03_behavior_consistency/run_eval_bf.sh
```

### 3. Training

```bash
source uv_webshop/bin/activate

# Start Reference Agent server for BehR computation
bash scripts/servers/start_reference_agent_server.sh -m Qwen/Qwen3-8B -p 8000 -gpu 0,1,2,3 --shared

# Run GRPO training (8× A100 recommended)
bash train/8gpu/run_grpo.sh
```

See [docs/TRAINING.md](docs/TRAINING.md) for detailed configuration options.

## Evaluation Metrics

| Metric | Description | Script |
|--------|-------------|--------|
| **EM** | Single-step prediction accuracy | `eval/01_single_step_accuracy/run.sh` |
| **WM SR** | Agent success rate in world model | `eval/02_task_success_rate/run_wm.sh` |
| **W2R SR** | WM action replay success in real env | `eval/02_task_success_rate/run_wm2real.sh` |
| **Real SR** | Agent success rate in real environment | `eval/02_task_success_rate/run_real.sh` |
| **CR** | Consistency Ratio = W2R / Real | `eval/02_task_success_rate/analyze_pairwise_cr.py` |
| **CR_pw** | Pairwise CR = \|Real✓ ∩ W2R✓\| / \|Real✓\| | `eval/02_task_success_rate/analyze_pairwise_cr.py` |
| **BehR** | Behavior Consistency reward score | `eval/03_behavior_consistency/run_eval_bf.sh` |

## Data

This work uses interaction trajectories from the following environments:
- **WebShop** ([Yao et al., 2022](https://arxiv.org/abs/2207.01206)): E-commerce web navigation
- **TextWorld** ([Côté et al., 2019](https://arxiv.org/abs/1806.11532)): Text-based game environment

Evaluation uses 200 standardized test tasks from [AgentEnv](https://github.com/WooooDyy/AgentGym) for each environment.

Training data, model weights, and evaluation datasets will be released upon paper acceptance.

## BehR Reward

The Behavior Consistency Reward measures behavioral indistinguishability between WM-predicted states and real states:

$$R_{BehR} = \exp\left(-\alpha \cdot \left|\frac{1}{N}\sum_{i=1}^{N}\log\pi(a_i|s_{pred}) - \frac{1}{N}\sum_{i=1}^{N}\log\pi(a_i|s_{real})\right|\right)$$

Where:
- $\pi(a_i|s)$ is the agent's action probability given state $s$
- $s_{pred}$ and $s_{real}$ are WM-predicted and real environment states
- $\alpha$ is the sensitivity coefficient (default: 1.0)
- Mean log probability (not sum) eliminates length bias

## Documentation

- [Installation Guide](docs/INSTALL.md)
- [Evaluation Guide](docs/EVALUATION.md)
- [Training Guide](docs/TRAINING.md)

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
