# Evaluation Guide

## Overview

BehR-WM provides a comprehensive evaluation framework with three categories of metrics:

| Category | Metric | Description |
|----------|--------|-------------|
| **Metric 1** | Single-step EM | World model's single-step prediction accuracy |
| **Metric 2** | Task Success Rate | |
| └─ 2.1 | WM SR | Agent success rate in world model |
| └─ 2.2 | W2R SR | World model action replay in real environment |
| └─ 2.3 | Real SR | Agent success rate in real environment |
| └─ 2.4 | CR | Consistency Ratio = W2R / Real |
| └─ 2.5 | CR_pw | Pairwise CR = \|Real✓ ∩ W2R✓\| / \|Real✓\| |
| **Metric 3** | Behavior Consistency | BehR reward validation |

## Prerequisites

1. Complete [Installation](INSTALL.md)
2. Start required servers (see below)

## Quick Start

### 1. Start Servers

```bash
# World Model server
bash scripts/servers/start_wm_server.sh -m <your_model> -p 8001 -gpu 0

# Reference Agent server (for Metric 3)
bash scripts/servers/start_reference_agent_server.sh -m Qwen/Qwen3-8B -p 8000 -gpu 1
```

### 2. Run Evaluations

```bash
source uv_webshop/bin/activate

# Metric 1: Single-step Accuracy
bash eval/01_single_step_accuracy/run.sh webshop <model_name> outputs/

# Metric 2.1: Agent in World Model
bash eval/02_task_success_rate/run_wm.sh

# Metric 2.2: Replay WM actions in Real Environment
bash eval/02_task_success_rate/run_wm2real.sh outputs/task_success_rate/wm/webshop/<experiment>

# Metric 2.3: Agent in Real Environment
bash eval/02_task_success_rate/run_real.sh

# Metric 2.4–2.5: CR and CR_pw
python eval/02_task_success_rate/analyze_pairwise_cr.py \
    --real-dir outputs/task_success_rate/real/webshop/<experiment> \
    --w2r-dir outputs/task_success_rate/w2r/webshop/<experiment>

# (alternative aggregate helper across many experiments)
python compute_cr.py --results-dir outputs/task_success_rate/

# Metric 3: Behavior Consistency
bash eval/03_behavior_consistency/run_eval_bf.sh
```

## TextWorld

All three metrics also apply to TextWorld. The `lookahead` variants of the
Metric 2 scripts run the agent with inference-time world-model simulation:

```bash
bash eval/02_task_success_rate/run_lookahead_textworld.sh
bash eval/02_task_success_rate/run_lookahead_full.sh         # WebShop equivalent
```

## Detailed Metrics

### Metric 1: Single-step Accuracy

Measures how accurately the world model predicts the next state given current state and action.

```bash
bash eval/01_single_step_accuracy/run.sh <task> <model> <output_dir>
```

**Output**: `accuracy_curve.png`, `outputs.jsonl`, `metrics.json`

### Metric 2: Task Success Rate (CR)

#### 2.1 WM — Agent in World Model

```bash
bash eval/02_task_success_rate/run_wm.sh [agent_model] [api_key]

# With local vLLM
bash eval/02_task_success_rate/run_wm.sh Qwen/Qwen2.5-7B-Instruct EMPTY http://localhost:8000/v1
```

#### 2.2 W2R — Replay in Real Environment

```bash
bash eval/02_task_success_rate/run_wm2real.sh <wm_output_dir>
```

#### 2.3 Real — Agent in Real Environment

```bash
bash eval/02_task_success_rate/run_real.sh [agent_model] [api_key]
```

#### 2.4–2.5 CR and CR_pw

```bash
python eval/02_task_success_rate/analyze_pairwise_cr.py \
    --real-dir <real_results_dir> \
    --w2r-dir <w2r_results_dir>
```

**CR** (aggregate): $CR = SR_{W2R} / SR_{Real}$

**CR_pw** (pairwise): $CR_{pw} = |Real_\checkmark \cap W2R_\checkmark| / |Real_\checkmark|$

- CR close to 1.0 indicates good functional equivalence
- CR_pw measures per-task consistency (more robust than aggregate CR)

### Metric 3: Behavior Consistency

```bash
bash eval/03_behavior_consistency/run_eval_bf.sh
```

**BehR** = $\exp(-\alpha \cdot |mean\_log\_prob_{pred} - mean\_log\_prob_{real}|)$

## Agent API Modes

### OpenAI API

```bash
API_KEY="sk-your-key"
API_BASE_URL="https://api.openai.com/v1"
MODEL="gpt-4o"
```

### Local vLLM

```bash
# Start vLLM server
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# Use in scripts
API_KEY="EMPTY"
API_BASE_URL="http://localhost:8000/v1"
MODEL="Qwen/Qwen2.5-7B-Instruct"
```

## Interpreting Results

| Metric | Good | Excellent |
|--------|------|-----------|
| Single-step EM | >70% | >80% |
| WM SR | >40% | >60% |
| W2R SR | >30% | >50% |
| Real SR | >50% | >70% |
| CR | >0.6 | >0.8 |
| CR_pw | >0.5 | >0.7 |
| BehR | >0.5 | >0.7 |

## Troubleshooting

### Server Connection Failed

```bash
curl http://localhost:8001/v1/models  # WM server
curl http://localhost:8000/v1/models  # Reference Agent server
```

### WebShop Environment Issues

```bash
java -version  # Must be Java 11+
```

### Out of Memory

- Reduce `--max-concurrency` in evaluation scripts
- Use smaller batch sizes

## Next Steps

- Compare your trained model against the SFT baseline
- Run ablation studies with different reward configurations
- Analyze failure cases in `outputs/` directories
