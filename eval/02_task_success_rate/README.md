# Task Success Rate Evaluation (Metric 2)

## Overview

This directory implements the task success rate evaluation pipeline:
- **WM**: Agent interacts with the World Model
- **W2R**: Replay WM-generated action sequences in the real environment
- **Real**: Agent interacts with the real environment (baseline)
- **CR**: Consistency Ratio = W2R SR / Real SR
- **CR_pw**: Pairwise CR = |Real✓ ∩ W2R✓| / |Real✓|

## Pipeline

```
Step 1: run_wm.sh        → Agent in World Model → WM trajectories
Step 2: run_wm2real.sh   → Replay WM actions in Real → W2R results
Step 3: run_real.sh       → Agent in Real Environment → Real baseline
Step 4: analyze_pairwise_cr.py → Compute CR and CR_pw
```

## Usage

### Prerequisites

```bash
# Start World Model server
bash scripts/servers/start_wm_server.sh -m <model> -p 8001 -gpu 0

# Start WebShop environment (for W2R and Real)
bash scripts/servers/start_webshop_env.sh 36001

# (Optional) Start Agent server (for local vLLM agent)
bash scripts/servers/start_agent_server.sh -m Qwen/Qwen3-8B -p 8000 -gpu 1
```

### Step 1: Agent in World Model

```bash
# With local vLLM agent
bash eval/02_task_success_rate/run_wm.sh <experiment_name> EMPTY http://localhost:8000/v1 vllm_agent

# With OpenAI API
bash eval/02_task_success_rate/run_wm.sh <experiment_name> sk-xxx https://api.openai.com/v1 gpt-4o
```

### Step 2: Replay in Real Environment

```bash
bash eval/02_task_success_rate/run_wm2real.sh outputs/task_success_rate/wm/webshop/<experiment>
```

### Step 3: Agent in Real Environment

```bash
# With local vLLM agent
bash eval/02_task_success_rate/run_real.sh <experiment_name> vllm_agent EMPTY http://localhost:8000/v1

# With OpenAI API
bash eval/02_task_success_rate/run_real.sh <experiment_name> gpt-4o sk-xxx https://api.openai.com/v1
```

### Step 4: Compute CR and CR_pw

```bash
python eval/02_task_success_rate/analyze_pairwise_cr.py \
    --real-dir outputs/task_success_rate/real/webshop/<experiment> \
    --w2r-dir outputs/task_success_rate/w2r/webshop/<experiment>
```

## Key Files

| File | Description |
|------|-------------|
| `interact_with_wm.py` | Agent-WM interaction loop |
| `interact_with_real.py` | Agent-Real environment interaction |
| `interact_with_lookahead.py` | K-step lookahead planning (WebShop) |
| `interact_with_lookahead_textworld.py` | K-step lookahead planning (TextWorld) |
| `cal_wm2real.py` | W2R replay calculation |
| `analyze_pairwise_cr.py` | CR and CR_pw computation |
| `run_wm.sh` | WM evaluation runner |
| `run_wm2real.sh` | W2R evaluation runner |
| `run_real.sh` | Real environment evaluation runner |
| `02_pipeline.sh` | Full WM→W2R→Real pipeline |

## Output Format

```
outputs/task_success_rate/
├── wm/webshop/<experiment>/       # WM interaction results
│   ├── webshop_0.json
│   └── ...
├── w2r/webshop/<experiment>/      # W2R replay results
│   └── ...
└── real/webshop/<experiment>/     # Real environment results
    └── ...
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| WM server not responding | Check `curl http://localhost:8001/v1/models` |
| WebShop env timeout | Restart env: `bash scripts/servers/start_webshop_env.sh` |
| GPU OOM | Reduce `--gpu-memory-utilization` or use TP>1 |
