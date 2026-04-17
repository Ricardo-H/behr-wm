# BehR-WM: Behavior Consistency Reward for Text-Based World Models

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](#citation)
[![Models](https://img.shields.io/badge/Models-TBA-yellow.svg)](#release-timeline)
[![Dataset](https://img.shields.io/badge/Dataset-TBA-yellow.svg)](#release-timeline)

</div>

> Beyond surface-level state consistency — training text-based world models to
> preserve **agent behavior** rather than match every token.

<p align="center">
  <img src="assets/pipeline.png" alt="BehR-WM training pipeline" width="92%"/>
</p>

## Overview

**BehR-WM** reframes world-model training around **functional equivalence**: a
world model is a good simulator of the real environment if a frozen Reference
Agent cannot distinguish the two through its actions. Instead of maximizing
Exact Match between predicted and real states, we optimize a **Behavior
Consistency Reward (BehR)** that directly measures behavioral indistinguishability
and provides a dense per-step signal for GRPO training.

**Core insight.** Textual faithfulness is neither necessary nor sufficient for
agent-facing correctness: small surface omissions can derail the agent, while
large but decision-irrelevant differences leave behavior unchanged. BehR asks
the right question — *does the agent act the same?* — and turns the answer into
a tractable, gradient-friendly reward.

**Key task-level metric — Consistency Ratio (CR):**

$$CR = \frac{SR_{W2R}}{SR_{Real}}, \qquad CR_{pw} = \frac{|Real_{\checkmark} \cap W2R_{\checkmark}|}{|Real_{\checkmark}|}$$

where $SR_{W2R}$ is the success rate of replaying the agent's
world-model-generated action sequence in the real environment and $SR_{Real}$
is the agent's success rate in the real environment. $CR \to 1.0$ indicates
functional equivalence.

## Features

- **BehR reward** — drop-in GRPO reward for the [verl](https://github.com/volcengine/verl)
  framework, with WebShop (behavior + physical-facts) and TextWorld (pure behavior) variants.
- **3-metric evaluation suite** — Single-step EM, Task Success Rate
  (SR<sub>WM</sub> / SR<sub>W2R</sub> / SR<sub>Real</sub> / CR / CR<sub>pw</sub>),
  and step-level Behavior Consistency.
- **Two environments** — WebShop (e-commerce web navigation) and TextWorld
  (text adventure games).
- **Two backends** — OpenAI-compatible API agents and locally hosted vLLM agents,
  via a uniform agent interface.

## Release Timeline

This repository follows an incremental open-source release. Check back for updates.

| Milestone | Content | Status |
|-----------|---------|--------|
| **v0.1 — 2026-03** | Initial eval/train skeleton, configs, docs | ✅ Released |
| **v0.2 — 2026-04** | BehR reward code (WebShop + TextWorld), full 3-stage evaluation pipeline, `compute_cr.py`, training config documentation | ✅ Released |
| **v0.3 — TBA** | Trained world-model checkpoints on HuggingFace (Qwen2.5-7B, Llama-3.1-8B, Ministral-3B) | 🔜 Planned |
| **v0.4 — TBA** | Training & evaluation datasets (trajectories + init contexts) on HuggingFace Hub | 🔜 Planned |
| **v0.5 — TBA** | Lookahead-planning inference extension, cross-judge evaluation artifacts | 🔜 Planned |

Artifact links will be added here and under [`data/README.md`](data/README.md) as
each milestone ships.

## Repository Layout

```
behr-wm/
├── src/
│   ├── agents/              # ReAct agent implementation
│   ├── api/                 # OpenAI-compatible / vLLM client
│   ├── reward/              # BehR reward: behr_reward_webshop.py, behr_reward_textworld.py
│   ├── world_model/         # World model client interface
│   ├── data/                # Data preparation helpers
│   └── utils/               # Common utilities
├── eval/
│   ├── 01_single_step_accuracy/   # Metric 1: Exact Match
│   ├── 02_task_success_rate/      # Metric 2: SR_WM / SR_W2R / SR_Real / CR / CR_pw
│   └── 03_behavior_consistency/   # Metric 3: step-level BehR score
├── scripts/
│   ├── env_setup/           # uv / verl environment bootstrap
│   ├── servers/             # vLLM & WebShop server launchers
│   └── download_data.py     # Fetch init contexts from HuggingFace (TBA)
├── configs/                 # Eval and training config templates
├── data/                    # Init contexts (via download_data.py)
├── docs/                    # INSTALL / EVALUATION / TRAINING
└── assets/                  # Figures used in README / docs
```

## Quick Start — Evaluation

### 1. Install

Requirements: Linux, Python 3.10+, CUDA 12.x, ≥ 1× A100-80GB (evaluation) or
≥ 4× A100-80GB (training).

```bash
git clone https://github.com/Ricardo-H/behr-wm.git
cd behr-wm

bash scripts/env_setup/uv_webshop.sh   # vLLM + WebShop environment
source uv_webshop/bin/activate

python scripts/download_data.py         # fetch init_contexts (TBA: HF repo)
```

See [docs/INSTALL.md](docs/INSTALL.md) for troubleshooting.

### 2. Start servers

```bash
# World model under test
bash scripts/servers/start_wm_server.sh -m <wm_model_path> -p 8001 -gpu 0

# WebShop backend
bash scripts/servers/start_webshop_env.sh 36001

# Reference Agent (required for Metric 3 / training)
bash scripts/servers/start_reference_agent_server.sh -m Qwen/Qwen3-8B -p 8000 -gpu 1
```

### 3. Run the 3-stage evaluation

```bash
# Metric 1 — Single-step Exact Match
bash eval/01_single_step_accuracy/run.sh webshop <model_name> outputs/

# Metric 2 — Task Success Rate & Consistency Ratio
bash eval/02_task_success_rate/run_wm.sh        # agent rolls out inside the WM
bash eval/02_task_success_rate/run_wm2real.sh outputs/task_success_rate/wm/webshop/
bash eval/02_task_success_rate/run_real.sh     # real-env baseline
python eval/02_task_success_rate/analyze_pairwise_cr.py \
    --real-dir outputs/task_success_rate/real/webshop/<exp> \
    --w2r-dir  outputs/task_success_rate/w2r/webshop/<exp>

# Metric 3 — Step-level Behavior Consistency
bash eval/03_behavior_consistency/run_eval_bf.sh
```

Full details and TextWorld variants live in [docs/EVALUATION.md](docs/EVALUATION.md).

## Training

We do **not** ship training launch scripts (paths & cluster-specific arguments
differ across setups). Instead, we document the verl GRPO configuration used to
produce the paper numbers, and provide the reward code so you can plug BehR into
your own launch command.

### Reward plug-in

Point verl's `custom_reward_function` at one of:

- [`src/reward/behr_reward_webshop.py`](src/reward/behr_reward_webshop.py) —
  BehR + physical-facts reward for WebShop
- [`src/reward/behr_reward_textworld.py`](src/reward/behr_reward_textworld.py) —
  pure BehR for TextWorld

Both expose `compute_score(data_source, solution_str, ground_truth, extra_info)`
matching the verl reward-manager signature.

### Key training hyper-parameters (GRPO, verl)

| Group | Parameter | Value |
|-------|-----------|-------|
| Algorithm | `algorithm.adv_estimator` | `grpo` |
| Algorithm | `actor_rollout_ref.rollout.n` | `8` (group size) |
| Algorithm | KL loss coefficient | `0.001` |
| Data | `data.train_batch_size` | `128` |
| Data | `data.max_prompt_length` | `8192` |
| Data | `data.max_response_length` | `1024` |
| Optim | learning rate (actor) | `1e-6` |
| Optim | warmup ratio | `0.0` |
| Rollout | `actor_rollout_ref.rollout.temperature` | `1.0` |
| Rollout | `actor_rollout_ref.rollout.top_p` | `1.0` |
| Reward | `reward_model.reward_manager` | `batch` |
| Reward | `custom_reward_function.path` | `src/reward/behr_reward_webshop.py` |
| Reward | `custom_reward_function.name` | `compute_score` |
| Reward | `behavior_weight` / `facts_weight` | `0.8` / `0.2` (WebShop) |
| Reward | `behavior_scale_coef` ($\alpha$) | `1.0` |
| Reward | reward mode | `cauchy` (recommended) |
| Hardware | world size | 4×A100-80GB or 8×A100-80GB |

A ready-to-edit template lives at
[`configs/train_config.yaml`](configs/train_config.yaml); the end-to-end walk-through
(reference-agent server, data registration, judge-model choice) is in
[docs/TRAINING.md](docs/TRAINING.md).

## The BehR Reward

BehR measures how much a frozen Reference Agent's likelihood of the logged next
action shifts between the real state $s_{\text{real}}$ and the world-model-predicted
state $s_{\text{pred}}$:

$$\Delta = \frac{1}{N}\sum_{i=1}^{N}\log\pi_\theta(a_i\mid s_{\text{pred}}) - \frac{1}{N}\sum_{i=1}^{N}\log\pi_\theta(a_i\mid s_{\text{real}})$$

$$R_{\text{BehR}}^{\text{cauchy}} = \frac{1}{1 + \alpha \cdot |\Delta|}$$

We use **mean** (not sum) log-probability to remove length bias, and a
**Cauchy**-shaped transform to keep the reward in $(0, 1]$ with non-vanishing
gradients even for large $|\Delta|$. For WebShop we additionally add a small
physical-facts term (weight 0.2) that grounds predicted states in ASIN / price /
page / rating mentions from the real env.

## Evaluation Metrics at a Glance

| Metric | What it measures | Where |
|--------|------------------|-------|
| EM | Single-step token-level match | `eval/01_single_step_accuracy/` |
| SR<sub>WM</sub> | Agent success rolling out inside the WM | `eval/02_task_success_rate/run_wm.sh` |
| SR<sub>W2R</sub> | WM-generated action replay in real env | `eval/02_task_success_rate/run_wm2real.sh` |
| SR<sub>Real</sub> | Agent success in real env (baseline) | `eval/02_task_success_rate/run_real.sh` |
| CR | SR<sub>W2R</sub> / SR<sub>Real</sub> | `analyze_pairwise_cr.py`, `compute_cr.py` |
| CR<sub>pw</sub> | Pairwise overlap of successful tasks | `analyze_pairwise_cr.py` |
| BehR | Mean-log-prob shift under Reference Agent | `eval/03_behavior_consistency/` |

## Data

- **WebShop** ([Yao et al., 2022](https://arxiv.org/abs/2207.01206)) — e-commerce web navigation.
- **TextWorld** ([Côté et al., 2019](https://arxiv.org/abs/1806.11532)) — text-based adventure games.
- Evaluation uses 200 standardized test tasks from [AgentGym](https://github.com/WooooDyy/AgentGym)
  per environment; trajectory datasets are produced by our reference agent against a frozen real env.

Init contexts, trajectory datasets, and trained world-model checkpoints are
served from HuggingFace Hub via `scripts/download_data.py`. Links appear under
[Release Timeline](#release-timeline) as each artifact ships.

## Documentation

- [Installation guide](docs/INSTALL.md)
- [Evaluation guide](docs/EVALUATION.md)
- [Training guide](docs/TRAINING.md)

## Citation

The paper is currently under review. A citation entry will be added here once
the preprint is public. A provisional BibTeX key lives in
[`CITATION.cff`](CITATION.cff).

## License

Code is released under the Apache License 2.0 — see [LICENSE](LICENSE).
WebShop, TextWorld, and AgentGym assets retain their upstream licenses.

## Acknowledgments

BehR-WM builds on [verl](https://github.com/volcengine/verl) (GRPO training
framework), [vLLM](https://github.com/vllm-project/vllm) (high-throughput
inference), [WebShop](https://github.com/princeton-nlp/webshop), and
[TextWorld](https://github.com/microsoft/TextWorld). We thank the authors of
these projects.
