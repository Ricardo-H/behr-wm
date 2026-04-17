# Installation Guide

BehR-WM ships a single installer that covers the evaluation environment. For
GRPO training, run the additional verl installer after the base install.

## Prerequisites

- Linux (Ubuntu 22.04 tested)
- Python 3.10
- CUDA 12.6 (or a compatible driver)
- A GPU with ≥ 40 GB VRAM for evaluation (≥ 4 × A100-80GB recommended for training)
- `git`, and `sudo`-accessible `apt-get` (the installer needs `unzip` and `openjdk-11-jdk`)

## Step 1 — Clone the repository

```bash
git clone https://github.com/Ricardo-H/behr-wm.git
cd behr-wm
```

## Step 2 — Install the evaluation environment

```bash
bash scripts/env_setup/install_env.sh
```

This one-shot installer will:

1. Install system packages (`unzip`, `openjdk-11-jdk`).
2. Install [`uv`](https://github.com/astral-sh/uv) if missing.
3. Shallow-clone [AgentGym](https://github.com/WooooDyy/AgentGym) into
   `./AgentGym/` (provides the WebShop + TextWorld backends).
4. Create a Python 3.10 virtual environment at `./.venv/`.
5. Install the pinned ML stack (PyTorch 2.8.0 + CUDA 12.6, vLLM 0.11.0,
   Flash Attention 2.8.3, FlashInfer 0.3.1).
6. Install the WebShop backend (`agentenv-webshop`) and the top-level
   `agentenv` controller needed by the eval scripts.
7. Install the remaining Python dependencies from `requirements.txt`.

Activate the environment afterwards:

```bash
source .venv/bin/activate
```

## Step 3 — (Optional) Install the training framework

Only needed for GRPO training via [verl](https://github.com/volcengine/verl):

```bash
bash scripts/env_setup/install_verl.sh
```

## Step 4 — Verify the installation

```bash
source .venv/bin/activate

python -c "import torch; print(f'PyTorch:       {torch.__version__}')"
python -c "import vllm;  print(f'vLLM:          {vllm.__version__}')"
python -c "import transformers; print(f'Transformers:  {transformers.__version__}')"

# AgentGym packages
python -c "from agentenv.envs import WebshopTask; print('agentenv.envs:      OK')"
python -c "from agentenv.controller import APIAgent, Evaluator; print('agentenv.controller: OK')"
```

## Troubleshooting

### CUDA mismatch

```bash
nvcc --version
nvidia-smi
# If you need a different CUDA build:
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu124
```

### Java not found

WebShop's search engine needs Java 11:

```bash
sudo apt install openjdk-11-jdk -y
java -version
```

### Out-of-memory during evaluation

- Lower `--max-concurrency` in the eval scripts (default: 64 on local vLLM).
- Reduce `max_model_len` when starting vLLM servers.

### `agentenv` import fails

The installer clones AgentGym to `./AgentGym/` and installs it editably into
`.venv`. If you see `ModuleNotFoundError: No module named 'agentenv'`, re-run:

```bash
source .venv/bin/activate
pip install -e AgentGym/agentenv
```

## Next steps

- [EVALUATION.md](EVALUATION.md) — run the 3-stage evaluation pipeline.
- [TRAINING.md](TRAINING.md) — reproduce the GRPO training runs.
