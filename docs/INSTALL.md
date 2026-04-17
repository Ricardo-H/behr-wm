# Installation Guide

## Prerequisites

- Python 3.10+
- CUDA 12.x
- GPU with at least 40GB VRAM (for training)

## Step 1: Clone Repository

```bash
git clone https://github.com/Ricardo-H/behr-wm.git
cd behr-wm
```

## Step 2: Install Evaluation Environment

```bash
bash scripts/env_setup/uv_webshop.sh
```

This will:
- Create a virtual environment (`uv_webshop/`)
- Install Python dependencies (PyTorch, vLLM, Flash Attention, etc.)
- Install Java (required for WebShop search engine)
- Install AgentEnv evaluation framework

## Step 3: Install Training Framework (Optional)

For GRPO training, install the verl framework:

```bash
source uv_webshop/bin/activate
pip install verl
```

## Step 4: Verify Installation

```bash
source uv_webshop/bin/activate

# Check Python packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Check WebShop environment
python -c "from agentenv.envs import WebshopTask; print('WebShop OK')"
python -c "from agentenv.controller import APIAgent, Evaluator; print('AgentEnv Controller OK')"
```

## Troubleshooting

### CUDA Issues

```bash
# Check CUDA version
nvcc --version
nvidia-smi

# Reinstall PyTorch for your CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Java Issues

WebShop's search engine requires Java 11:

```bash
# Ubuntu/Debian
sudo apt install openjdk-11-jdk -y
java -version
```

### Memory Issues

If you run out of GPU memory during training:

1. Reduce batch size in config
2. Use gradient checkpointing
3. Increase tensor parallel size

## Next Steps

- [TRAINING.md](TRAINING.md) - Training guide
- [EVALUATION.md](EVALUATION.md) - Evaluation guide
