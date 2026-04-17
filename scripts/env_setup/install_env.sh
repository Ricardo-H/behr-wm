#!/bin/bash
# =============================================================================
# BehR-WM environment installer
# =============================================================================
# Sets up a Python 3.10 virtual environment at ./.venv with vLLM, verl-compatible
# PyTorch, Flash Attention, and the AgentGym backends (WebShop + TextWorld).
#
# Tested configuration (April 2026):
#   Python:          3.10
#   PyTorch:         2.8.0 + CUDA 12.6
#   vLLM:            0.11.0
#   Flash Attention: 2.8.3
#   FlashInfer:      0.3.1
#
# Usage:
#   bash scripts/env_setup/install_env.sh
#   source .venv/bin/activate
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ----------------------------------------------------------------------------
# 1. System packages
# ----------------------------------------------------------------------------
if command -v sudo >/dev/null 2>&1; then
    echo "[install_env] Installing system packages (unzip, openjdk-11-jdk) ..."
    sudo apt-get update -qq
    sudo apt-get install -y unzip openjdk-11-jdk >/dev/null
else
    echo "[install_env] 'sudo' not available; skipping apt-get step."
    echo "              Make sure unzip and openjdk-11-jdk are installed manually."
fi

# ----------------------------------------------------------------------------
# 2. uv (fast Python package manager)
# ----------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "[install_env] Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# ----------------------------------------------------------------------------
# 3. AgentGym (WebShop + TextWorld backends)
#    Shallow-cloned on first run; left untouched on subsequent runs.
# ----------------------------------------------------------------------------
if [ ! -d "AgentGym" ]; then
    echo "[install_env] Cloning AgentGym ..."
    git clone --depth 1 https://github.com/WooooDyy/AgentGym.git AgentGym
fi

# ----------------------------------------------------------------------------
# 4. Python virtual environment (.venv)
# ----------------------------------------------------------------------------
uv venv .venv --python 3.10
# shellcheck disable=SC1091
source .venv/bin/activate
export UV_LINK_MODE=copy
uv pip install --upgrade pip

# ----------------------------------------------------------------------------
# 5. Core ML stack (vLLM + Torch + Flash Attention)
# ----------------------------------------------------------------------------
uv pip install packaging wheel setuptools "numpy<2.0.0" psutil ninja

# vLLM pulls its own torch; install with --no-deps and pin versions ourselves.
uv pip install vllm==0.11.0 --no-deps
uv pip install \
    aiohttp blake3 cachetools cbor2 cloudpickle compressed-tensors depyf \
    diskcache einops fastapi filelock gguf lark llguidance lm-format-enforcer \
    mistral_common msgspec numba openai openai-harmony opencv-python-headless \
    outlines_core partial-json-parser pillow prometheus-fastapi-instrumentator \
    prometheus_client protobuf py-cpuinfo pybase64 pydantic python-json-logger \
    python-multipart uvicorn \
    pyyaml pyzmq ray regex scipy sentencepiece setproctitle tiktoken tokenizers \
    tqdm typing_extensions watchfiles xformers xgrammar

uv pip install torch==2.8.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

uv pip install \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
uv pip install flashinfer-python==0.3.1
uv pip install "tensordict>=0.8.0,<=0.10.0,!=0.9.0"

# Project requirements (everything else we declared).
uv pip install -r requirements.txt --no-deps

# ----------------------------------------------------------------------------
# 6. WebShop backend (AgentGym)
# ----------------------------------------------------------------------------
echo "[install_env] Installing WebShop backend ..."
pushd AgentGym/agentenv-webshop/webshop >/dev/null

uv pip install mkl faiss-cpu gdown
uv pip install nltk itsdangerous pyjnius pytz python-dateutil huggingface_hub \
    threadpoolctl onnxruntime setuptools
uv pip install openai "azure.identity<2" omegaconf

uv pip install -e .
uv pip install thinc "numpy==1.26.4" langcodes "spacy==3.7.1"
python -m pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
python -m pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl"
popd >/dev/null

# ----------------------------------------------------------------------------
# 7. AgentGym top-level controller (needed by eval/ scripts)
# ----------------------------------------------------------------------------
echo "[install_env] Installing AgentGym controller ..."
uv pip install -e AgentGym/agentenv

# ----------------------------------------------------------------------------
# 8. Extra eval/inference helpers
# ----------------------------------------------------------------------------
uv pip install matplotlib aiohttp tqdm requests \
    "transformers>=4.45.0,<5.0.0" orjson

# ----------------------------------------------------------------------------
# 9. Post-install version repairs
#    Some packages silently upgrade torch / numpy; pin them back explicitly.
# ----------------------------------------------------------------------------
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "missing")
if [[ "$TORCH_VERSION" != 2.8.0* ]]; then
    echo "[install_env] torch version drifted to $TORCH_VERSION; pinning back to 2.8.0 ..."
    uv pip install torch==2.8.0 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
fi
uv pip install "numpy>=1.26.0,<2.0.0"

# ----------------------------------------------------------------------------
# 10. Sanity check
# ----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "  Environment installed. Versions:"
echo "=========================================="
python -c "import torch; print(f'  PyTorch:         {torch.__version__}')"
python -c "import vllm;  print(f'  vLLM:            {vllm.__version__}')"
python -c "import flash_attn; print(f'  Flash Attention: {flash_attn.__version__}')"
python -c "import transformers; print(f'  Transformers:    {transformers.__version__}')"
python -c "from agentenv.envs import WebshopTask; print('  agentenv:        OK')"
echo "=========================================="
echo ""
echo "Activate the environment with:"
echo "    source .venv/bin/activate"
echo ""
echo "Next step (optional, for training):"
echo "    bash scripts/env_setup/install_verl.sh"
