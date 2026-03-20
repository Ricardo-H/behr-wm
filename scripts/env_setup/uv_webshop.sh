# !/bin/bash
# WebShop 评估环境安装脚本
# 此脚本安装 WebShop 环境及所有评估/训练依赖
# 
# 版本配置 (2026-02, verl 官方推荐):
#   Python: 3.10 (spacy 兼容性)
#   PyTorch: 2.8.0+cu126
#   vLLM: 0.11.0
#   Flash Attention: 2.8.3
#   FlashInfer: 0.3.1
set -ex

# 确保在 behr-wm 根目录运行
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# 安装系统依赖
sudo apt-get update
# gh: 推送代码到 GitHub
# unzip: 解压评估环境数据 (alfworld/textworld/webshop)
sudo apt-get install -y gh unzip

# 安装 uv (如果没有)
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 创建 WebShop 虚拟环境
uv venv uv_webshop --python 3.10
source uv_webshop/bin/activate
uv pip install --upgrade pip
export UV_LINK_MODE=copy

# AgentGym/agentenv-webshop 已包含在仓库中
cd AgentGym/agentenv-webshop/webshop
# ./setup.sh -d all

# Install Python Dependencies (verl 官方推荐版本)
uv pip install packaging wheel setuptools "numpy<2.0.0" psutil ninja

# === 核心 ML 框架安装 ===
# 注意: vLLM 会尝试安装自己的 torch 依赖，必须用 --no-deps 阻止
# 安装顺序: vLLM (--no-deps) -> torch 2.8.0 -> flash_attn (针对 torch 2.8 编译)
uv pip install vllm==0.11.0 --no-deps
# vLLM 必要依赖 (排除 torch/torchvision/torchaudio，这些我们手动指定版本)
uv pip install aiohttp blake3 cachetools cbor2 cloudpickle compressed-tensors depyf \
    diskcache einops fastapi filelock gguf lark llguidance lm-format-enforcer \
    mistral_common msgspec ninja numba openai openai-harmony opencv-python-headless \
    outlines_core partial-json-parser pillow prometheus-fastapi-instrumentator \
    prometheus_client protobuf py-cpuinfo pybase64 pydantic python-json-logger \
    python-multipart uvicorn \
    pyyaml pyzmq ray regex scipy sentencepiece setproctitle tiktoken tokenizers \
    tqdm typing_extensions watchfiles xformers xgrammar

# 安装指定版本的 PyTorch (cu126)
uv pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Flash Attention (必须在 torch 2.8.0 安装后)
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
uv pip install flashinfer-python==0.3.1
uv pip install "tensordict>=0.8.0,<=0.10.0,!=0.9.0"
uv pip install -r requirements.txt --no-deps

# conda install mkl
# conda install -c conda-forge faiss-cpu
uv pip install mkl faiss-cpu
uv pip install gdown


# Install Environment Dependencies via `conda`
# conda install -c pytorch faiss-cpu;
# conda install -c conda-forge openjdk=11;
sudo apt-get update
sudo apt install openjdk-11-jdk -y

## We have packed into data/webshop.zip
# # Download dataset into `data` folder via `gdown` command
# mkdir -p data;
# cd data;
# gdown https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib; # items_shuffle_1000 - product scraped info
# gdown https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu; # items_ins_v2_1000 - product attributes
# gdown https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB; # items_shuffle
# gdown https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi; # items_ins_v2
# gdown https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O # items_human_ins
# cd ..

# spacy 3.7.2 requires typer<0.10.0,>=0.3.0, but you have typer 0.15.2 which is incompatible.
# weasel 0.3.4 requires typer<0.10.0,>=0.3.0, but you have typer 0.15.2 which is incompatible.
# The warnings can be safely ignored.

# Build search engine index
uv pip install nltk itsdangerous pyjnius pytz python-dateutil huggingface_hub threadpoolctl onnxruntime setuptools
cd search_engine

## We have packed into data/webshop_index.zip
# mkdir -p resources resources_100 resources_1k resources_100k
# python convert_product_file_format.py # convert items.json => required doc format
# mkdir -p indexes
# ./run_indexing.sh

uv pip install openai azure.identity
uv pip install omegaconf

cd ../..
uv pip install -e .
uv pip install thinc numpy==1.26.4 langcodes spacy==3.7.1
python -m pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
python -m pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl"

# === 安装完整的 agentenv 包 (包含 controller 和 envs 模块) ===
# agentenv-webshop 只提供 WebShop 服务器，eval 脚本需要完整的 agentenv 包
cd "$SCRIPT_DIR/../.."
uv pip install -e AgentGym/agentenv

# === eval 依赖 ===
# 01_single_step_accuracy: vllm, transformers, matplotlib
# 02_task_success_rate: agentenv (controller, envs), openai, tqdm
# 03_behavior_consistency: requests, aiohttp, tqdm
# transformers 5.x 与 vLLM 0.11.0 不兼容，必须使用 4.x
uv pip install matplotlib aiohttp tqdm requests "transformers>=4.45.0,<5.0.0" orjson

# === 最终验证和修复 torch 版本 ===
# 某些包可能会意外降级 torch，这里确保最终版本正确
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not installed")
if [[ "$TORCH_VERSION" != 2.8.0* ]]; then
    echo "警告: torch 版本 ($TORCH_VERSION) 不正确，重新安装 torch 2.8.0..."
    uv pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
fi

# === 最终修复 numpy 版本 ===
# spacy 3.7.x + thinc 8.2.x 的 C 扩展与 numpy 2.x 不兼容
# 某些包 (如 transformers, matplotlib) 会把 numpy 升级回 2.x，这里确保最终版本正确
uv pip install "numpy>=1.26.0,<2.0.0"

# === 验证安装 ===
echo ""
echo "=========================================="
echo "  环境安装完成，验证核心包版本:"
echo "=========================================="
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import vllm; print(f'  vLLM: {vllm.__version__}')"
python -c "import flash_attn; print(f'  Flash Attention: {flash_attn.__version__}')"
echo "=========================================="
echo ""
echo "下一步: 安装 verl 训练框架"
echo "  bash scripts/env_setup/install_verl.sh"
echo ""
