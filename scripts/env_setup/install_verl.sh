#!/bin/bash
# BehR-WM verl framework installation (GRPO training)
# Usage: bash scripts/env_setup/install_verl.sh
# 
# Prerequisite: Run uv_webshop.sh first to set up the evaluation environment

set -e

echo "=========================================="
echo "  Installing verl framework (GRPO Training)"
echo "=========================================="

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Activate evaluation environment
if [ -d "uv_webshop" ]; then
    source uv_webshop/bin/activate
else
    echo -e "${RED}Error: Please run uv_webshop.sh first${NC}"
    echo "  bash scripts/env_setup/uv_webshop.sh"
    exit 1
fi

# Install verl from PyPI
echo -e "\n${GREEN}[1/2] Installing verl framework from PyPI...${NC}"
pip install verl

# Install additional training dependencies
echo -e "\n${GREEN}[2/2] Installing training dependencies...${NC}"
pip install wandb hydra-core codetiming 2>/dev/null || true

echo -e "\n${GREEN}=========================================="
echo "  verl installation complete!"
echo "==========================================${NC}"
echo ""
echo "Verify installation:"
python -c "import verl; print(f'  verl version: {verl.__version__}')" || echo "  verl installed"
echo ""
echo "Now you can run GRPO training:"
echo "  # 4 GPU setup"
echo "  1. Start Reference Agent: bash scripts/servers/start_reference_agent_server.sh -m Qwen/Qwen3-8B -p 8000 -gpu 0,1,2,3 --shared"
echo "  2. Run training: bash train/4gpu/run_grpo.sh"
echo ""
echo "  # 8 GPU setup (recommended)"
echo "  1. Start Reference Agent: bash scripts/servers/start_reference_agent_server.sh -m Qwen/Qwen3-8B -p 8000 -gpu 0,1,2,3,4,5,6,7 --shared"
echo "  2. Run training: bash train/8gpu/run_grpo.sh"
