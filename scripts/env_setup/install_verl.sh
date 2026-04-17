#!/bin/bash
# =============================================================================
# Install verl (GRPO training framework)
# =============================================================================
# Prerequisite: the main environment must already be installed:
#     bash scripts/env_setup/install_env.sh
# =============================================================================

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Installing verl (GRPO training framework)"
echo "=========================================="

if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: .venv not found. Run scripts/env_setup/install_env.sh first.${NC}"
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo -e "\n${GREEN}[1/2] Installing verl from PyPI ...${NC}"
pip install verl

echo -e "\n${GREEN}[2/2] Installing training-side helpers ...${NC}"
pip install wandb hydra-core codetiming 2>/dev/null || true

echo -e "\n${GREEN}==========================================${NC}"
echo -e "${GREEN}  verl installed.${NC}"
echo -e "${GREEN}==========================================${NC}"
python -c "import verl; print(f'  verl version: {verl.__version__}')" 2>/dev/null || echo "  verl installed"

cat <<'EOS'

Next: launch GRPO training (4-GPU default). Before running, start the
Reference Agent server and edit the data paths inside the script:

    bash scripts/servers/start_reference_agent_server.sh -m Qwen/Qwen3-8B -p 8000 -gpu 0,1,2,3 --shared
    bash train/run_grpo_4gpu.sh

See docs/TRAINING.md for 8-GPU / TextWorld variants.
EOS
