#!/bin/bash
# =============================================================================
# WebShop Environment Server
# =============================================================================
#
# Usage:
#   bash start_webshop_env.sh [port]
#
# Default port: Random (30000-99999)
# =============================================================================

set -e
source uv_webshop/bin/activate

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Port configuration
PORT="${1:-$((30000 + RANDOM % (99999-30000+1)))}"

# Activate WebShop environment
if [ -f "$PROJECT_ROOT/uv_webshop/bin/activate" ]; then
    source "$PROJECT_ROOT/uv_webshop/bin/activate"
else
    echo "Warning: uv_webshop environment not found."
    echo "Please run: bash scripts/env_setup/uv_webshop.sh"
fi

echo "=========================================="
echo "Starting WebShop Environment Server"
echo "=========================================="
echo "Port: $PORT"
echo "=========================================="

webshop --host 0.0.0.0 --port $PORT
