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

# Resolve script location and project root before any activation attempt.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Port configuration
PORT="${1:-$((30000 + RANDOM % (99999-30000+1)))}"

# Activate the project's virtual environment.
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.venv/bin/activate"
else
    echo "Warning: .venv environment not found."
    echo "Please run: bash scripts/env_setup/install_env.sh"
    exit 1
fi

echo "=========================================="
echo "Starting WebShop Environment Server"
echo "=========================================="
echo "Port: $PORT"
echo "=========================================="

webshop --host 0.0.0.0 --port $PORT
