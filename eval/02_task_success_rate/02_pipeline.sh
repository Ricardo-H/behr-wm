#!/bin/bash
# =============================================================================
# Task Success Rate - Full Pipeline (WM → W2R → Real)
# =============================================================================
#
# Usage:
#   TASK=webshop   bash eval/02_task_success_rate/02_pipeline.sh [output_name]
#   TASK=textworld bash eval/02_task_success_rate/02_pipeline.sh [output_name]
#
# Supported tasks: webshop, textworld, alfworld, sciworld
# Default: webshop
# =============================================================================

TASK="${TASK:-webshop}"
OUTPUT_NAME="${1:-Qwen3-8B}"

echo "Running full eval pipeline for TASK=$TASK, OUTPUT=$OUTPUT_NAME"

# 2.1 WM - Agent 在世界模型中的成功率
TASK=$TASK bash eval/02_task_success_rate/run_wm.sh "$OUTPUT_NAME"

# 2.2 W2R - WM 动作序列在真实环境重放成功率
TASK=$TASK bash eval/02_task_success_rate/run_wm2real.sh "outputs/task_success_rate/wm/$TASK/$OUTPUT_NAME"

# 2.3 Real - Agent 在真实环境中的成功率
TASK=$TASK bash eval/02_task_success_rate/run_real.sh "$OUTPUT_NAME"