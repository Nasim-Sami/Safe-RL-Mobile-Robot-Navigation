#!/usr/bin/env bash
# ==============================================================================
# Pure End-to-End Safe Navigation Agent Runner (360 Rays, Zero Heuristic Rules)
# ==============================================================================

set -e

# Source ROS 2
source /opt/ros/lyrical/setup.bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXEC="/home/sami/mp_env/bin/python3"

ALGO=${1:-"ppo_lagrangian"}

echo "======================================================================"
echo "🧠 Launching Pure Neural Safe Navigation Controller: ${ALGO^^}"
echo "⚡ Full 360 LiDAR Rays | 100% Policy-Driven Decisions | 0 Heuristics"
echo "======================================================================"

export CUDA_VISIBLE_DEVICES=0
$PYTHON_EXEC "$SCRIPT_DIR/deploy_node_360.py" --algo "$ALGO"
