#!/usr/bin/env bash
# ==============================================================================
# All-In-One Simulation Launcher: Gazebo Sim + Robot + Bridge + Costmap + RViz2
# Everything runs with NVIDIA GPU Hardware Acceleration
# ==============================================================================

set -e

# Source ROS 2
source /opt/ros/lyrical/setup.bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_WS="$(dirname "$PARENT_DIR")"
ROOT_ROS2="$ROOT_WS/ros2"
BRINGUP_DIR="/opt/ros/lyrical/share/nav2_minimal_tb3_sim"

echo "======================================================================"
echo "🚀 Starting Gazebo 3D Arena + RViz2 + TurtleBot3 (NVIDIA GPU Accelerated)"
echo "======================================================================"

# 1. Resource path for Gazebo models
export GZ_SIM_RESOURCE_PATH="$BRINGUP_DIR/models:$BRINGUP_DIR/worlds:$GZ_SIM_RESOURCE_PATH"

# 2. Launch Gazebo with TurtleBot3 embedded in the arena (GPU Accelerated)
echo "[1/4] Launching Gazebo 3D Arena on dedicated GPU..."
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia gz sim -r "$ROOT_ROS2/tb3_arena.sdf" &
GZ_PID=$!

sleep 3

# 3. Launch ROS-Gazebo Bridge & Robot State Publisher (TF)
echo "[2/4] Starting ROS-Gazebo Bridge and 3D Robot State Publisher..."
ros2 launch "$ROOT_ROS2/bridge_and_tf.launch.py" &
BRIDGE_PID=$!

sleep 2

# 4. Launch Real-Time 2D Costmap Generator
echo "[3/4] Starting Real-Time 2D Costmap Publisher..."
/home/sami/mp_env/bin/python3 "$ROOT_ROS2/costmap_generator.py" &
COSTMAP_PID=$!

sleep 1

# 5. Launch RViz2 (GPU Accelerated)
echo "[4/4] Launching RViz2 Visualizer..."
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia rviz2 -d "$ROOT_ROS2/tb3_view.rviz" &
RVIZ_PID=$!

# Trap Ctrl+C to clean up all background processes safely
trap 'echo -e "\n🛑 Shutting down Gazebo, Bridge, Costmap, and RViz2..."; kill $GZ_PID $BRIDGE_PID $COSTMAP_PID $RVIZ_PID 2>/dev/null; exit 0' SIGINT SIGTERM

echo "======================================================================"
echo "✅ Simulation Suite running! Press Ctrl+C in this terminal to stop all."
echo "======================================================================"

wait
