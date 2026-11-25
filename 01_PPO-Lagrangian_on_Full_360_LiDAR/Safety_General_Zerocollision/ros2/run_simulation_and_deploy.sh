#!/bin/bash
set -e

# 1. Clean symlink without spaces or '&'
CLEAN_WS="/home/sami/ppo_safe_ws"
if [ ! -d "$CLEAN_WS" ]; then
    ln -sfn "/home/sami/Documents/PPO & PPO-Lag" "$CLEAN_WS"
fi

SCRIPT_DIR="$CLEAN_WS/SafeRL_General_ZeroCollision/ros2"
WS_DIR="$CLEAN_WS/SafeRL_General_ZeroCollision"

echo "=========================================================================="
echo "🤖 Generalizable 360-Ray Zero-Collision Safe Navigation (Gazebo + RViz2)"
echo "=========================================================================="

source /opt/ros/lyrical/setup.bash
source /home/sami/mp_env/bin/activate
export PYTHONPATH="$WS_DIR:$PYTHONPATH"

# Crucial Gazebo resource path for 3D meshes (hexagon, walls, pillars)
export GZ_SIM_RESOURCE_PATH="/opt/ros/lyrical/share/nav2_minimal_tb3_sim/models:$GZ_SIM_RESOURCE_PATH"
export IGN_GAZEBO_RESOURCE_PATH="/opt/ros/lyrical/share/nav2_minimal_tb3_sim/models:$IGN_GAZEBO_RESOURCE_PATH"

ALGO="${1:-ppo_lagrangian}"

# Clean any existing orphan processes
pkill -9 -f "gz-sim" 2>/dev/null || true
pkill -9 -f "parameter_bridge" 2>/dev/null || true
pkill -9 -f "costmap_generator" 2>/dev/null || true
pkill -9 -f "rviz2" 2>/dev/null || true
pkill -9 -f "deploy_node" 2>/dev/null || true
sleep 1

echo "1. Launching Gazebo Sim Arena (NVIDIA Discrete GPU)..."
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia gz sim -r "$CLEAN_WS/ros2/tb3_arena.sdf" &
GZ_PID=$!
sleep 5

echo "2. Launching ROS-Gazebo Parameter Bridge & TF..."
ros2 launch "$CLEAN_WS/ros2/bridge_and_tf.launch.py" &
BRIDGE_PID=$!
sleep 2

echo "3. Launching Real-time 2D Costmap Generator..."
python3 "$CLEAN_WS/ros2/costmap_generator.py" &
COSTMAP_PID=$!
sleep 1

echo "4. Launching RViz2 with 2D Goal Pose..."
rviz2 -d "$CLEAN_WS/ros2/tb3_view.rviz" &
RVIZ_PID=$!
sleep 2

echo "5. Starting Pure End-to-End Neural Policy Controller ($ALGO)..."
python3 "$SCRIPT_DIR/deploy_node_general.py" --algo "$ALGO" &
DEPLOY_PID=$!

trap "kill -9 $GZ_PID $BRIDGE_PID $COSTMAP_PID $RVIZ_PID $DEPLOY_PID 2>/dev/null || true" EXIT INT TERM

wait
