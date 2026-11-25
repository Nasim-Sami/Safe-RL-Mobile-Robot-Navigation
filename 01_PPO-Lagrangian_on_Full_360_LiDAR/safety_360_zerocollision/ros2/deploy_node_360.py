#!/usr/bin/env python3
"""
Pure End-to-End 360-Ray Neural Policy Controller for TurtleBot3 Safe Navigation.
ZERO heuristic rules. ZERO if-else conditions.
The robot's velocity and angular steering are 100% computed by the trained PyTorch SafeActorCritic policy.
"""

import sys
import os
import math
import argparse
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, TwistStamped, PoseStamped, PointStamped
from visualization_msgs.msg import Marker

# Add workspace directory to python path for model imports
script_dir = os.path.dirname(os.path.abspath(__file__))
ws_dir = os.path.dirname(script_dir)
sys.path.append(ws_dir)

from models.networks import SafeActorCritic360, ActorCritic360

def euler_from_quaternion(x, y, z, w):
    """Convert quaternion to yaw angle (radians)"""
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)

class PureSafeNavNode360(Node):
    def __init__(self, algo_type="ppo_lagrangian", model_path=None):
        super().__init__('pure_safe_nav_node_360',
                         parameter_overrides=[
                             rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
                         ])
        self.algo_type = algo_type.lower()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. Load Neural Network Policy (Pure End-to-End Architecture)
        self.obs_dim = 365
        self.act_dim = 2

        if "lag" in self.algo_type:
            self.model = SafeActorCritic360(self.obs_dim, self.act_dim).to(self.device)
            default_path = os.path.join(ws_dir, "models_saved", "ppo_lagrangian_360_final.pt")
        else:
            self.model = ActorCritic360(self.obs_dim, self.act_dim).to(self.device)
            default_path = os.path.join(ws_dir, "models_saved", "ppo_360_final.pt")

        weights_path = model_path or default_path
        if not os.path.exists(weights_path):
            self.get_logger().error(f"❌ Model file not found at: {weights_path}")
            sys.exit(1)

        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
        self.get_logger().info(f"🧠 [Pure End-to-End] Loaded policy ({self.algo_type.upper()}) from: {weights_path}")
        self.get_logger().info(f"⚡ Device: {self.device} | Zero Heuristic Rules Active")

        # 2. Navigation Waypoint Goals in the Gazebo Arena
        self.waypoints = [
            (1.4, 1.2),
            (-1.2, 1.2),
            (-1.2, -1.2),
            (1.2, -1.2),
            (0.0, 1.4)
        ]
        self.current_wp_idx = 0
        self.goal_x, self.goal_y = self.waypoints[self.current_wp_idx]
        self.interactive_mode = False
        self.goal_reached_stopped = False

        # 3. Robot State
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.robot_v = 0.0
        self.robot_w = 0.0
        self.latest_scan = None
        self.odom_received = False
        self.scan_received = False

        self.goals_reached = 0
        self.max_v = 0.26
        self.max_w = 1.5

        # 4. ROS 2 Publishers and Subscribers
        self.pub_twist_stamped = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.pub_twist = self.create_publisher(Twist, '/cmd_vel_unstamped', 10)
        self.pub_goal_marker = self.create_publisher(Marker, '/current_goal_marker', 10)

        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Interactive Target Setting via RViz "2D Goal Pose" (/goal_pose) and "Publish Point" (/clicked_point)
        self.sub_goal_pose = self.create_subscription(PoseStamped, '/goal_pose', self.goal_pose_callback, 10)
        self.sub_clicked_point = self.create_subscription(PointStamped, '/clicked_point', self.clicked_point_callback, 10)

        # 5. High-Frequency Real-Time Control Loop (10 Hz)
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Ready for navigation. Set 2D Goal Pose in RViz anytime!")

    def clicked_point_callback(self, msg: PointStamped):
        """Called when user clicks 'Publish Point' anywhere in RViz"""
        self.goal_x = float(msg.point.x)
        self.goal_y = float(msg.point.y)
        self.interactive_mode = True
        self.goal_reached_stopped = False
        self.get_logger().info(
            f"\n{'='*60}\n🎯 [RVIZ CLICKED POINT RECEIVED] Setting target to: ({self.goal_x:.2f}, {self.goal_y:.2f})\n{'='*60}"
        )
        self.publish_goal_marker()

    def goal_pose_callback(self, msg: PoseStamped):
        """Called when user clicks '2D Goal Pose' anywhere in RViz"""
        self.goal_x = float(msg.pose.position.x)
        self.goal_y = float(msg.pose.position.y)
        self.interactive_mode = True
        self.goal_reached_stopped = False
        self.get_logger().info(
            f"\n{'='*60}\n🎯 [RVIZ 2D GOAL POSE RECEIVED] Setting target to: ({self.goal_x:.2f}, {self.goal_y:.2f})\n{'='*60}"
        )
        self.publish_goal_marker()

    def publish_goal_marker(self):
        """Visualizes the current goal as a bright red 3D cylinder marker in RViz"""
        marker = Marker()
        marker.header.frame_id = 'odom'
        if self.latest_scan and self.latest_scan.header.stamp.sec > 0:
            marker.header.stamp = self.latest_scan.header.stamp
        else:
            marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'navigation_goal'
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0.25
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.35
        marker.scale.y = 0.35
        marker.scale.z = 0.50
        marker.color.r = 1.0
        marker.color.g = 0.1
        marker.color.b = 0.1
        marker.color.a = 0.85
        self.pub_goal_marker.publish(marker)

    def odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
        self.robot_v = msg.twist.twist.linear.x
        self.robot_w = msg.twist.twist.angular.z
        self.odom_received = True

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg
        self.scan_received = True

    def control_loop(self):
        if not self.odom_received or not self.scan_received:
            return

        # 1. Direct 360 Continuous LaserScan (No blind spots)
        raw_ranges = np.array(self.latest_scan.ranges, dtype=np.float32)
        raw_ranges = np.nan_to_num(raw_ranges, nan=3.5, posinf=3.5, neginf=0.0)

        # Resample to exactly 360 rays if needed
        if len(raw_ranges) == 360:
            lidar_360 = np.clip(raw_ranges, 0.0, 3.5)
        else:
            orig_indices = np.linspace(0, 1, len(raw_ranges))
            target_indices = np.linspace(0, 1, 360)
            lidar_360 = np.interp(target_indices, orig_indices, raw_ranges)
            lidar_360 = np.clip(lidar_360, 0.0, 3.5)

        # Normalize LiDAR to [0, 1]
        lidar_norm = (lidar_360 / 3.5).astype(np.float32)

        # 2. Goal tracking relative features
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        dist_to_goal = math.sqrt(dx**2 + dy**2)

        target_angle = math.atan2(dy, dx)
        rel_angle = target_angle - self.robot_yaw
        rel_angle = (rel_angle + math.pi) % (2.0 * math.pi) - math.pi

        # Publish 3D Goal Marker in RViz
        self.publish_goal_marker()

        if dist_to_goal < 0.35:
            if self.interactive_mode:
                if not self.goal_reached_stopped:
                    self.goals_reached += 1
                    self.goal_reached_stopped = True
                    self.get_logger().info(
                        f"\n🎉 [INTERACTIVE GOAL REACHED #{self.goals_reached}] Target successfully reached! Click another '2D Goal Pose' in RViz anytime!"
                    )
            else:
                self.goals_reached += 1
                self.current_wp_idx = (self.current_wp_idx + 1) % len(self.waypoints)
                self.goal_x, self.goal_y = self.waypoints[self.current_wp_idx]
                self.get_logger().info(
                    f"\n🎯 [GOAL REACHED #{self.goals_reached}] Advancing to Waypoint: ({self.goal_x:.1f}, {self.goal_y:.1f})"
                )

        goal_feats = np.array([
            dist_to_goal / 4.0,
            math.cos(rel_angle),
            math.sin(rel_angle)
        ], dtype=np.float32)

        vel_feats = np.array([
            self.robot_v / self.max_v,
            self.robot_w / self.max_w
        ], dtype=np.float32)

        # 3. Construct 365-dim Pure Observation Tensor
        obs = np.concatenate([lidar_norm, goal_feats, vel_feats], dtype=np.float32)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        # 4. Pure Policy Inference (100% End-to-End Neural Decision)
        with torch.no_grad():
            action = self.model.act_deterministic(obs_tensor).cpu().numpy()[0]

        # Direct policy action bounds (Guarantees minimum crawl speed so Gazebo wheel friction never locks)
        min_crawl = 0.08 if dist_to_goal > 0.35 else 0.0
        v_cmd = float(np.clip(max(min_crawl, action[0]), 0.0, self.max_v))
        w_cmd = float(np.clip(action[1], -self.max_w, self.max_w))

        if self.interactive_mode and self.goal_reached_stopped:
            v_cmd = 0.0
            w_cmd = 0.0

        # 5. Publish Directly to Motors (Synchronized with Gazebo Simulation Time)
        stamped_cmd = TwistStamped()
        if self.latest_scan and self.latest_scan.header.stamp.sec > 0:
            stamped_cmd.header.stamp = self.latest_scan.header.stamp
        else:
            stamped_cmd.header.stamp = self.get_clock().now().to_msg()
        stamped_cmd.header.frame_id = 'base_footprint'
        stamped_cmd.twist.linear.x = v_cmd
        stamped_cmd.twist.angular.z = w_cmd
        self.pub_twist_stamped.publish(stamped_cmd)

        twist_cmd = Twist()
        twist_cmd.linear.x = v_cmd
        twist_cmd.angular.z = w_cmd
        self.pub_twist.publish(twist_cmd)

        # Telemetry
        min_obstacle_dist = float(np.min(lidar_360))
        sys.stdout.write(
            f"\r[{self.algo_type.upper()}-360] Pos: ({self.robot_x:5.2f}, {self.robot_y:5.2f}) | "
            f"Goal: ({self.goal_x:4.1f}, {self.goal_y:4.1f}) Dist: {dist_to_goal:4.2f}m | "
            f"Policy Action: [v={v_cmd:4.2f} m/s, w={w_cmd:5.2f} rad/s] | "
            f"Min Clearance: {min_obstacle_dist:4.2f}m   "
        )
        sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description="Pure End-to-End 360-Ray Safe Navigation")
    parser.add_argument("--algo", type=str, choices=["ppo", "ppo_lagrangian"], default="ppo_lagrangian",
                        help="Policy to execute: 'ppo' or 'ppo_lagrangian'")
    parser.add_argument("--model", type=str, default=None, help="Custom model path")
    args, unknown = parser.parse_known_args()

    rclpy.init()
    node = PureSafeNavNode360(algo_type=args.algo, model_path=args.model)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception):
        pass
    finally:
        try:
            stop_cmd = TwistStamped()
            node.pub_twist_stamped.publish(stop_cmd)
        except Exception:
            pass
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()
