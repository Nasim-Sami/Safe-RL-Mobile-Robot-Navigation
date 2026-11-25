import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SafeNav360Env(gym.Env):
    """
    High-Fidelity 360-Ray 2D Mobile Robot Safe Navigation Environment.
    
    Features:
      - Full 360 continuous LiDAR rays (1-degree resolution).
      - Realistic TurtleBot3 dynamics with linear/angular acceleration limits.
      - Exact Gazebo TurtleBot3 World layout (9 obstacle pillars + boundary) + Randomized Mode.
      - Strict Zero-Collision constraint:
        * Danger zone buffer (dist < 0.30m) accumulates smooth proximity cost.
        * Physical collision (dist < 0.18m) causes hard episode termination with collision penalty.
      - 100% end-to-end RL without any rule-based safety heuristics.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, randomize_obstacles=True, max_steps=500):
        super().__init__()
        self.num_rays = 360
        self.max_lidar_range = 3.5  # meters
        self.robot_radius = 0.15    # TurtleBot3 physical footprint radius (meters)
        self.safety_margin = 0.25   # Forward directional danger margin (meters)
        self.max_steps = max_steps
        self.dt = 0.1
        self.randomize_obstacles = randomize_obstacles

        # Acceleration and velocity limits (matching TurtleBot3 physical motors)
        self.max_v = 0.26          # m/s
        self.max_w = 1.5           # rad/s
        self.max_acc_v = 1.0       # m/s^2 -> max dv = 0.10 m/s per step
        self.max_acc_w = 3.2       # rad/s^2 -> max dw = 0.32 rad/s per step

        # Action: Continuous [target_v (0 to 0.26 m/s), target_w (-1.5 to 1.5 rad/s)]
        self.action_space = spaces.Box(
            low=np.array([0.0, -self.max_w], dtype=np.float32),
            high=np.array([self.max_v, self.max_w], dtype=np.float32),
            dtype=np.float32
        )

        # Observation Space:
        # - 360 normalized LiDAR rays [0, 1] (0 = collision, 1 = max range clear)
        # - Goal: [norm_distance, cos(rel_angle), sin(rel_angle)] (3)
        # - Current velocity: [v / max_v, w / max_w] (2)
        # Total observation dimension: 360 + 3 + 2 = 365
        self.obs_dim = self.num_rays + 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # Precompute ray angles relative to robot body: [0, 1, 2, ..., 359] degrees
        self.ray_angles = np.linspace(0.0, 2.0 * np.pi, self.num_rays, endpoint=False, dtype=np.float32)

        # State Variables
        self.robot_pos = np.zeros(2, dtype=np.float32)
        self.robot_heading = 0.0
        self.robot_v = 0.0
        self.robot_w = 0.0
        self.goal_pos = np.zeros(2, dtype=np.float32)
        self.prev_dist_to_goal = 0.0
        self.step_count = 0
        self.stall_steps = 0
        self.last_stall_pos = np.zeros(2, dtype=np.float32)

        # World geometry
        self.arena_radius = 2.0  # Hexagonal/Circular arena radius
        self.obstacles = []      # List of (x, y, radius)

    def _setup_obstacles(self):
        self.obstacles = []
        if not self.randomize_obstacles or np.random.rand() < 0.5:
            # Exact TurtleBot3 Gazebo World 3x3 pillar grid
            coords = [-0.6, 0.0, 0.6]
            for cx in coords:
                for cy in coords:
                    self.obstacles.append((cx, cy, 0.15))
        else:
            # Randomized obstacle positions for policy robustness
            num_obs = np.random.randint(6, 10)
            for _ in range(num_obs):
                pos = np.random.uniform(-1.2, 1.2, size=2).astype(np.float32)
                rad = np.random.uniform(0.12, 0.22)
                self.obstacles.append((float(pos[0]), float(pos[1]), float(rad)))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.robot_v = 0.0
        self.robot_w = 0.0

        self._setup_obstacles()

        # Spawn robot in safe location away from obstacles
        while True:
            # Spawn in open area of the arena
            r = np.random.uniform(0.5, 1.4)
            theta = np.random.uniform(-np.pi, np.pi)
            pos = np.array([r * np.cos(theta), r * np.sin(theta)], dtype=np.float32)
            
            # Check clearance from all obstacles
            min_dist = min([np.linalg.norm(pos - np.array([ox, oy])) - orad for ox, oy, orad in self.obstacles])
            if min_dist > 0.35:
                self.robot_pos = pos
                break

        self.robot_heading = float(np.random.uniform(-np.pi, np.pi))

        # Spawn goal away from robot and obstacles
        while True:
            r_g = np.random.uniform(0.8, 1.6)
            theta_g = np.random.uniform(-np.pi, np.pi)
            g_pos = np.array([r_g * np.cos(theta_g), r_g * np.sin(theta_g)], dtype=np.float32)
            
            dist_to_robot = np.linalg.norm(g_pos - self.robot_pos)
            min_obs_dist = min([np.linalg.norm(g_pos - np.array([ox, oy])) - orad for ox, oy, orad in self.obstacles])
            if dist_to_robot > 1.2 and min_obs_dist > 0.25:
                self.goal_pos = g_pos
                break

        self.prev_dist_to_goal = float(np.linalg.norm(self.goal_pos - self.robot_pos))
        self.stall_steps = 0
        self.last_stall_pos = np.copy(self.robot_pos)
        return self._get_obs(), {}

    def _compute_360_lidar(self):
        """Vectorized 360-ray LiDAR raycast against circular pillars and outer wall"""
        # Global angles of all 360 rays
        global_angles = self.robot_heading + self.ray_angles
        cos_a = np.cos(global_angles)
        sin_a = np.sin(global_angles)
        ray_dirs = np.stack([cos_a, sin_a], axis=1)  # (360, 2)

        ranges = np.full(self.num_rays, self.max_lidar_range, dtype=np.float32)

        # 1. Intersect with circular obstacles
        for ox, oy, orad in self.obstacles:
            oc = np.array([ox, oy], dtype=np.float32) - self.robot_pos  # (2,)
            proj = np.sum(ray_dirs * oc, axis=1)                       # (360,)
            dist_sq = np.sum(oc**2) - proj**2                          # (360,)
            hit_mask = (proj > 0) & (dist_sq < orad**2)

            if np.any(hit_mask):
                d = proj[hit_mask] - np.sqrt(np.maximum(0.0, orad**2 - dist_sq[hit_mask]))
                valid = (d > 0.0) & (d < ranges[hit_mask])
                if np.any(valid):
                    indices = np.where(hit_mask)[0][valid]
                    ranges[indices] = np.minimum(ranges[indices], d[valid])

        # 2. Intersect with outer arena boundary (circle of arena_radius)
        # Ray from robot_pos along ray_dirs hitting circle: |robot_pos + t*ray_dir|^2 = arena_radius^2
        dot_p_d = np.sum(self.robot_pos * ray_dirs, axis=1)
        p_sq = np.sum(self.robot_pos**2)
        disc = dot_p_d**2 - (p_sq - self.arena_radius**2)
        disc = np.maximum(0.0, disc)
        t_wall = -dot_p_d + np.sqrt(disc)
        ranges = np.minimum(ranges, t_wall)

        # Return distances clipped to [0, max_lidar_range]
        return np.clip(ranges, 0.0, self.max_lidar_range)

    def _get_obs(self, lidar_ranges=None):
        if lidar_ranges is None:
            lidar_ranges = self._compute_360_lidar()
        # Normalized LiDAR: [0, 1]
        lidar_norm = (lidar_ranges / self.max_lidar_range).astype(np.float32)

        # Goal features
        rel_goal = self.goal_pos - self.robot_pos
        dist_to_goal = np.linalg.norm(rel_goal)
        target_angle = math.atan2(rel_goal[1], rel_goal[0])
        rel_angle = target_angle - self.robot_heading
        rel_angle = (rel_angle + math.pi) % (2.0 * math.pi) - math.pi

        goal_feats = np.array([
            dist_to_goal / (2.0 * self.arena_radius),
            math.cos(rel_angle),
            math.sin(rel_angle)
        ], dtype=np.float32)

        # Normalized velocities
        vel_feats = np.array([
            self.robot_v / self.max_v,
            self.robot_w / self.max_w
        ], dtype=np.float32)

        return np.concatenate([lidar_norm, goal_feats, vel_feats], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        target_v = float(np.clip(action[0], 0.0, self.max_v))
        target_w = float(np.clip(action[1], -self.max_w, self.max_w))

        # Apply realistic physical acceleration constraints
        dv = np.clip(target_v - self.robot_v, -self.max_acc_v * self.dt, self.max_acc_v * self.dt)
        dw = np.clip(target_w - self.robot_w, -self.max_acc_w * self.dt, self.max_acc_w * self.dt)

        self.robot_v += dv
        self.robot_w += dw

        # Kinematic Unicycle Integration
        self.robot_heading += self.robot_w * self.dt
        self.robot_heading = (self.robot_heading + np.pi) % (2.0 * np.pi) - np.pi
        self.robot_pos[0] += self.robot_v * math.cos(self.robot_heading) * self.dt
        self.robot_pos[1] += self.robot_v * math.sin(self.robot_heading) * self.dt

        # Compute full 360 LiDAR scan
        lidar_ranges = self._compute_360_lidar()

        # Forward danger sector clearance (+/- 45 degrees in front of robot)
        front_clearance = min(float(np.min(lidar_ranges[:45])), float(np.min(lidar_ranges[315:])))

        # Calculate minimum obstacle distance from robot center
        min_dist = float('inf')
        for ox, oy, orad in self.obstacles:
            d = np.linalg.norm(self.robot_pos - np.array([ox, oy])) - orad
            if d < min_dist:
                min_dist = d
        wall_dist = self.arena_radius - np.linalg.norm(self.robot_pos)
        min_dist = min(min_dist, wall_dist)

        # Distance to goal & Heading alignment
        dist_to_goal = float(np.linalg.norm(self.goal_pos - self.robot_pos))
        progress = (self.prev_dist_to_goal - dist_to_goal) * 15.0
        self.prev_dist_to_goal = dist_to_goal

        # Goal relative heading
        rel_goal = self.goal_pos - self.robot_pos
        target_angle = math.atan2(rel_goal[1], rel_goal[0])
        rel_angle = target_angle - self.robot_heading
        rel_angle = (rel_angle + math.pi) % (2.0 * math.pi) - math.pi

        # 1. Base progress reward
        reward = progress

        # 2. Forward momentum & heading alignment reward
        cos_align = math.cos(rel_angle)
        if cos_align > 0:
            reward += self.robot_v * cos_align * 3.0

        # 3. Small time efficiency step penalty
        reward -= 0.02

        # 4. Strict Anti-Spinning & Anti-Freezing Penalties
        if self.robot_v < 0.04:
            # Heavily punish spinning in place
            reward -= 0.10 * abs(self.robot_w)
            # If path ahead is clear and away from goal, punish freezing/idling
            if front_clearance > 0.35 and dist_to_goal > 0.35:
                reward -= 0.06

        # 5. Anti-Stall / Freezing Detector (30 steps = 3.0s threshold)
        if np.linalg.norm(self.robot_pos - self.last_stall_pos) < 0.12:
            self.stall_steps += 1
        else:
            self.stall_steps = 0
            self.last_stall_pos = np.copy(self.robot_pos)

        is_stalled = self.stall_steps >= 30

        # Goal Reached Bonus
        terminated = False
        if dist_to_goal < 0.35:
            reward += 30.0
            self.stall_steps = 0
            self.last_stall_pos = np.copy(self.robot_pos)
            # Continuous multi-goal respawn
            while True:
                r_g = np.random.uniform(0.8, 1.6)
                theta_g = np.random.uniform(-np.pi, np.pi)
                g_pos = np.array([r_g * np.cos(theta_g), r_g * np.sin(theta_g)], dtype=np.float32)
                d_rob = np.linalg.norm(g_pos - self.robot_pos)
                d_obs = min([np.linalg.norm(g_pos - np.array([ox, oy])) - orad for ox, oy, orad in self.obstacles])
                if d_rob > 1.2 and d_obs > 0.25:
                    self.goal_pos = g_pos
                    break
            self.prev_dist_to_goal = float(np.linalg.norm(self.goal_pos - self.robot_pos))

        # DIRECTIONAL SAFETY FORMULATION (Only moving into danger triggers cost!)
        cost = 0.0
        if min_dist < self.robot_radius:
            # HARD PHYSICAL COLLISION (Any direction)
            cost = 1.0
            reward -= 20.0       # Heavy collision punishment
            terminated = True   # Episode immediately ends upon collision!
        elif front_clearance < self.safety_margin and self.robot_v > 0.03:
            # Danger proximity buffer zone in direction of travel
            cost = float((self.safety_margin - front_clearance) / (self.safety_margin - self.robot_radius))

        if is_stalled:
            reward -= 5.0       # Stall / freezing punishment
            truncated = True
        else:
            truncated = self.step_count >= self.max_steps

        obs = self._get_obs(lidar_ranges=lidar_ranges)

        info = {
            "cost": cost,
            "min_obstacle_dist": min_dist,
            "front_clearance": front_clearance,
            "is_collision": min_dist < self.robot_radius
        }
        return obs, reward, terminated, truncated, info
