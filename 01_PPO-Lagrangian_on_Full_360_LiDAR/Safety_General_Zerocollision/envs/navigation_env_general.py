import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class SafeNavGeneralEnv(gym.Env):
    """
    Generalizable 360-Ray Zero-Collision Safe Mobile Robot Navigation Environment.
    
    Key Features:
    1. Procedural Domain Randomization: Dynamic obstacle clutter (count, size, position).
    2. Zero Map Overfitting: Randomizes between structured grid corridors and cluttered mazes.
    3. Mathematical Steering Symmetry: Explicit bidirectional heading alignment reward.
    4. Anti-Spin & Anti-Freeze Constraints: Heavily penalizes spinning in place or idling.
    5. Realistic Unicycle Dynamics: Acceleration and velocity bounds matching TurtleBot3.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, randomize_obstacles=True, max_steps=500):
        super().__init__()
        self.randomize_obstacles = randomize_obstacles
        self.max_steps = max_steps
        self.num_rays = 360
        self.max_lidar_range = 3.5  # meters
        self.dt = 0.1               # 10 Hz control loop

        # Acceleration and velocity limits (matching physical TurtleBot3 waffle)
        self.max_v = 0.26          # m/s
        self.max_w = 1.5           # rad/s
        self.max_acc_v = 1.0       # m/s^2 -> max dv = 0.10 m/s per step
        self.max_acc_w = 3.2       # rad/s^2 -> max dw = 0.32 rad/s per step

        # Action Space: Normalized [-1, 1] for both dimensions
        # Action[0] -> forward speed: mapped to [0, max_v]
        # Action[1] -> angular rate: mapped to [-max_w, max_w] (perfect symmetry!)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation Space:
        # - 360 normalized LiDAR rays [0, 1] (0 = collision, 1 = max clear)
        # - Goal features: [norm_distance, cos(rel_angle), sin(rel_angle)] (3)
        # - Velocity features: [v / max_v, w / max_w] (2)
        # Total observation dimension = 360 + 3 + 2 = 365
        self.obs_dim = self.num_rays + 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # Precompute relative ray angles: [0, 1, 2, ..., 359] degrees in radians
        self.ray_angles = np.linspace(0.0, 2.0 * np.pi, self.num_rays, endpoint=False, dtype=np.float32)

        # State Variables
        self.robot_pos = np.zeros(2, dtype=np.float32)
        self.robot_heading = 0.0
        self.robot_v = 0.0
        self.robot_w = 0.0
        self.goal_pos = np.zeros(2, dtype=np.float32)
        self.prev_dist_to_goal = 0.0
        self.prev_rel_angle = 0.0
        self.step_count = 0
        self.stall_steps = 0
        self.last_stall_pos = np.zeros(2, dtype=np.float32)

        # World geometry
        self.arena_radius = 2.4   # Arena outer boundary (matching 2.4m hex/circle)
        self.obstacles = []       # List of (x, y, radius)

    def _setup_obstacles(self):
        """Domain Randomization for robust generalizability across any room or obstacle count"""
        self.obstacles = []
        mode = np.random.rand()

        if mode < 0.40:
            # Mode A: Gazebo 3x3 Corridor Layout with random jitter (prevents memorization)
            base_coords = [-1.1, 0.0, 1.1]
            for cx in base_coords:
                for cy in base_coords:
                    jitter = np.random.uniform(-0.15, 0.15, size=2)
                    rad = np.random.uniform(0.12, 0.18)
                    self.obstacles.append((cx + jitter[0], cy + jitter[1], rad))
        elif mode < 0.80:
            # Mode B: Random Procedural Clutter (4 to 10 obstacles scattered across the arena)
            num_obs = np.random.randint(5, 11)
            for _ in range(num_obs):
                r = np.random.uniform(0.4, self.arena_radius - 0.4)
                th = np.random.uniform(-np.pi, np.pi)
                rad = np.random.uniform(0.12, 0.28)
                self.obstacles.append((r * np.cos(th), r * np.sin(th), rad))
        else:
            # Mode C: Narrow Corridor / Slalom configuration
            xs = [-0.9, 0.0, 0.9]
            for x in xs:
                y = np.random.uniform(-0.8, 0.8)
                rad = np.random.uniform(0.18, 0.30)
                self.obstacles.append((x, y, rad))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.robot_v = 0.0
        self.robot_w = 0.0

        self._setup_obstacles()

        # Spawn robot in free space
        for _ in range(100):
            r = np.random.uniform(0.3, self.arena_radius - 0.5)
            theta = np.random.uniform(-np.pi, np.pi)
            pos = np.array([r * np.cos(theta), r * np.sin(theta)], dtype=np.float32)
            
            clearance = min([np.linalg.norm(pos - np.array([ox, oy])) - orad for ox, oy, orad in self.obstacles])
            if clearance > 0.35:
                self.robot_pos = pos
                break

        self.robot_heading = float(np.random.uniform(-np.pi, np.pi))

        # Spawn goal at sufficient distance from robot and away from obstacles
        for _ in range(100):
            r_g = np.random.uniform(0.4, self.arena_radius - 0.4)
            theta_g = np.random.uniform(-np.pi, np.pi)
            g_pos = np.array([r_g * np.cos(theta_g), r_g * np.sin(theta_g)], dtype=np.float32)
            
            dist = np.linalg.norm(g_pos - self.robot_pos)
            clearance = min([np.linalg.norm(g_pos - np.array([ox, oy])) - orad for ox, oy, orad in self.obstacles])
            if dist > 1.2 and clearance > 0.25:
                self.goal_pos = g_pos
                break

        self.prev_dist_to_goal = float(np.linalg.norm(self.goal_pos - self.robot_pos))
        rel_goal = self.goal_pos - self.robot_pos
        t_angle = math.atan2(rel_goal[1], rel_goal[0])
        self.prev_rel_angle = (t_angle - self.robot_heading + math.pi) % (2.0 * math.pi) - math.pi

        self.stall_steps = 0
        self.last_stall_pos = np.copy(self.robot_pos)
        return self._get_obs(), {}

    def _compute_360_lidar(self):
        """Vectorized 360-ray LiDAR raycast against cylindrical obstacles and outer wall"""
        global_angles = self.robot_heading + self.ray_angles
        cos_a = np.cos(global_angles)
        sin_a = np.sin(global_angles)
        ray_dirs = np.stack([cos_a, sin_a], axis=1)  # (360, 2)

        ranges = np.full(self.num_rays, self.max_lidar_range, dtype=np.float32)

        # 1. Circular obstacle intersections
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

        # 2. Outer arena boundary (cylinder/circle of arena_radius)
        dot_p_d = np.sum(self.robot_pos * ray_dirs, axis=1)
        p_sq = np.sum(self.robot_pos**2)
        disc = np.maximum(0.0, dot_p_d**2 - (p_sq - self.arena_radius**2))
        t_wall = -dot_p_d + np.sqrt(disc)
        ranges = np.minimum(ranges, t_wall)

        return np.clip(ranges, 0.0, self.max_lidar_range)

    def _get_obs(self, lidar_ranges=None):
        if lidar_ranges is None:
            lidar_ranges = self._compute_360_lidar()

        lidar_norm = (lidar_ranges / self.max_lidar_range).astype(np.float32)

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

        vel_feats = np.array([
            self.robot_v / self.max_v,
            self.robot_w / self.max_w
        ], dtype=np.float32)

        return np.concatenate([lidar_norm, goal_feats, vel_feats], dtype=np.float32)

    def step(self, action):
        self.step_count += 1

        # Scale action from [-1, 1] to physical unicycle bounds
        norm_v = float(np.clip(action[0], -1.0, 1.0))
        norm_w = float(np.clip(action[1], -1.0, 1.0))

        # Map norm_v [-1, 1] -> [0.0, max_v] (or crawl floor if far from goal)
        target_v = float((norm_v + 1.0) / 2.0 * self.max_v)
        target_w = float(norm_w * self.max_w)

        # Apply motor acceleration limits
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
        min_clearance = float(np.min(lidar_ranges))
        front_clearance = min(float(np.min(lidar_ranges[:30])), float(np.min(lidar_ranges[330:])))

        # Distance to goal & Heading error
        dist_to_goal = float(np.linalg.norm(self.goal_pos - self.robot_pos))
        rel_goal = self.goal_pos - self.robot_pos
        target_angle = math.atan2(rel_goal[1], rel_goal[0])
        rel_angle = target_angle - self.robot_heading
        rel_angle = (rel_angle + math.pi) % (2.0 * math.pi) - math.pi

        # -------------------------------------------------------------
        # 1. REWARD FORMULATION (Symmetric, Anti-Spin, Anti-Meander)
        # -------------------------------------------------------------
        # Base distance progress reward
        progress = (self.prev_dist_to_goal - dist_to_goal) * 20.0
        self.prev_dist_to_goal = dist_to_goal

        # Symmetric Heading Alignment Progress Reward:
        # Heavily rewards turning TOWARDS the goal, penalizes turning away
        heading_progress = (abs(self.prev_rel_angle) - abs(rel_angle)) * 6.0
        self.prev_rel_angle = rel_angle

        # Forward velocity alignment reward (only when heading roughly matches goal)
        cos_align = math.cos(rel_angle)
        align_bonus = 0.0
        if cos_align > 0:
            align_bonus = self.robot_v * cos_align * 2.5

        # Small step efficiency penalty
        step_penalty = -0.02

        # Action smoothness penalty (eliminates high-frequency jerky wandering)
        smooth_penalty = -0.05 * (dv / (self.max_acc_v * self.dt))**2 - 0.05 * (dw / (self.max_acc_w * self.dt))**2

        reward = progress + heading_progress + align_bonus + step_penalty + smooth_penalty

        # Strict Anti-Spin Penalty: If robot is spinning with almost zero forward speed
        if self.robot_v < 0.05:
            reward -= 0.25 * abs(self.robot_w)

        # -------------------------------------------------------------
        # 2. CONSTRAINED SAFETY COST (Zero Collision Constraint)
        # -------------------------------------------------------------
        # Safety danger threshold: 0.20m (Physical robot radius is 0.15m)
        safety_buffer = 0.20
        physical_collision_dist = 0.15

        # Cost: 1.0 only when dangerously breaching the 0.20m threshold
        cost = 1.0 if min_clearance < safety_buffer else 0.0

        # Proactive forward danger penalty
        if front_clearance < 0.38 and self.robot_v > 0.10:
            reward -= 0.15

        # -------------------------------------------------------------
        # 3. TERMINATION & TRUNCATION CONDITIONS
        # -------------------------------------------------------------
        terminated = False
        truncated = False
        is_collision = False
        is_goal = False

        # Physical Collision Check (Terminated with severe safety penalty)
        if min_clearance < physical_collision_dist:
            terminated = True
            is_collision = True
            cost += 10.0
            reward -= 25.0

        # Goal Reached Condition (Within 0.30m of target)
        elif dist_to_goal < 0.30:
            terminated = True
            is_goal = True
            reward += 35.0

        # Anti-Stall / Freezing Detector (Terminated after 25 stuck steps)
        if np.linalg.norm(self.robot_pos - self.last_stall_pos) < 0.10:
            self.stall_steps += 1
            if self.stall_steps > 25:
                truncated = True
                reward -= 5.0
        else:
            self.stall_steps = 0
            self.last_stall_pos = np.copy(self.robot_pos)

        # Max Episode Length Truncation
        if self.step_count >= self.max_steps:
            truncated = True

        info = {
            "cost": float(cost),
            "is_collision": is_collision,
            "is_goal": is_goal,
            "dist_to_goal": dist_to_goal,
            "min_clearance": min_clearance
        }

        return self._get_obs(lidar_ranges), reward, terminated, truncated, info
