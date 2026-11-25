# SafeRL-360: Pure End-to-End Constrained Safe RL for Mobile Robot Navigation

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15%20CUDA-ee4c2c.svg)](https://pytorch.org/)
[![ROS 2](https://img.shields.io/badge/ROS%202-Lyrical-blue.svg)](https://docs.ros.org/)
[![Gazebo](https://img.shields.io/badge/Gazebo-Harmonic%20Sim-orange.svg)](https://gazebosim.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An industrial-grade, research-ready implementation of **Continuous PPO vs. Constrained PPO-Lagrangian** for safe autonomous mobile robot navigation.

---

## 🌟 Key Highlights & Engineering Features

- **100% Pure Neural End-to-End Decision Making:** No hand-crafted `if-else` safety shields or rule-based braking hacks. The policy directly outputs optimal linear velocity $v$ and angular velocity $\omega$ from sensory inputs.
- **Full 360-Ray Continuous LiDAR Input:** 360 continuous 1D range measurements ($1^\circ$ resolution) fed into a **Spatial 1D-CNN + Proprioceptive Feature Extractor**.
- **Strict Zero-Collision Constraint ($d_k = 0.0$):**
  - **Danger Buffer Zone:** Smooth proximity cost gradients when distance drops below $0.30$m.
  - **Physical Collision ($< 0.18$m):** Hard episode termination with heavy collision penalty.
- **Twin Critics & Dual GAE:**
  - Reward Value Critic: $V^R(s) \implies \hat{A}^R(s, a)$
  - Safety Cost Critic: $V^C(s) \implies \hat{A}^C(s, a)$
  - Adaptive Lagrange Multiplier $\lambda$: Dynamically penalizes trajectories violating the zero-cost budget.
- **Sim-to-Real Kinematic Consistency:** Includes motor acceleration limits ($\Delta v \le 1.0\text{ m/s}^2, \Delta \omega \le 3.2\text{ rad/s}^2$) preventing inertial overshoot in 3D physics simulators (Gazebo).

---

## 📁 Repository Structure

```
SafeRL_360_ZeroCollision/
├── algorithms/
│   ├── ppo.py                  # Standard Unconstrained PPO (Baseline)
│   └── ppo_lagrangian.py       # Constrained PPO-Lagrangian (Twin Critics + Adaptive λ)
├── models/
│   └── networks.py             # Spatial 1D-CNN + Proprioceptive Feature Extractor
├── envs/
│   ├── navigation_env_360.py   # High-Fidelity 360-Ray LiDAR Safe Navigation Env
│   └── make_env.py             # Vectorized Multi-Env Factory
├── utils/
│   ├── buffer.py               # Safe Rollout Buffer supporting Dual GAE
│   └── eval.py                 # Deterministic 20-episode benchmark evaluator
├── ros2/
│   ├── deploy_node_360.py      # ROS 2 Pure Neural Deployment Node (Zero Heuristics)
│   ├── launch_simulation.sh    # 1-Click Gazebo + TF + Costmap + RViz2
│   └── run_agent.sh            # 1-Click Policy Runner (ppo_lagrangian / ppo)
├── models_saved/               # Exported final PyTorch policy weights
├── train.py                    # Unified Training & Benchmarking CLI
└── README.md
```

---

## 🚀 Quickstart & Training

### 1. Train Both Policies (Standard PPO vs. PPO-Lagrangian)
```bash
cd "/home/sami/Documents/PPO & PPO-Lag/SafeRL_360_ZeroCollision"
export CUDA_VISIBLE_DEVICES=0
/home/sami/mp_env/bin/python3 train.py --algo both --total_timesteps 400000 --num_envs 8 --cost_limit 0.0
```

### 2. Train Only PPO-Lagrangian (Zero Collision Focus)
```bash
/home/sami/mp_env/bin/python3 train.py --algo ppo_lagrangian --total_timesteps 400000 --num_envs 8 --cost_limit 0.0
```

---

## 🤖 3D Simulation Deployment (ROS 2 + Gazebo Sim + RViz2)

### Terminal 1: Launch Gazebo Sim & RViz2
```bash
cd "/home/sami/Documents/PPO & PPO-Lag/SafeRL_360_ZeroCollision/ros2"
./launch_simulation.sh
```

### Terminal 2: Run Pure Neural Policy
```bash
cd "/home/sami/Documents/PPO & PPO-Lag/SafeRL_360_ZeroCollision/ros2"
./run_agent.sh ppo_lagrangian
```
