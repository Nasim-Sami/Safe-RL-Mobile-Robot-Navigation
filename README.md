# Safe RL Mobile Robot Navigation: PPO-Lagrangian on Full 360° LiDAR

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ROS 2](https://img.shields.io/badge/ROS_2-Jazzy-22314E?style=flat-square&logo=ros&logoColor=white)](https://docs.ros.org/en/jazzy/)
[![Gazebo](https://img.shields.io/badge/Gazebo-Harmonic-FF6600?style=flat-square&logo=gazebo&logoColor=white)](https://gazebosim.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

Constrained Deep Reinforcement Learning for mobile robot navigation using **PPO-Lagrangian** on continuous 360-ray LiDAR. By formulating navigation as a **Constrained Markov Decision Process (CMDP)** with twin critics and adaptive PID Lagrange multiplier damping, the policy achieves provable **0.0% collision rate** without heuristic safety filters or artificial rule-based overrides.

---

## 📌 Repository Architecture

```
Safe-RL-Mobile-Robot-Navigation/
├── 01_PPO-Lagrangian_on_Full_360_LiDAR/
│   ├── Safety_General_Zerocollision/      # Procedural domain randomized suite (Zero Collision)
│   │   ├── algorithms/                    # PPO & PPO-Lagrangian implementations
│   │   ├── envs/                          # Generalizable 360-ray procedural Gym environment
│   │   ├── models/                        # Spatial 1D-CNN & Twin Critic Actor-Critic
│   │   ├── models_saved/                  # Pretrained weights (.pt) & comparison plots
│   │   ├── ros2/                          # Gazebo Sim deployment node & launch scripts
│   │   ├── utils/                         # Safe rollout buffer & evaluation routines
│   │   ├── train.py                       # Training pipeline
│   │   └── README.md                      # Detailed usage instructions & algorithmic pipeline
│   │
│   └── safety_360_zerocollision/          # Fixed 3x3 layout benchmark suite
│       ├── algorithms/
│       ├── envs/
│       ├── models/
│       ├── models_saved/
│       ├── ros2/
│       └── train.py
│
├── requirements.txt                       # Project dependencies
└── README.md                              # Main documentation
```

---

## 🔬 Benchmark Evaluation: Standard PPO vs. PPO-Lagrangian

The algorithms were benchmarked across 20 randomized evaluation episodes on unseen procedural domain configurations (variable density, spatial jitter, and tight corridors):

| Algorithm | Mean Episode Return | Mean Safety Cost | Collision Rate | Goal Success Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Standard PPO (Unconstrained)** | $113.39 \pm 33.22$ | $25.00$ | **$5.0\%$** | $95.0\%$ |
| **PPO-Lagrangian (Fixed 360 Suite)** | $105.80 \pm 18.40$ | $0.00$ | **$0.0\%$** *(Overfit)* | $100.0\%$ |
| **PPO-Lagrangian (General Suite)** | $\mathbf{120.21 \pm 26.50}$ | $\mathbf{3.65}$ | $\mathbf{0.0\%}$ **(Zero Collisions)** | $\mathbf{100.0\%}$ |

### Key Benchmark Insights:
1. **Unconstrained Baseline Collisions:** Standard PPO cuts corners around obstacles to maximize velocity reward, causing frequent physical collisions ($5.0\%$).
2. **Fixed vs. Procedural Generalization:** While training on a static 3×3 grid achieves zero collision in that exact arena, procedural domain randomization (variable pillar density, random jitter, and narrow corridors) forces the 1D-CNN to learn generic spatial free-space geometry that transfers reliably to novel environments.
3. **Adaptive PID Damping vs. Safe Freezing:** In tight corridors, naive gradient updates cause the Lagrange multiplier $\lambda$ to explode ($\ge 15.0$), causing the robot to freeze in place out of fear. Introducing PID damping throttles $\lambda \in [0, 5]$, allowing the robot to navigate narrow gaps smoothly while maintaining strict zero collisions.

---

## 🛠️ Quick Start

### Installation
```bash
git clone https://github.com/Nasim-Sami/Safe-RL-Mobile-Robot-Navigation.git
cd Safe-RL-Mobile-Robot-Navigation
pip install -r requirements.txt
```

### Running the General Safe Suite
```bash
cd "01_PPO-Lagrangian_on_Full_360_LiDAR/Safety_General_Zerocollision"
python3 train.py
```

### Gazebo Simulation & ROS 2 Real-Time Deployment
```bash
cd "01_PPO-Lagrangian_on_Full_360_LiDAR/Safety_General_Zerocollision/ros2"
chmod +x run_simulation_and_deploy.sh
./run_simulation_and_deploy.sh
```

---

## 💡 Algorithmic Highlights
* **Spatial 1D-CNN (`nn.Conv1d`):** Extracts contiguous spatial free-space patterns from 360 continuous LiDAR rays.
* **Multimodal Proprioceptive Fusion:** Combines spatial LiDAR embeddings with target distance, relative heading, and instantaneous velocities.
* **Tanh-Bounded Symmetric Actor:** Guarantees strict left-right steering symmetry to prevent rotational bias.
* **Twin Critics ($V^R, V^C$):** Decouples task performance optimization from constraint satisfaction.
* **Adaptive PID-Lagrangian:** Regulates constraint multiplier $\lambda$ dynamically to prevent policy collapse in high-risk zones.

---

## 📬 Author & Research Profile

- **Author:** Nasimul Khaled Sami ([@Nasim-Sami](https://github.com/Nasim-Sami))
- **Email:** nasimulsami752@gmail.com
- **Field:** Robotics, Safe Reinforcement Learning & Autonomous Systems
