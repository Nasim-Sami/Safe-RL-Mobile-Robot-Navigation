# Safety General ZeroCollision: Procedural Safe RL Navigation

This sub-package contains the **generalizable zero-collision navigation suite** powered by **PPO-Lagrangian** on full 360-ray continuous LiDAR. The agent is trained across procedurally randomized domains—including variable pillar density, spatial jitter, and narrow corridors—to guarantee zero collision without reliance on rule-based heuristics or safety filter overrides.

---

## 🚀 How to Run

### 1. Requirements & Dependencies
Ensure Python 3.10+ and PyTorch are installed:
```bash
pip install -r ../../requirements.txt
```

### 2. Training the General Safe Policy
To train both baseline Unconstrained PPO and PPO-Lagrangian on CUDA device 0:
```bash
python3 train.py
```
* Training logs and reward/cost curves are saved inside `runs/`.
* Trained model weights (`.pt`) and evaluation graphs are stored in `models_saved/`.

### 3. Gazebo Sim & ROS 2 Deployment
To launch the procedural environment in **Gazebo Sim (Harmonic)** and deploy the trained PPO-Lagrangian neural policy in real time:
```bash
cd ros2
chmod +x run_simulation_and_deploy.sh
./run_simulation_and_deploy.sh
```

---

## 🧠 Logic & Algorithmic Steps

The policy directly maps raw 360-degree LiDAR range beams and proprioceptive goal vectors to unicycle steering velocities without any safety filters or intermediate waypoint planners.

```
+--------------------------------------------------------------------------+
|                            Observation Space                             |
|  [360 Continuous LiDAR Beams] + [Goal Dist, Rel Angle, v, w, Min Clear]  |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
|                  Spatial 1D-CNN Feature Extractor                        |
|  Conv1d(1->16) -> Conv1d(16->32) -> Conv1d(32->64) -> AdaptiveAvgPool   |
|  Extracts local free-space contours and corridor openings around robot  |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
|                       Multimodal Latent Fusion                           |
|  Concatenates 128-D spatial LiDAR latent features with 5-D internal state|
+--------------------------------------------------------------------------+
                                    |
          +-------------------------+-------------------------+
          |                                                   |
          v                                                   v
+---------------------------+       +------------------------------------+
|  Tanh-Bounded Actor Head  |       |            Twin Critics            |
| Outputs symmetric action  |       |  V^R: Expected cumulative reward   |
| [v, w] in range [-1, 1]   |       |  V^C: Expected cumulative cost/risk|
+---------------------------+       +------------------------------------+
          |                                                   |
          +-------------------------+-------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
|                       PPO-Lagrangian Optimization                        |
|  Dual GAE: Computes Advantage A^R and Risk Advantage A^C                 |
|  Lagrangian Advantage: A_combined = A^R - lambda * A^C                   |
|  Adaptive PID Lambda: Smoothly throttles lambda in [0, 5]               |
+--------------------------------------------------------------------------+
```

### Step 1: Analytical 360° LiDAR Raycasting
Instead of coarse ray discretization, the environment computes true obstacle clearances analytically:
1. **Ray-Pillar Chord Interception:** Projects each unit direction ray $\vec{r}$ against relative obstacle vectors $\vec{oc} = \mathbf{C} - \mathbf{P}$. The perpendicular clearance determines interception: $CP = \sqrt{\|\vec{oc}\|^2 - (\vec{r} \cdot \vec{oc})^2}$.
2. **Boundary Wall Interception:** Solves quadratic line-circle intersections against circular arena boundaries.
3. **Continuous Free Space:** Beams with no obstacles report the hardware maximum sensor range ($3.5\text{ m}$).

### Step 2: Unicycle Kinematics & Symmetric State Integration
1. **Kinematic Updates:** Computes robot trajectory via continuous unicycle differential drive equations.
2. **Motor Acceleration Clamping:** Clamps velocity step changes ($\Delta v \le 1.0\text{ m/s}^2, \Delta w \le 3.2\text{ rad/s}^2$) ensuring physically realizable commands.
3. **Symmetric Heading Alignment:** Awards reward when the robot rotates towards the target and penalizes deviation symmetrically, removing unilateral circling bias.
4. **Anti-Stall Movement Detection:** Monitors physical ground displacement $\|\mathbf{pos}_t - \mathbf{pos}_{t-1}\|_2$ (rather than wheel velocities) to terminate episodes when the robot is wedged against obstacles.

### Step 3: Spatial 1D-CNN & Multimodal Fusion
1. **Local Angular Feature Extraction:** Feeds 360 continuous beams into a 3-stage 1D convolutional network (`nn.Conv1d`) to capture contiguous spatial free-space openings.
2. **State Fusion:** Concatenates the 128-D spatial latent embedding with 5-D proprioceptive signals (goal distance, relative heading angle, current linear velocity, current angular velocity, minimum obstacle clearance).

### Step 4: Constrained Optimization via Adaptive PID PPO-Lagrangian
1. **Twin Critic Decomposition:** Independent critic heads estimate expected future return ($V^R$) and cumulative safety constraint violation cost ($V^C$).
2. **Dual GAE Formulation:** Calculates reward advantage $A^R$ and cost advantage $A^C$.
3. **Lagrangian Advantage Fusion:** Fuses advantages via $A_{\text{combined}} = A^R - \lambda \cdot A^C$. If a path poses collision risk ($A^C > 0$), the safety multiplier $\lambda$ suppresses the action score.
4. **Adaptive PID Multiplier Regulation:** Dynamically tunes the Lagrange multiplier $\lambda \in [0.0, 5.0]$ using a PID controller with derivative damping, eliminating multiplier explosion in tight corridors and preventing safe robot freezing.
