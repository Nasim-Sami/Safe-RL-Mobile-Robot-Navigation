import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models.networks import SafeActorCritic360
from utils.buffer import SafeRolloutBuffer360

class PPOLagrangian360:
    """
    Constrained PPO-Lagrangian for 360-ray Safe Navigation with Zero Collision constraint.
    Dual Critics, Dual GAE, and Adaptive Lagrange Multiplier update.
    """
    def __init__(
        self,
        envs,
        device,
        total_timesteps=400000,
        learning_rate=3e-4,
        num_steps=1024,
        num_minibatches=4,
        update_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        clip_vloss=True,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        cost_limit=0.0,         # Strict ZERO COLLISION target budget!
        init_lambda=0.05,
        lambda_lr=0.02,
        cost_gamma=0.99,
        cost_gae_lambda=0.95,
        save_dir="./runs"
    ):
        self.envs = envs
        self.device = device
        self.total_timesteps = total_timesteps
        self.lr = learning_rate
        self.num_steps = num_steps
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.cost_limit = cost_limit
        self.lambda_param = init_lambda
        self.lambda_lr = lambda_lr
        self.cost_gamma = cost_gamma
        self.cost_gae_lambda = cost_gae_lambda
        self.save_dir = save_dir

        self.num_envs = envs.num_envs
        self.batch_size = self.num_envs * self.num_steps
        self.minibatch_size = self.batch_size // self.num_minibatches

        # Dimensions
        self.obs_dim = envs.single_observation_space.shape[0]
        self.act_dim = envs.single_action_space.shape[0]

        # Model and Optimizer
        self.agent = SafeActorCritic360(self.obs_dim, self.act_dim).to(device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.lr, eps=1e-5)

        # Rollout Buffer
        self.buffer = SafeRolloutBuffer360(
            self.num_steps, self.num_envs, self.obs_dim, self.act_dim, device
        )

        # PID parameters for smooth Lagrange multiplier control
        self.pid_i = 0.0
        self.pid_prev_error = 0.0

    def update_lagrange_multiplier(self, mean_episode_cost):
        """Adaptive PID-based update for Lagrange Multiplier lambda"""
        error = mean_episode_cost - self.cost_limit
        self.pid_i = max(0.0, self.pid_i + error)
        d_error = error - self.pid_prev_error
        self.pid_prev_error = error

        # PID Update
        kp, ki, kd = 0.15, 0.02, 0.05
        step = kp * error + ki * self.pid_i + kd * d_error
        self.lambda_param = float(np.clip(self.lambda_param + self.lambda_lr * step, 0.0, 50.0))
        return self.lambda_param

    def train(self, logger=None):
        global_step = 0
        start_time = time.time()

        obs, _ = self.envs.reset()
        next_obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        next_done = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        num_updates = self.total_timesteps // self.batch_size
        print(f"🚀 [PPOLagrangian-360] Starting Training: {self.total_timesteps:,} steps | Batch: {self.batch_size} | Envs: {self.num_envs}")
        print(f"🎯 Zero Collision Constraint Active (Cost Limit = {self.cost_limit})")

        ep_returns = []
        ep_costs = []
        recent_returns = []
        recent_costs = []

        cur_ep_returns = np.zeros(self.num_envs)
        cur_ep_costs = np.zeros(self.num_envs)

        for update in range(1, num_updates + 1):
            # 1. Rollout Phase
            for step in range(self.num_steps):
                global_step += self.num_envs

                with torch.no_grad():
                    action, logprob, _, r_val, c_val = self.agent.get_action_and_values(next_obs)

                # Step environments
                cpu_action = action.cpu().numpy()
                real_action = np.clip(cpu_action, self.envs.single_action_space.low, self.envs.single_action_space.high)
                obs, reward, terminated, truncated, infos = self.envs.step(real_action)

                done = np.logical_or(terminated, truncated)
                cur_ep_returns += reward
                if isinstance(infos, dict) and "cost" in infos:
                    step_costs = np.array(infos["cost"], dtype=np.float32)
                else:
                    step_costs = np.zeros(self.num_envs, dtype=np.float32)

                cur_ep_costs += step_costs

                for i in range(self.num_envs):
                    if done[i]:
                        ep_returns.append(cur_ep_returns[i])
                        ep_costs.append(cur_ep_costs[i])
                        recent_returns.append(cur_ep_returns[i])
                        recent_costs.append(cur_ep_costs[i])
                        cur_ep_returns[i] = 0.0
                        cur_ep_costs[i] = 0.0

                # Buffer storage
                t_obs = next_obs
                t_act = action
                t_logprob = logprob
                t_rew = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
                t_cost = torch.as_tensor(step_costs, dtype=torch.float32, device=self.device)
                t_done = next_done

                self.buffer.add(t_obs, t_act, t_logprob, t_rew, t_cost, t_done, r_val, c_val)

                next_obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                next_done = torch.as_tensor(done, dtype=torch.float32, device=self.device)

            # 2. GAE Computation
            with torch.no_grad():
                next_r_val = self.agent.get_reward_value(next_obs)
                next_c_val = self.agent.get_cost_value(next_obs)

            self.buffer.compute_gae(
                next_obs, next_done, next_r_val, next_c_val,
                self.gamma, self.gae_lambda, self.cost_gamma, self.cost_gae_lambda
            )

            # 3. Update Adaptive Lambda
            mean_cost = float(np.mean(recent_costs[-50:])) if len(recent_costs) > 0 else 0.0
            cur_lambda = self.update_lagrange_multiplier(mean_cost)

            # 4. PPO Mini-Batch Optimization
            clipfracs = []
            pg_losses = []
            v_losses = []
            cost_v_losses = []
            ent_losses = []

            for epoch in range(self.update_epochs):
                for b_obs, b_act, b_logp, b_r_adv, b_r_ret, b_r_val, b_c_adv, b_c_ret, b_c_val in self.buffer.get_batches(self.minibatch_size):
                    _, new_logp, entropy, new_r_val, new_c_val = self.agent.get_action_and_values(b_obs, b_act)
                    logratio = new_logp - b_logp
                    ratio = torch.exp(logratio)

                    with torch.no_grad():
                        clipfracs.append(((ratio - 1.0).abs() > self.clip_coef).float().mean().item())

                    # Dual GAE Advantage Fusion with Lagrange Multiplier
                    # Normalize advantages
                    norm_r_adv = (b_r_adv - b_r_adv.mean()) / (b_r_adv.std() + 1e-8)
                    norm_c_adv = (b_c_adv - b_c_adv.mean()) / (b_c_adv.std() + 1e-8)

                    # Constrained Surrogate Advantage: A_safe = (A^R - lambda * A^C) / (1 + lambda)
                    safe_adv = (norm_r_adv - cur_lambda * norm_c_adv) / (1.0 + cur_lambda)

                    # Policy Loss
                    pg_loss1 = -safe_adv * ratio
                    pg_loss2 = -safe_adv * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Reward Value Loss
                    if self.clip_vloss:
                        v_loss_unclipped = (new_r_val - b_r_ret) ** 2
                        v_clipped = b_r_val + torch.clamp(new_r_val - b_r_val, -self.clip_coef, self.clip_coef)
                        v_loss_clipped = (v_clipped - b_r_ret) ** 2
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = 0.5 * ((new_r_val - b_r_ret) ** 2).mean()

                    # Cost Value Loss
                    if self.clip_vloss:
                        cv_loss_unclipped = (new_c_val - b_c_ret) ** 2
                        cv_clipped = b_c_val + torch.clamp(new_c_val - b_c_val, -self.clip_coef, self.clip_coef)
                        cv_loss_clipped = (cv_clipped - b_c_ret) ** 2
                        cv_loss = 0.5 * torch.max(cv_loss_unclipped, cv_loss_clipped).mean()
                    else:
                        cv_loss = 0.5 * ((new_c_val - b_c_ret) ** 2).mean()

                    entropy_loss = entropy.mean()

                    # Joint Objective
                    total_loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * (v_loss + cv_loss)

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    pg_losses.append(pg_loss.item())
                    v_losses.append(v_loss.item())
                    cost_v_losses.append(cv_loss.item())
                    ent_losses.append(entropy_loss.item())

            # Progress Logging
            sps = int(global_step / (time.time() - start_time))
            avg_ret = float(np.mean(recent_returns[-30:])) if len(recent_returns) > 0 else 0.0
            avg_cst = float(np.mean(recent_costs[-30:])) if len(recent_costs) > 0 else 0.0

            if update % 5 == 0 or update == num_updates:
                print(
                    f"[{update:03d}/{num_updates:03d}] Steps: {global_step:7d} | "
                    f"Return: {avg_ret:6.1f} | Cost: {avg_cst:5.2f} (Limit: {self.cost_limit:.1f}) | "
                    f"λ: {cur_lambda:6.3f} | V_Loss: {np.mean(v_losses):5.2f} | SPS: {sps}"
                )

            # Periodic Checkpoint
            if global_step % 50000 < self.batch_size:
                ckpt_path = os.path.join(self.save_dir, f"ppo_lag_360_step_{global_step}.pt")
                torch.save(self.agent.state_dict(), ckpt_path)

        # Save Final Model
        final_path = os.path.join(self.save_dir, "ppo_lagrangian_360_final.pt")
        torch.save(self.agent.state_dict(), final_path)
        print(f"✅ Saved trained policy to: {final_path}")
        return self.agent
