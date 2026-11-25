import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models.networks import SafeActorCriticGeneral
from utils.buffer import SafeRolloutBufferGeneral

class PPOLagrangianGeneral:
    """
    Constrained PPO-Lagrangian for Generalizable 360-Ray Mobile Robot Safe Navigation.
    Dual Critics, Dual GAE, and Adaptive PID-Lagrange multiplier for Strict Zero Collision.
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
        cost_limit=0.0,
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

        self.obs_dim = envs.single_observation_space.shape[0]
        self.act_dim = envs.single_action_space.shape[0]

        self.agent = SafeActorCriticGeneral(self.obs_dim, self.act_dim).to(device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.lr, eps=1e-5)

        self.buffer = SafeRolloutBufferGeneral(
            self.num_steps, self.num_envs, self.obs_dim, self.act_dim, device
        )

        self.pid_i = 0.0
        self.pid_prev_error = 0.0

    def update_lagrange_multiplier(self, mean_episode_cost):
        error = mean_episode_cost - self.cost_limit
        self.pid_i = max(0.0, self.pid_i + error)
        d_error = error - self.pid_prev_error
        self.pid_prev_error = error

        # PID Update with smooth damping
        kp, ki, kd = 0.08, 0.01, 0.02
        step = kp * error + ki * self.pid_i + kd * d_error
        self.lambda_param = float(np.clip(self.lambda_param + self.lambda_lr * step, 0.0, 5.0))
        return self.lambda_param

    def train(self):
        global_step = 0
        start_time = time.time()

        obs, _ = self.envs.reset()
        next_obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        next_done = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        num_updates = self.total_timesteps // self.batch_size
        print(f"🚀 [PPOLagrangian-General] Steps: {self.total_timesteps:,} | Batch: {self.batch_size} | Envs: {self.num_envs}")
        print(f"🛡️ Target Safety Budget d_k: {self.cost_limit} (Strict Zero Collision)")

        ep_returns, ep_costs = [], []
        recent_returns, recent_costs = [], []
        cur_ep_returns = np.zeros(self.num_envs)
        cur_ep_costs = np.zeros(self.num_envs)

        for update in range(1, num_updates + 1):
            self.buffer.reset()
            # 1. Rollout Phase
            for step in range(self.num_steps):
                global_step += self.num_envs

                with torch.no_grad():
                    action, logprob, _, r_val, c_val = self.agent.get_action_and_values(next_obs)

                cpu_action = action.cpu().numpy()
                real_action = np.clip(cpu_action, -1.0, 1.0)
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

                self.buffer.add(next_obs, action, logprob,
                                torch.as_tensor(reward, dtype=torch.float32, device=self.device),
                                torch.as_tensor(step_costs, dtype=torch.float32, device=self.device),
                                next_done, r_val, c_val)

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

            # 3. Update Lagrange Multiplier lambda
            mean_ep_cost = float(np.mean(recent_costs[-40:])) if len(recent_costs) > 0 else 0.0
            lam = self.update_lagrange_multiplier(mean_ep_cost)

            # Flatten buffer
            b_obs = self.buffer.obs.reshape((-1, self.obs_dim))
            b_act = self.buffer.actions.reshape((-1, self.act_dim))
            b_logprob = self.buffer.logprobs.reshape(-1)
            b_r_adv = self.buffer.reward_advantages.reshape(-1)
            b_c_adv = self.buffer.cost_advantages.reshape(-1)
            b_r_ret = self.buffer.reward_returns.reshape(-1)
            b_c_ret = self.buffer.cost_returns.reshape(-1)
            b_r_val = self.buffer.reward_values.reshape(-1)
            b_c_val = self.buffer.cost_values.reshape(-1)

            # Lagrangian Advantage
            b_combined_adv = b_r_adv - lam * b_c_adv
            b_combined_adv = (b_combined_adv - b_combined_adv.mean()) / (b_combined_adv.std() + 1e-8)

            # 4. Optimization Epochs
            b_inds = np.arange(self.batch_size)
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, new_r_val, new_c_val = self.agent.get_action_and_values(
                        b_obs[mb_inds], b_act[mb_inds]
                    )

                    logratio = newlogprob - b_logprob[mb_inds]
                    ratio = torch.exp(logratio)

                    # PPO Policy Loss
                    mb_adv = b_combined_adv[mb_inds]
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Reward Value Loss
                    if self.clip_vloss:
                        r_v_loss_unclipped = (new_r_val - b_r_ret[mb_inds]) ** 2
                        r_v_clipped = b_r_val[mb_inds] + torch.clamp(
                            new_r_val - b_r_val[mb_inds], -self.clip_coef, self.clip_coef
                        )
                        r_v_loss_clipped = (r_v_clipped - b_r_ret[mb_inds]) ** 2
                        r_v_loss = 0.5 * torch.max(r_v_loss_unclipped, r_v_loss_clipped).mean()
                    else:
                        r_v_loss = 0.5 * ((new_r_val - b_r_ret[mb_inds]) ** 2).mean()

                    # Cost Value Loss
                    c_v_loss = 0.5 * ((new_c_val - b_c_ret[mb_inds]) ** 2).mean()

                    # Total Loss
                    loss = pg_loss - self.ent_coef * entropy.mean() + self.vf_coef * (r_v_loss + c_v_loss)

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

            # Logging
            if update % 5 == 0 or update == num_updates:
                fps = int(global_step / (time.time() - start_time))
                mean_r = float(np.mean(recent_returns[-40:])) if len(recent_returns) > 0 else 0.0
                mean_c = float(np.mean(recent_costs[-40:])) if len(recent_costs) > 0 else 0.0
                print(f"[{update:3d}/{num_updates}] Step: {global_step:,} | Ret: {mean_r:6.1f} | Cost: {mean_c:4.2f} | λ: {lam:4.2f} | FPS: {fps}")

        final_path = os.path.join(self.save_dir, "ppo_lagrangian_general_final.pt")
        torch.save(self.agent.state_dict(), final_path)
        print(f"💾 Saved general Safe policy to: {final_path}")
        return self.agent
