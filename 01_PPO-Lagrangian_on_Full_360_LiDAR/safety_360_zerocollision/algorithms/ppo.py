import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models.networks import ActorCritic360
from utils.buffer import SafeRolloutBuffer360

class PPO360:
    """
    Standard Unconstrained Continuous PPO for 360-ray Navigation.
    Serves as the baseline comparison showing reward greediness vs collisions.
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
        self.save_dir = save_dir

        self.num_envs = envs.num_envs
        self.batch_size = self.num_envs * self.num_steps
        self.minibatch_size = self.batch_size // self.num_minibatches

        self.obs_dim = envs.single_observation_space.shape[0]
        self.act_dim = envs.single_action_space.shape[0]

        self.agent = ActorCritic360(self.obs_dim, self.act_dim).to(device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.lr, eps=1e-5)

        self.buffer = SafeRolloutBuffer360(
            self.num_steps, self.num_envs, self.obs_dim, self.act_dim, device
        )

    def train(self, logger=None):
        global_step = 0
        start_time = time.time()

        obs, _ = self.envs.reset()
        next_obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        next_done = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        num_updates = self.total_timesteps // self.batch_size
        print(f"🚀 [Standard-PPO-360] Starting Training: {self.total_timesteps:,} steps | Envs: {self.num_envs}")

        ep_returns = []
        ep_costs = []
        recent_returns = []
        recent_costs = []

        cur_ep_returns = np.zeros(self.num_envs)
        cur_ep_costs = np.zeros(self.num_envs)

        for update in range(1, num_updates + 1):
            for step in range(self.num_steps):
                global_step += self.num_envs

                with torch.no_grad():
                    action, logprob, _, r_val = self.agent.get_action_and_value(next_obs)

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

                t_obs = next_obs
                t_act = action
                t_logprob = logprob
                t_rew = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
                t_cost = torch.as_tensor(step_costs, dtype=torch.float32, device=self.device)
                t_done = next_done

                self.buffer.add(t_obs, t_act, t_logprob, t_rew, t_cost, t_done, r_val, None)

                next_obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                next_done = torch.as_tensor(done, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                next_r_val = self.agent.get_value(next_obs)

            self.buffer.compute_gae(
                next_obs, next_done, next_r_val, None,
                self.gamma, self.gae_lambda
            )

            for epoch in range(self.update_epochs):
                for b_obs, b_act, b_logp, b_r_adv, b_r_ret, b_r_val, _, _, _ in self.buffer.get_batches(self.minibatch_size):
                    _, new_logp, entropy, new_r_val = self.agent.get_action_and_value(b_obs, b_act)
                    logratio = new_logp - b_logp
                    ratio = torch.exp(logratio)

                    norm_r_adv = (b_r_adv - b_r_adv.mean()) / (b_r_adv.std() + 1e-8)

                    pg_loss1 = -norm_r_adv * ratio
                    pg_loss2 = -norm_r_adv * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    if self.clip_vloss:
                        v_loss_unclipped = (new_r_val - b_r_ret) ** 2
                        v_clipped = b_r_val + torch.clamp(new_r_val - b_r_val, -self.clip_coef, self.clip_coef)
                        v_loss_clipped = (v_clipped - b_r_ret) ** 2
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = 0.5 * ((new_r_val - b_r_ret) ** 2).mean()

                    entropy_loss = entropy.mean()
                    total_loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

            sps = int(global_step / (time.time() - start_time))
            avg_ret = float(np.mean(recent_returns[-30:])) if len(recent_returns) > 0 else 0.0
            avg_cst = float(np.mean(recent_costs[-30:])) if len(recent_costs) > 0 else 0.0

            if update % 5 == 0 or update == num_updates:
                print(
                    f"[{update:03d}/{num_updates:03d}] Steps: {global_step:7d} | "
                    f"Return: {avg_ret:6.1f} | Cost: {avg_cst:5.2f} | SPS: {sps}"
                )

        final_path = os.path.join(self.save_dir, "ppo_360_final.pt")
        torch.save(self.agent.state_dict(), final_path)
        print(f"✅ Saved trained policy to: {final_path}")
        return self.agent
