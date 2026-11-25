import torch
import numpy as np

class SafeRolloutBuffer360:
    """
    Rollout buffer for 360-ray safe reinforcement learning.
    Stores transitions and computes Generalized Advantage Estimation (GAE)
    for both reward returns and safety cost returns.
    """
    def __init__(self, num_steps, num_envs, obs_dim, act_dim, device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device

        self.obs = torch.zeros((num_steps, num_envs, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((num_steps, num_envs, act_dim), dtype=torch.float32, device=device)
        self.logprobs = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.costs = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.reward_values = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.cost_values = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)

        self.reward_advantages = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.reward_returns = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.cost_advantages = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.cost_returns = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)

        self.step = 0

    def add(self, obs, action, logprob, reward, cost, done, reward_val, cost_val=None):
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.logprobs[self.step] = logprob
        self.rewards[self.step] = reward
        self.costs[self.step] = cost
        self.dones[self.step] = done
        self.reward_values[self.step] = reward_val
        if cost_val is not None:
            self.cost_values[self.step] = cost_val
        self.step += 1

    def compute_gae(self, next_obs, next_done, next_reward_val, next_cost_val=None,
                    gamma=0.99, gae_lambda=0.95, cost_gamma=0.99, cost_gae_lambda=0.95):
        # Reward GAE
        last_gae_r = 0.0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - next_done.float()
                next_r_val = next_reward_val
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_r_val = self.reward_values[t + 1]

            delta_r = self.rewards[t] + gamma * next_r_val * next_non_terminal - self.reward_values[t]
            last_gae_r = delta_r + gamma * gae_lambda * next_non_terminal * last_gae_r
            self.reward_advantages[t] = last_gae_r
        self.reward_returns = self.reward_advantages + self.reward_values

        # Cost GAE (if cost_val provided)
        if next_cost_val is not None:
            last_gae_c = 0.0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    next_non_terminal = 1.0 - next_done.float()
                    next_c_val = next_cost_val
                else:
                    next_non_terminal = 1.0 - self.dones[t + 1]
                    next_c_val = self.cost_values[t + 1]

                delta_c = self.costs[t] + cost_gamma * next_c_val * next_non_terminal - self.cost_values[t]
                last_gae_c = delta_c + cost_gamma * cost_gae_lambda * next_non_terminal * last_gae_c
                self.cost_advantages[t] = last_gae_c
            self.cost_returns = self.cost_advantages + self.cost_values

        self.step = 0

    def get_batches(self, batch_size):
        b_obs = self.obs.reshape((-1, self.obs_dim))
        b_actions = self.actions.reshape((-1, self.act_dim))
        b_logprobs = self.logprobs.reshape(-1)
        b_r_adv = self.reward_advantages.reshape(-1)
        b_r_returns = self.reward_returns.reshape(-1)
        b_r_values = self.reward_values.reshape(-1)

        b_c_adv = self.cost_advantages.reshape(-1)
        b_c_returns = self.cost_returns.reshape(-1)
        b_c_values = self.cost_values.reshape(-1)

        total_size = b_obs.shape[0]
        indices = np.arange(total_size)
        np.random.shuffle(indices)

        for start in range(0, total_size, batch_size):
            end = start + batch_size
            idx = indices[start:end]
            yield (
                b_obs[idx],
                b_actions[idx],
                b_logprobs[idx],
                b_r_adv[idx],
                b_r_returns[idx],
                b_r_values[idx],
                b_c_adv[idx],
                b_c_returns[idx],
                b_c_values[idx]
            )
