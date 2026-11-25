import torch

class SafeRolloutBufferGeneral:
    """Rollout buffer for PPO-Lagrangian storing reward and safety cost transitions"""
    def __init__(self, num_steps, num_envs, obs_dim, act_dim, device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self.reset()

    def reset(self):
        self.obs = torch.zeros((self.num_steps, self.num_envs, self.obs_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.num_steps, self.num_envs, self.act_dim), dtype=torch.float32, device=self.device)
        self.logprobs = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.costs = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.reward_values = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.cost_values = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.step = 0

    def add(self, obs, action, logprob, reward, cost, done, reward_value, cost_value):
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.logprobs[self.step] = logprob
        self.rewards[self.step] = reward
        self.costs[self.step] = cost
        self.dones[self.step] = done
        self.reward_values[self.step] = reward_value
        self.cost_values[self.step] = cost_value
        self.step += 1

    def compute_gae(self, next_obs, next_done, next_r_val, next_c_val, gamma, gae_lambda, cost_gamma, cost_gae_lambda):
        self.reward_advantages = torch.zeros_like(self.rewards)
        self.cost_advantages = torch.zeros_like(self.costs)

        last_r_gae = 0.0
        last_c_gae = 0.0

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - next_done
                next_r_value = next_r_val
                next_c_value = next_c_val
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_r_value = self.reward_values[t + 1]
                next_c_value = self.cost_values[t + 1]

            # Reward GAE
            r_delta = self.rewards[t] + gamma * next_r_value * next_non_terminal - self.reward_values[t]
            last_r_gae = r_delta + gamma * gae_lambda * next_non_terminal * last_r_gae
            self.reward_advantages[t] = last_r_gae

            # Cost GAE
            c_delta = self.costs[t] + cost_gamma * next_c_value * next_non_terminal - self.cost_values[t]
            last_c_gae = c_delta + cost_gamma * cost_gae_lambda * next_non_terminal * last_c_gae
            self.cost_advantages[t] = last_c_gae

        self.reward_returns = self.reward_advantages + self.reward_values
        self.cost_returns = self.cost_advantages + self.cost_values

class RolloutBufferGeneral:
    """Standard unconstrained PPO rollout buffer"""
    def __init__(self, num_steps, num_envs, obs_dim, act_dim, device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self.reset()

    def reset(self):
        self.obs = torch.zeros((self.num_steps, self.num_envs, self.obs_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.num_steps, self.num_envs, self.act_dim), dtype=torch.float32, device=self.device)
        self.logprobs = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.values = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.step = 0

    def add(self, obs, action, logprob, reward, done, value):
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.logprobs[self.step] = logprob
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value
        self.step += 1

    def compute_gae(self, next_obs, next_done, next_val, gamma, gae_lambda):
        self.advantages = torch.zeros_like(self.rewards)
        last_gae = 0.0

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - next_done
                next_value = next_val
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values
