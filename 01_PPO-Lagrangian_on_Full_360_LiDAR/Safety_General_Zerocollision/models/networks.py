import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal layer initialization for stable PPO convergence"""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class Lidar360FeatureExtractor(nn.Module):
    """
    Spatial 1D-CNN + Proprioceptive Feature Extractor for 360 continuous LiDAR rays.
    1D-CNN extracts local spatial obstacle free-space features around the unicycle robot.
    """
    def __init__(self, lidar_dim=360, extra_dim=5, feature_dim=128):
        super().__init__()
        self.lidar_dim = lidar_dim
        self.extra_dim = extra_dim

        # 3-layer 1D Convolution with circular awareness
        self.lidar_conv = nn.Sequential(
            layer_init(nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2)), # 360 -> 180
            nn.ReLU(),
            layer_init(nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)), # 180 -> 90
            nn.ReLU(),
            layer_init(nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)), # 90 -> 45
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8), # 64 x 8 = 512
            nn.Flatten(),
            layer_init(nn.Linear(64 * 8, 128)),
            nn.ReLU()
        )

        # Fusion layer combining spatial LiDAR latent features with target distance & heading
        self.fusion = nn.Sequential(
            layer_init(nn.Linear(128 + extra_dim, feature_dim)),
            nn.ReLU()
        )

    def forward(self, x):
        lidar = x[:, :self.lidar_dim].unsqueeze(1)
        extra = x[:, self.lidar_dim:]
        lidar_feats = self.lidar_conv(lidar)
        combined = torch.cat([lidar_feats, extra], dim=1)
        return self.fusion(combined)

class SafeActorCriticGeneral(nn.Module):
    """
    Constrained Safe Actor-Critic with Twin Critics (Reward Critic V^R and Cost Critic V^C).
    Features a Tanh-bounded actor mean ensuring perfect left-right steering symmetry.
    """
    def __init__(self, obs_dim=365, act_dim=2, feature_dim=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Separate encoders for policy and critics
        self.actor_encoder = Lidar360FeatureExtractor(lidar_dim=360, extra_dim=obs_dim-360, feature_dim=feature_dim)
        self.reward_encoder = Lidar360FeatureExtractor(lidar_dim=360, extra_dim=obs_dim-360, feature_dim=feature_dim)
        self.cost_encoder = Lidar360FeatureExtractor(lidar_dim=360, extra_dim=obs_dim-360, feature_dim=feature_dim)

        # Actor head: Outputs mean in [-1, 1] via Tanh (Perfect Left/Right Symmetry)
        self.actor_head = nn.Sequential(
            layer_init(nn.Linear(feature_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
            nn.Tanh()
        )
        self.actor_logstd = nn.Parameter(torch.full((1, act_dim), -0.5))

        # Reward Critic Head (V^R)
        self.reward_critic = nn.Sequential(
            layer_init(nn.Linear(feature_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

        # Safety Cost Critic Head (V^C)
        self.cost_critic = nn.Sequential(
            layer_init(nn.Linear(feature_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

    def get_action_distribution(self, x):
        feats = self.actor_encoder(x)
        action_mean = self.actor_head(feats)
        action_logstd = torch.clamp(self.actor_logstd.expand_as(action_mean), -2.0, 0.5)
        action_std = torch.exp(action_logstd)
        return Normal(action_mean, action_std)

    def get_action_and_values(self, x, action=None):
        probs = self.get_action_distribution(x)
        if action is None:
            action = probs.rsample()

        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)

        r_feats = self.reward_encoder(x)
        reward_value = self.reward_critic(r_feats).squeeze(-1)

        c_feats = self.cost_encoder(x)
        cost_value = self.cost_critic(c_feats).squeeze(-1)

        return action, log_prob, entropy, reward_value, cost_value

    def get_reward_value(self, x):
        r_feats = self.reward_encoder(x)
        return self.reward_critic(r_feats).squeeze(-1)

    def get_cost_value(self, x):
        c_feats = self.cost_encoder(x)
        return self.cost_critic(c_feats).squeeze(-1)

    def act_deterministic(self, x):
        """Zero stochasticity deterministic policy inference for deployment"""
        feats = self.actor_encoder(x)
        return self.actor_head(feats)

class ActorCriticGeneral(nn.Module):
    """
    Standard Unconstrained PPO Actor-Critic (Single Critic V^R).
    Used as the baseline to show how unconstrained PPO collides with obstacles.
    """
    def __init__(self, obs_dim=365, act_dim=2, feature_dim=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.actor_encoder = Lidar360FeatureExtractor(lidar_dim=360, extra_dim=obs_dim-360, feature_dim=feature_dim)
        self.reward_encoder = Lidar360FeatureExtractor(lidar_dim=360, extra_dim=obs_dim-360, feature_dim=feature_dim)

        self.actor_head = nn.Sequential(
            layer_init(nn.Linear(feature_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
            nn.Tanh()
        )
        self.actor_logstd = nn.Parameter(torch.full((1, act_dim), -0.5))

        self.reward_critic = nn.Sequential(
            layer_init(nn.Linear(feature_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

    def get_action_distribution(self, x):
        feats = self.actor_encoder(x)
        action_mean = self.actor_head(feats)
        action_logstd = torch.clamp(self.actor_logstd.expand_as(action_mean), -2.0, 0.5)
        action_std = torch.exp(action_logstd)
        return Normal(action_mean, action_std)

    def get_action_and_value(self, x, action=None):
        probs = self.get_action_distribution(x)
        if action is None:
            action = probs.rsample()
        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        r_feats = self.reward_encoder(x)
        reward_value = self.reward_critic(r_feats).squeeze(-1)
        return action, log_prob, entropy, reward_value

    def get_value(self, x):
        r_feats = self.reward_encoder(x)
        return self.reward_critic(r_feats).squeeze(-1)

    def act_deterministic(self, x):
        feats = self.actor_encoder(x)
        return self.actor_head(feats)
