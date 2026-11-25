import torch
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from envs.navigation_env_360 import SafeNav360Env

def evaluate_policy_360(model, device, num_episodes=10, is_lagrangian=True):
    """
    Evaluates policy deterministically over test episodes.
    Reports:
      - Mean Return +/- Std
      - Mean Cost +/- Std
      - Collision Rate (% of episodes with a collision)
    """
    env = SafeNav360Env(randomize_obstacles=False, max_steps=500)
    model.eval()

    returns = []
    costs = []
    collisions = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        ep_ret = 0.0
        ep_cost = 0.0
        ep_col = False

        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = model.act_deterministic(obs_tensor).cpu().numpy()[0]

            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, rew, terminated, truncated, info = env.step(action)
            ep_ret += rew
            ep_cost += info.get("cost", 0.0)
            if info.get("is_collision", False):
                ep_col = True
            done = terminated or truncated

        returns.append(ep_ret)
        costs.append(ep_cost)
        collisions.append(1.0 if ep_col else 0.0)

    model.train()
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
        "collision_rate": float(np.mean(collisions)) * 100.0
    }

def generate_comparison_plot(ppo_metrics, lag_metrics, cost_limit, output_path):
    """Generates comparison bar charts between Standard PPO and PPO-Lagrangian"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    algos = ['Standard PPO (360)', 'PPO-Lagrangian (360)']

    # 1. Mean Return
    returns = [ppo_metrics['mean_return'], lag_metrics['mean_return']]
    yerr_ret = [ppo_metrics['std_return'], lag_metrics['std_return']]
    axes[0].bar(algos, returns, yerr=yerr_ret, capsize=8, color=['#e74c3c', '#2ecc71'])
    axes[0].set_title('Mean Episode Return (Higher is Better)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Cumulative Return')
    axes[0].grid(axis='y', alpha=0.3)

    # 2. Mean Safety Cost
    costs = [ppo_metrics['mean_cost'], lag_metrics['mean_cost']]
    yerr_cost = [ppo_metrics['std_cost'], lag_metrics['std_cost']]
    axes[1].bar(algos, costs, yerr=yerr_cost, capsize=8, color=['#e74c3c', '#2ecc71'])
    axes[1].axhline(y=cost_limit, color='black', linestyle='--', linewidth=2, label=f'Safety Budget ($d_k={cost_limit}$)')
    axes[1].set_title('Safety Cost (Target $\\leq$ Budget)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Mean Safety Cost')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    # 3. Collision Rate
    col_rates = [ppo_metrics['collision_rate'], lag_metrics['collision_rate']]
    axes[2].bar(algos, col_rates, color=['#e74c3c', '#2ecc71'])
    axes[2].set_title('Collision Rate % (Zero Collision Target)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Collision Rate (%)')
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"📊 Saved comparison plot to: {output_path}")
