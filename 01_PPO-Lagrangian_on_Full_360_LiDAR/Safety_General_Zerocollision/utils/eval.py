import torch
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from envs.navigation_env_general import SafeNavGeneralEnv

def evaluate_general_policy(model, device, num_episodes=20, is_lagrangian=True):
    """
    Evaluates policy across both randomized domain environments and structured corridors.
    Reports:
      - Mean Return +/- Std
      - Mean Cost +/- Std
      - Collision Rate (% episodes with collision)
      - Goal Success Rate (% episodes reaching target)
    """
    env = SafeNavGeneralEnv(randomize_obstacles=True, max_steps=500)
    model.eval()

    returns, costs, collisions, successes = [], [], [], []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=2000 + ep)
        done = False
        ep_ret = 0.0
        ep_cost = 0.0
        ep_col = False
        ep_goal = False

        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = model.act_deterministic(obs_tensor).cpu().numpy()[0]

            obs, rew, terminated, truncated, info = env.step(action)
            ep_ret += rew
            ep_cost += info.get("cost", 0.0)
            if info.get("is_collision", False):
                ep_col = True
            if info.get("is_goal", False):
                ep_goal = True
            done = terminated or truncated

        returns.append(ep_ret)
        costs.append(ep_cost)
        collisions.append(1.0 if ep_col else 0.0)
        successes.append(1.0 if ep_goal else 0.0)

    model.train()
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
        "collision_rate": float(np.mean(collisions)) * 100.0,
        "success_rate": float(np.mean(successes)) * 100.0
    }

def generate_general_comparison_plot(ppo_metrics, lag_metrics, cost_limit, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    algos = ['Standard PPO', 'PPO-Lagrangian (Safe RL)']

    # 1. Mean Return
    returns = [ppo_metrics['mean_return'], lag_metrics['mean_return']]
    yerr_ret = [ppo_metrics['std_return'], lag_metrics['std_return']]
    axes[0].bar(algos, returns, yerr=yerr_ret, capsize=8, color=['#e74c3c', '#2ecc71'])
    axes[0].set_title('Mean Episode Return', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Cumulative Return')
    axes[0].grid(axis='y', alpha=0.3)

    # 2. Mean Safety Cost
    costs = [ppo_metrics['mean_cost'], lag_metrics['mean_cost']]
    yerr_cost = [ppo_metrics['std_cost'], lag_metrics['std_cost']]
    axes[1].bar(algos, costs, yerr=yerr_cost, capsize=8, color=['#e74c3c', '#2ecc71'])
    axes[1].axhline(y=cost_limit, color='black', linestyle='--', linewidth=2, label=f'Safety Budget ($d_k={cost_limit}$)')
    axes[1].set_title('Safety Cost Violation', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Mean Cost')
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
