#!/usr/bin/env python3
import os
import sys
import argparse
import time
import shutil
import torch
import numpy as np

# Add folder to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from envs.make_env import make_vector_env
from algorithms.ppo import PPO360
from algorithms.ppo_lagrangian import PPOLagrangian360
from utils.eval import evaluate_policy_360, generate_comparison_plot

def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End 360-Ray Zero Collision Safe RL")
    parser.add_argument("--algo", type=str, choices=["ppo", "ppo_lagrangian", "both"], default="both",
                        help="Algorithm to train: 'ppo', 'ppo_lagrangian', or 'both'")
    parser.add_argument("--total_timesteps", type=int, default=400000,
                        help="Total timesteps per algorithm (default: 400000)")
    parser.add_argument("--num_envs", type=int, default=8,
                        help="Number of vectorized parallel environments")
    parser.add_argument("--cost_limit", type=float, default=0.0,
                        help="Safety cost limit d_k (default: 0.0 for strict Zero Collision)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Torch device ('cuda' or 'cpu')")
    return parser.parse_args()

def main():
    args = parse_args()

    # Device allocation
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("🤖 End-to-End 360-Ray Safe Mobile Robot Navigation (Zero-Collision)")
    print(f"Device: {device} | Total Timesteps: {args.total_timesteps:,} | Target Budget d_k: {args.cost_limit}")
    print("=" * 70)

    # Output directory
    timestamp = int(time.time())
    run_dir = os.path.join(current_dir, "runs", f"run_360_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    models_saved_dir = os.path.join(current_dir, "models_saved")
    os.makedirs(models_saved_dir, exist_ok=True)

    ppo_metrics = None
    lag_metrics = None

    # 1. Train Standard PPO (Baseline)
    if args.algo in ["ppo", "both"]:
        print("\n" + "="*50)
        print("STAGE 1: Training Unconstrained Standard PPO (360 Rays)")
        print("="*50)
        envs = make_vector_env(num_envs=args.num_envs, seed=args.seed)
        ppo_trainer = PPO360(
            envs=envs,
            device=device,
            total_timesteps=args.total_timesteps,
            save_dir=run_dir
        )
        ppo_model = ppo_trainer.train()
        envs.close()

        # Copy to models_saved
        src = os.path.join(run_dir, "ppo_360_final.pt")
        dst = os.path.join(models_saved_dir, "ppo_360_final.pt")
        if os.path.exists(src):
            shutil.copyfile(src, dst)

        print("\nEvaluating Standard PPO...")
        ppo_metrics = evaluate_policy_360(ppo_model, device, num_episodes=20, is_lagrangian=False)
        print(f"PPO Results -> Return: {ppo_metrics['mean_return']:.2f} +/- {ppo_metrics['std_return']:.2f} | "
              f"Cost: {ppo_metrics['mean_cost']:.2f} +/- {ppo_metrics['std_cost']:.2f} | "
              f"Collision Rate: {ppo_metrics['collision_rate']:.1f}%")

    # 2. Train PPO-Lagrangian (Safe Constrained RL)
    if args.algo in ["ppo_lagrangian", "both"]:
        print("\n" + "="*50)
        print("STAGE 2: Training Zero-Collision PPO-Lagrangian (360 Rays)")
        print("="*50)
        envs = make_vector_env(num_envs=args.num_envs, seed=args.seed + 100)
        lag_trainer = PPOLagrangian360(
            envs=envs,
            device=device,
            total_timesteps=args.total_timesteps,
            cost_limit=args.cost_limit,
            save_dir=run_dir
        )
        lag_model = lag_trainer.train()
        envs.close()

        # Copy to models_saved
        src = os.path.join(run_dir, "ppo_lagrangian_360_final.pt")
        dst = os.path.join(models_saved_dir, "ppo_lagrangian_360_final.pt")
        if os.path.exists(src):
            shutil.copyfile(src, dst)

        print("\nEvaluating PPO-Lagrangian...")
        lag_metrics = evaluate_policy_360(lag_model, device, num_episodes=20, is_lagrangian=True)
        print(f"PPO-Lagrangian Results -> Return: {lag_metrics['mean_return']:.2f} +/- {lag_metrics['std_return']:.2f} | "
              f"Cost: {lag_metrics['mean_cost']:.2f} +/- {lag_metrics['std_cost']:.2f} | "
              f"Collision Rate: {lag_metrics['collision_rate']:.1f}%")

    # 3. Final Summary Table & Comparison Plot
    if ppo_metrics and lag_metrics:
        plot_path = os.path.join(run_dir, "comparison_360_reward_vs_cost.png")
        generate_comparison_plot(ppo_metrics, lag_metrics, args.cost_limit, plot_path)
        shutil.copyfile(plot_path, os.path.join(models_saved_dir, "comparison_360_reward_vs_cost.png"))

        print("\n" + "="*70)
        print("🏆 FINAL ZERO-COLLISION 360-RAY COMPARISON RESULTS")
        print("="*70)
        print(f"{'Algorithm':<20} | {'Mean Return':<20} | {'Mean Cost':<15} | {'Collision Rate':<15} | {'Zero-Collision?'}")
        print("-" * 90)
        p_sat = "YES ✅" if ppo_metrics['collision_rate'] == 0.0 else f"NO ❌ ({ppo_metrics['collision_rate']:.1f}%)"
        l_sat = "YES ✅" if lag_metrics['collision_rate'] == 0.0 else f"{lag_metrics['collision_rate']:.1f}%"
        print(f"{'Standard PPO':<20} | {ppo_metrics['mean_return']:6.2f} +/- {ppo_metrics['std_return']:5.2f}  | {ppo_metrics['mean_cost']:6.2f} +/- {ppo_metrics['std_cost']:4.2f} | {ppo_metrics['collision_rate']:5.1f}%          | {p_sat}")
        print(f"{'PPO-Lagrangian':<20} | {lag_metrics['mean_return']:6.2f} +/- {lag_metrics['std_return']:5.2f}  | {lag_metrics['mean_cost']:6.2f} +/- {lag_metrics['std_cost']:4.2f} | {lag_metrics['collision_rate']:5.1f}%          | {l_sat}")
        print("="*70)

if __name__ == '__main__':
    main()
