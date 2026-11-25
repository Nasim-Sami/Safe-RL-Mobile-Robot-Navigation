#!/usr/bin/env python3
import os
import sys
import argparse
import time
import shutil
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from envs.make_env import make_vector_env
from algorithms.ppo import PPOGeneral
from algorithms.ppo_lagrangian import PPOLagrangianGeneral
from utils.eval import evaluate_general_policy, generate_general_comparison_plot

def parse_args():
    parser = argparse.ArgumentParser(description="Generalizable 360-Ray Zero-Collision Safe RL")
    parser.add_argument("--algo", type=str, choices=["ppo", "ppo_lagrangian", "both"], default="both")
    parser.add_argument("--total_timesteps", type=int, default=400000)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--cost_limit", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("🌍 Generalizable 360-Ray Safe Mobile Robot Navigation (Domain Randomized)")
    print(f"Device: {device} | Total Timesteps: {args.total_timesteps:,} | Target Budget d_k: {args.cost_limit}")
    print("=" * 70)

    timestamp = int(time.time())
    run_dir = os.path.join(current_dir, "runs", f"run_general_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    models_saved_dir = os.path.join(current_dir, "models_saved")
    os.makedirs(models_saved_dir, exist_ok=True)

    ppo_metrics = None
    lag_metrics = None

    # 1. Train Standard PPO Baseline
    if args.algo in ["ppo", "both"]:
        print("\n" + "="*50)
        print("STAGE 1: Training Standard Unconstrained PPO Baseline")
        print("="*50)
        envs = make_vector_env(num_envs=args.num_envs, seed=args.seed)
        ppo_trainer = PPOGeneral(
            envs=envs,
            device=device,
            total_timesteps=args.total_timesteps,
            save_dir=run_dir
        )
        ppo_model = ppo_trainer.train()
        envs.close()

        src = os.path.join(run_dir, "ppo_general_final.pt")
        dst = os.path.join(models_saved_dir, "ppo_general_final.pt")
        if os.path.exists(src):
            shutil.copyfile(src, dst)

        print("\nEvaluating Standard PPO...")
        ppo_metrics = evaluate_general_policy(ppo_model, device, num_episodes=20, is_lagrangian=False)
        print(f"PPO Results -> Return: {ppo_metrics['mean_return']:.2f} +/- {ppo_metrics['std_return']:.2f} | "
              f"Cost: {ppo_metrics['mean_cost']:.2f} | "
              f"Collision Rate: {ppo_metrics['collision_rate']:.1f}% | "
              f"Success Rate: {ppo_metrics['success_rate']:.1f}%")

    # 2. Train Constrained Safe PPO-Lagrangian
    if args.algo in ["ppo_lagrangian", "both"]:
        print("\n" + "="*50)
        print("STAGE 2: Training Constrained Safe PPO-Lagrangian (Zero Collision)")
        print("="*50)
        envs = make_vector_env(num_envs=args.num_envs, seed=args.seed + 100)
        lag_trainer = PPOLagrangianGeneral(
            envs=envs,
            device=device,
            total_timesteps=args.total_timesteps,
            cost_limit=args.cost_limit,
            save_dir=run_dir
        )
        lag_model = lag_trainer.train()
        envs.close()

        src = os.path.join(run_dir, "ppo_lagrangian_general_final.pt")
        dst = os.path.join(models_saved_dir, "ppo_lagrangian_general_final.pt")
        if os.path.exists(src):
            shutil.copyfile(src, dst)

        print("\nEvaluating Safe PPO-Lagrangian...")
        lag_metrics = evaluate_general_policy(lag_model, device, num_episodes=20, is_lagrangian=True)
        print(f"Safe PPO-Lagrangian Results -> Return: {lag_metrics['mean_return']:.2f} +/- {lag_metrics['std_return']:.2f} | "
              f"Cost: {lag_metrics['mean_cost']:.2f} | "
              f"Collision Rate: {lag_metrics['collision_rate']:.1f}% | "
              f"Success Rate: {lag_metrics['success_rate']:.1f}%")

    # 3. Save Summary Plot & Print Table
    if ppo_metrics and lag_metrics:
        plot_path = os.path.join(run_dir, "comparison_general_reward_vs_cost.png")
        generate_general_comparison_plot(ppo_metrics, lag_metrics, args.cost_limit, plot_path)
        shutil.copyfile(plot_path, os.path.join(models_saved_dir, "comparison_general_reward_vs_cost.png"))

        print("\n" + "="*80)
        print("🏆 GENERALIZABLE ZERO-COLLISION 360-RAY BENCHMARK RESULTS")
        print("="*80)
        print(f"{'Algorithm':<22} | {'Return':<18} | {'Cost':<12} | {'Collision':<12} | {'Success':<12}")
        print("-" * 80)
        print(f"{'Standard PPO':<22} | {ppo_metrics['mean_return']:6.2f} +/- {ppo_metrics['std_return']:5.2f} | {ppo_metrics['mean_cost']:5.2f}        | {ppo_metrics['collision_rate']:5.1f}%      | {ppo_metrics['success_rate']:5.1f}%")
        print(f"{'PPO-Lagrangian':<22} | {lag_metrics['mean_return']:6.2f} +/- {lag_metrics['std_return']:5.2f} | {lag_metrics['mean_cost']:5.2f}        | {lag_metrics['collision_rate']:5.1f}%      | {lag_metrics['success_rate']:5.1f}%")
        print("="*80)

if __name__ == '__main__':
    main()
