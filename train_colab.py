#!/usr/bin/env python3
# train_colab.py
# One-click training entrypoint for Google Colab and CLI.

import sys
import argparse
from trainer import run_benchmark_mode, run_compare_mode
from config import ensure_runtime_dirs

def main():
    parser = argparse.ArgumentParser(description="FORGE-v4 Colab Training Entrypoint")
    parser.add_argument("--compare", action="store_true", help="Run baseline vs model comparison")
    parser.add_argument("--benchmark", action="store_true", help="Run single policy benchmark")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to run")
    parser.add_argument("--policy", type=str, default="model", help="Policy name for benchmark mode")
    parser.add_argument("--candidates", type=int, default=3, help="Candidates per step")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose progress logs")
    
    args = parser.parse_args()
    
    ensure_runtime_dirs()
    print("\n" + "="*60)
    print("  FORGE-v4: Colab Training Entrypoint")
    print("="*60 + "\n")
    
    verbose = not args.quiet
    
    if args.compare:
        print(f"  Starting comparison mode ({args.episodes} episodes)...")
        run_compare_mode(
            model_policy_name="model",
            episodes=args.episodes,
            candidates_per_step=args.candidates,
            verbose=verbose
        )
    elif args.benchmark:
        print(f"  Starting benchmark mode for '{args.policy}' ({args.episodes} episodes)...")
        run_benchmark_mode(
            policy_name=args.policy,
            episodes=args.episodes,
            candidates_per_step=args.candidates,
            verbose=verbose
        )
    else:
        # Default to comparison if no mode specified
        print(f"  No mode specified. Defaulting to comparison ({args.episodes} episodes)...")
        run_compare_mode(
            model_policy_name="model",
            episodes=args.episodes,
            candidates_per_step=args.candidates,
            verbose=verbose
        )

    print("\n" + "="*60)
    print("  Run complete. Results available in outputs/ and logs/")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
