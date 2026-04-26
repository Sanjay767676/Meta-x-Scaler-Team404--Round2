#!/usr/bin/env python3
# train_colab.py
# One-click training entrypoint for Google Colab and CLI.

import sys
import argparse
from trainer import run_benchmark_mode, run_compare_mode
from config import DPO_DATASET_FILE, MIN_DPO_PAIRS_TARGET, ensure_runtime_dirs


def count_dpo_pairs(path: str = DPO_DATASET_FILE) -> int:
    """Count valid jsonl rows in DPO dataset file."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def topup_dpo_pairs(target_pairs: int, policy: str, candidates: int, quiet: bool) -> int:
    """Run additional benchmark batches until target DPO pair count is reached."""
    rounds = 0
    current = count_dpo_pairs()
    while current < target_pairs and rounds < 20:
        rounds += 1
        run_benchmark_mode(
            policy_name=policy,
            episodes=20,
            candidates_per_step=candidates,
            verbose=not quiet,
        )
        new_count = count_dpo_pairs()
        if new_count <= current:
            break
        current = new_count
    return current

def main():
    parser = argparse.ArgumentParser(description="FORGE-v4 Colab Training Entrypoint")
    parser.add_argument("--compare", action="store_true", help="Run baseline vs model comparison")
    parser.add_argument("--benchmark", action="store_true", help="Run single policy benchmark")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to run")
    parser.add_argument("--policy", type=str, default="model", help="Policy name for benchmark mode")
    parser.add_argument("--candidates", type=int, default=3, help="Candidates per step")
    parser.add_argument("--target-pairs", type=int, default=MIN_DPO_PAIRS_TARGET, help="Minimum DPO pairs to produce")
    parser.add_argument("--topup-dpo", action="store_true", help="Run extra batches until target pairs are reached")
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

    if args.topup_dpo:
        current = topup_dpo_pairs(
            target_pairs=max(1, args.target_pairs),
            policy=args.policy,
            candidates=args.candidates,
            quiet=args.quiet,
        )
        print(f"  DPO pairs available: {current}")

    print("\n" + "="*60)
    print("  Run complete. Results available in outputs/ and logs/")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
