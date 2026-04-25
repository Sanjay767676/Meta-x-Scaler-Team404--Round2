# app.py
# Main runner script for FORGE-v4.
# Runs a minimal CLI demo of one sample episode.

import sys
import json
from env import FORGEEnv
from memory import CoachMemory
from trainer import default_coder_policy, default_breaker_policy
from config import STEPS_PER_EPISODE


def run_demo_episode() -> None:
    """
    Execute a single demo episode and print the results to stdout.
    """
    print("=" * 60)
    print("  FORGE-v4  |  Adversarial Code Generation Environment")
    print("=" * 60)

    # Initialise coach memory and environment
    memory = CoachMemory()
    env = FORGEEnv(memory=memory)

    # Reset to start the episode
    state = env.reset()

    print(f"\n[Episode {state['episode']}]  Task prompt:\n")
    print(state["task_prompt"])
    print()

    for step in range(1, STEPS_PER_EPISODE + 1):
        print(f"── Step {step}/{STEPS_PER_EPISODE} " + "─" * 40)

        # Agents produce their actions (placeholder policies for the demo)
        coder_code    = default_coder_policy(state)
        breaker_tests = default_breaker_policy(state)

        action = {
            "coder_code":    coder_code,
            "breaker_tests": breaker_tests,
        }

        result = env.step(action)

        cr = result["coder_reward"]
        br = result["breaker_reward"]

        print(
            f"  Coder   → pass_rate: {cr['pass_rate']:.2f}  "
            f"| passes: {cr['pass_count']}  "
            f"| fails: {cr['fail_count']}  "
            f"| errors: {cr['error_count']}  "
            f"| reward: {cr['total_reward']:+.2f}"
        )
        print(
            f"  Breaker → break_rate: {br['break_rate']:.2f}  "
            f"| breaks: {br['breaks']}  "
            f"| passes: {br['passes']}  "
            f"| reward: {br['total_reward']:+.2f}"
        )

        if result["done"]:
            break

    print("\n" + "=" * 60)
    print("  Episode complete.  Coach memory summary:")
    print(json.dumps(memory.summary(), indent=2))
    print("=" * 60)


def main() -> None:
    """Entry point — parse minimal CLI args and run."""
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print("Usage: python app.py [--steps N]")
        print("  --steps N   Override STEPS_PER_EPISODE for this run (default: from config.py)")
        sys.exit(0)

    # Optional: override step count via CLI
    if "--steps" in args:
        idx = args.index("--steps")
        try:
            import config
            config.STEPS_PER_EPISODE = int(args[idx + 1])
        except (IndexError, ValueError):
            print("Error: --steps requires an integer argument.")
            sys.exit(1)

    run_demo_episode()


if __name__ == "__main__":
    main()
