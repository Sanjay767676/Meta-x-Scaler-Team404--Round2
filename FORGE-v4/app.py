# app.py
# Main runner script for FORGE-v4.
# Runs one demo episode with the improving_coder and tiered BreakerAgent,
# then prints a structured results report.

import sys
import json

from env import FORGEEnv
from memory import CoachMemory
from agents import get_coder_code, coder_version_label, BreakerAgent
from logger import log_episode, update_summary, print_log_paths
from config import STEPS_PER_EPISODE


# ──────────────────────────────────────────────
# Demo configuration
# ──────────────────────────────────────────────
DEFAULT_CODER_VERSION = "improving_coder"


def run_demo_episode(coder_version: str = DEFAULT_CODER_VERSION) -> None:
    """
    Execute one demo episode and print a rich results report.

    Args:
        coder_version: Which coder strategy to use.
            "weak_coder_v1" | "weak_coder_v2" | "improving_coder"
    """
    _banner()

    memory = CoachMemory()
    memory.clear()          # Start fresh for the demo run
    env = FORGEEnv(memory=memory)
    state = env.reset()

    episode = state["episode"]
    print(f"\n{'─'*60}")
    print(f"  Task ID  : {state['task_id']}")
    print(f"  Episode  : {episode}")
    print(f"  Coder    : {coder_version_label(coder_version, episode)}")
    print(f"  Breaker  : {env.breaker.tier_name}  (starts here, tiers up during run)")
    print(f"{'─'*60}")
    print(f"\n  Problem:\n")
    print(f"  {state['problem_description']}")
    print()

    # ── Accumulators ──────────────────────────────────────────────────────
    ep_coder_rewards:   list[float] = []
    ep_breaker_rewards: list[float] = []
    ep_pass_rates:      list[float] = []
    ep_fail_counts:     list[int]   = []
    ep_error_counts:    list[int]   = []
    ep_timeout_counts:  list[int]   = []
    ep_break_rates:     list[float] = []

    for step_num in range(1, STEPS_PER_EPISODE + 1):
        # Build coder action
        code   = get_coder_code(coder_version, episode=episode)
        action = {"coder_code": code, "coder_version": coder_version}

        result = env.step(action)
        state  = result["state"]

        cr = result["coder_reward"]
        br = result["breaker_reward"]
        info = result["info"]

        # Accumulate
        ep_coder_rewards.append(cr["total_reward"])
        ep_breaker_rewards.append(br["total_reward"])
        ep_pass_rates.append(cr["pass_rate"])
        ep_fail_counts.append(cr["fail_count"])
        ep_error_counts.append(cr["error_count"])
        ep_timeout_counts.append(cr["error_count"])
        ep_break_rates.append(br["break_rate"])

        # Per-step print
        print(f"  ── Step {step_num}/{STEPS_PER_EPISODE}  [breaker: {info['breaker_tier_name']}]")
        print(
            f"     Coder   → pass_rate: {cr['pass_rate']:.2f}  "
            f"| passes: {cr['pass_count']}  "
            f"| fails: {cr['fail_count']}  "
            f"| errors: {cr['error_count']}  "
            f"| reward: {cr['total_reward']:+.2f}"
        )
        print(
            f"     Breaker → break_rate: {br['break_rate']:.2f}  "
            f"| breaks: {br['breaks']}  "
            f"| no-break: {br['passes']}  "
            f"| reward: {br['total_reward']:+.2f}"
        )
        if state.get("recent_breaker_case") is not None:
            print(f"     Recent adversarial input: {state['recent_breaker_case']}")
        print()

        if result["done"]:
            break

    # ── Episode log ───────────────────────────────────────────────────────
    def avg(lst: list) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    log_episode(
        episode=episode,
        coder_version=coder_version,
        breaker_tier=env.breaker.current_tier,
        avg_coder_reward=avg(ep_coder_rewards),
        avg_breaker_reward=avg(ep_breaker_rewards),
        avg_pass_rate=avg(ep_pass_rates),
        total_fail_count=sum(ep_fail_counts),
        total_error_count=sum(ep_error_counts),
        total_timeout_count=sum(ep_timeout_counts),
        avg_break_rate=avg(ep_break_rates),
        steps=env.step_count,
    )

    update_summary(
        total_episodes=1,
        coder_version=coder_version,
        final_breaker_tier=env.breaker.current_tier,
        all_coder_rewards=ep_coder_rewards,
        all_breaker_rewards=ep_breaker_rewards,
        all_pass_rates=ep_pass_rates,
        all_break_rates=ep_break_rates,
        coach_memory_summary=memory.summary(),
    )

    # ── Final report ──────────────────────────────────────────────────────
    print(f"{'═'*60}")
    print("  EPISODE SUMMARY")
    print(f"{'═'*60}")
    print(f"  Coder version       : {coder_version_label(coder_version, episode)}")
    print(f"  Final breaker tier  : {env.breaker.tier_name}")
    print(f"  Avg pass rate       : {avg(ep_pass_rates):.2f}")
    print(f"  Avg coder reward    : {avg(ep_coder_rewards):+.4f}")
    print(f"  Avg breaker reward  : {avg(ep_breaker_rewards):+.4f}")
    print(f"  Total fail count    : {sum(ep_fail_counts)}")
    print(f"  Total error count   : {sum(ep_error_counts)}")
    print(f"  Avg break rate      : {avg(ep_break_rates):.2f}")
    print()
    print("  Coach memory summary:")
    summary = memory.summary()
    print(f"    Lessons stored    : {summary.get('total_lessons', 0)}")
    notes = summary.get("recent_coach_notes", [])
    if notes:
        print("    Recent coach notes:")
        for note in notes:
            print(f"      • {note}")
    print()
    print("  Log files updated:")
    print_log_paths()
    print(f"{'═'*60}")


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _banner() -> None:
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   FORGE-v4  |  Adversarial Code Generation Environment  ║")
    print("╚══════════════════════════════════════════════════════════╝")


def _print_help() -> None:
    print("Usage: python app.py [OPTIONS]")
    print()
    print("Options:")
    print("  --coder VERSION   Coder strategy to use:")
    print("                      weak_coder_v1   (bubble sort — slow/weak)")
    print("                      weak_coder_v2   (selection sort + abs() bug)")
    print("                      improving_coder (adapts each episode)  [default]")
    print("  --steps N         Override STEPS_PER_EPISODE for this run")
    print("  --help / -h       Show this message")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        _print_help()
        sys.exit(0)

    coder_version = DEFAULT_CODER_VERSION
    if "--coder" in args:
        idx = args.index("--coder")
        try:
            coder_version = args[idx + 1]
            valid = ("weak_coder_v1", "weak_coder_v2", "improving_coder")
            if coder_version not in valid:
                print(f"Error: unknown coder version '{coder_version}'. Choose from: {valid}")
                sys.exit(1)
        except IndexError:
            print("Error: --coder requires a version argument.")
            sys.exit(1)

    if "--steps" in args:
        idx = args.index("--steps")
        try:
            import config
            config.STEPS_PER_EPISODE = int(args[idx + 1])
        except (IndexError, ValueError):
            print("Error: --steps requires an integer argument.")
            sys.exit(1)

    run_demo_episode(coder_version=coder_version)


if __name__ == "__main__":
    main()
