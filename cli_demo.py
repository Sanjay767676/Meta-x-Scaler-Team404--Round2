# app.py
# Main runner script for FORGE-v4.
# Runs one demo episode with the improving_coder and tiered BreakerAgent,
# then prints a structured results report.

import sys

from env import FORGEEnv
from memory import CoachMemory
from agents import get_coder_code, coder_version_label
from logger import log_episode, update_summary, print_log_paths, write_episode_report
from config import DEFAULT_CANDIDATES_PER_STEP, STEPS_PER_EPISODE, ensure_runtime_dirs
from policies.factory import build_policy
from trainer import run_benchmark_mode, run_compare_mode


# ──────────────────────────────────────────────
# Demo configuration
# ──────────────────────────────────────────────
DEFAULT_CODER_VERSION = "improving_coder"
DEFAULT_POLICY = "heuristic"


def run_demo_episode(
    coder_version: str = DEFAULT_CODER_VERSION,
    policy_name: str = DEFAULT_POLICY,
    candidates_per_step: int = DEFAULT_CANDIDATES_PER_STEP,
    generate_metrics: bool = False,
) -> None:
    """
    Execute one demo episode and print a rich results report.

    Args:
        coder_version: Which coder strategy to use.
            "weak_coder_v1" | "weak_coder_v2" | "improving_coder"
    """
    _banner()

    ensure_runtime_dirs()
    memory = CoachMemory()
    memory.clear()          # Start fresh for the demo run
    env = FORGEEnv(memory=memory)
    policy = build_policy(policy_name, strategy=coder_version)
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
        candidates = policy.generate_candidates(state, num_candidates=candidates_per_step)
        candidate_solutions = [candidate.code for candidate in candidates if candidate.code.strip()]
        fallback_code = get_coder_code(coder_version, episode=episode)
        action = {
            "coder_code": candidate_solutions[0] if candidate_solutions else fallback_code,
            "candidate_solutions": candidate_solutions,
            "coder_version": coder_version,
        }

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
        ep_timeout_counts.append(cr.get("timeout_count", 0))
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
        rankings = info.get("candidate_rankings", [])
        if rankings:
            best = rankings[0]
            print(
                f"     Candidate ranking → count: {len(rankings)} | "
                f"selected_idx: {info.get('selected_candidate_index', -1)} | "
                f"best pass_rate: {best['pass_rate']:.2f} | "
                f"best runtime_ms: {best['avg_runtime_ms']:.2f}"
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

    write_episode_report(
        episode=episode,
        payload={
            "episode": episode,
            "coder_version": coder_version,
            "policy": policy.name,
            "avg_coder_reward": avg(ep_coder_rewards),
            "avg_breaker_reward": avg(ep_breaker_rewards),
            "avg_pass_rate": avg(ep_pass_rates),
            "avg_break_rate": avg(ep_break_rates),
            "total_fail_count": sum(ep_fail_counts),
            "total_error_count": sum(ep_error_counts),
            "total_timeout_count": sum(ep_timeout_counts),
            "steps": env.step_count,
        },
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
    if generate_metrics:
        from metrics import generate_charts
        chart_paths = generate_charts()
        if chart_paths:
            print("  Charts generated:")
            for key, path in chart_paths.items():
                print(f"    - {key}: {path}")
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
    print("  --policy NAME     Defender policy: heuristic | api | local | offline | model")
    print("  --candidates N    Candidate solutions to evaluate per step")
    print("  --charts          Generate trend charts in outputs/")
    print("  --benchmark N     Run benchmark mode for N episodes (minimum 20)")
    print("  --compare         Run baseline heuristic vs model policy comparison")
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
    policy_name = DEFAULT_POLICY
    candidates_per_step = DEFAULT_CANDIDATES_PER_STEP
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

    if "--policy" in args:
        idx = args.index("--policy")
        try:
            policy_name = args[idx + 1].strip().lower()
            if policy_name not in ("heuristic", "api", "local", "offline", "mock", "model"):
                raise ValueError(policy_name)
        except (IndexError, ValueError):
            print("Error: --policy must be one of: heuristic, api, local, offline, model.")
            sys.exit(1)

    if "--candidates" in args:
        idx = args.index("--candidates")
        try:
            candidates_per_step = max(1, int(args[idx + 1]))
        except (IndexError, ValueError):
            print("Error: --candidates requires an integer >= 1.")
            sys.exit(1)

    if "--compare" in args:
        report = run_compare_mode(
            model_policy_name="model",
            episodes=20,
            candidates_per_step=candidates_per_step,
            verbose=False,
        )
        print("Comparison complete")
        print(f"  Pass-rate delta      : {report['improvement']['pass_rate_delta']:+.4f}")
        print(f"  Defender reward delta: {report['improvement']['defender_reward_delta']:+.4f}")
        print(f"  Adversary reward delta: {report['improvement']['adversary_reward_delta']:+.4f}")
        print(f"  Tier Progression Delta: {report['improvement']['max_tier_delta']:+d}")
        print("  Judge assets exported to outputs/")
        sys.exit(0)

    if "--benchmark" in args:
        idx = args.index("--benchmark")
        try:
            benchmark_episodes = int(args[idx + 1])
        except (IndexError, ValueError):
            print("Error: --benchmark requires an integer argument.")
            sys.exit(1)

        report = run_benchmark_mode(
            policy_name=policy_name,
            episodes=benchmark_episodes,
            candidates_per_step=candidates_per_step,
            verbose=False,
        )
        print("Benchmark complete")
        print(f"  Episodes: {report['episodes']}")
        for row in report.get("rows", []):
            print(
                f"  Ep {row['episode']:>3} | pass={row['pass_rate']:.2f} "
                f"| defender={row['defender_reward']:+.2f} "
                f"| adversary={row['adversary_reward']:+.2f} "
                f"| rank={row['chosen_candidate_rank']} "
                f"| tier={row['tier_progression']}"
            )
        print("  Judge assets exported to outputs/")
        sys.exit(0)

    run_demo_episode(
        coder_version=coder_version,
        policy_name=policy_name,
        candidates_per_step=candidates_per_step,
        generate_metrics=("--charts" in args),
    )


if __name__ == "__main__":
    main()
