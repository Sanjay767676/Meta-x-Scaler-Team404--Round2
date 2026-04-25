# trainer.py
# Training loop for FORGE-v4.
# Uses the real coder strategies and tiered BreakerAgent from agents.py.
# Hook placeholders are ready for TRL / Unsloth / Hugging Face integration.

from typing import Any, Callable
from env import FORGEEnv
from memory import CoachMemory
from agents import get_coder_code, coder_version_label
from logger import log_episode, update_summary
from config import MAX_EPISODES, STEPS_PER_EPISODE


# ──────────────────────────────────────────────
# Built-in coder policies
# ──────────────────────────────────────────────

def make_coder_policy(version: str) -> Callable[[dict[str, Any]], dict[str, str]]:
    """
    Factory: return a coder policy function for the given version name.

    The returned callable takes a state dict and returns an action dict:
        {"coder_code": str, "coder_version": str}

    Args:
        version: "weak_coder_v1" | "weak_coder_v2" | "improving_coder"
    """
    def policy(state: dict[str, Any]) -> dict[str, str]:
        episode = state.get("episode", 1)
        code    = get_coder_code(version, episode=episode)
        return {"coder_code": code, "coder_version": version}
    return policy


# Convenience pre-built policies
weak_coder_v1_policy    = make_coder_policy("weak_coder_v1")
weak_coder_v2_policy    = make_coder_policy("weak_coder_v2")
improving_coder_policy  = make_coder_policy("improving_coder")

# Default used by app.py
default_coder_policy    = improving_coder_policy


# ──────────────────────────────────────────────
# Core training loop
# ──────────────────────────────────────────────

def train(
    coder_policy: Callable[[dict[str, Any]], dict[str, str]] = default_coder_policy,
    num_episodes: int = MAX_EPISODES,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run the FORGE-v4 training loop.

    The BreakerAgent is managed by the environment — it automatically tiers up
    based on performance. Only the coder policy needs to be supplied here.

    Args:
        coder_policy:  Callable(state) → {"coder_code": str, "coder_version": str}
        num_episodes:  Number of episodes to run.
        verbose:       Print per-episode summaries when True.

    Returns:
        Training summary dict.
    """
    memory = CoachMemory()
    env    = FORGEEnv(memory=memory)

    episode_history: list[dict[str, Any]] = []

    # Aggregate accumulators for final summary
    all_coder_rewards:   list[float] = []
    all_breaker_rewards: list[float] = []
    all_pass_rates:      list[float] = []
    all_break_rates:     list[float] = []

    for ep in range(1, num_episodes + 1):
        state = env.reset()

        ep_coder_rewards:   list[float] = []
        ep_breaker_rewards: list[float] = []
        ep_pass_rates:      list[float] = []
        ep_fail_counts:     list[int]   = []
        ep_error_counts:    list[int]   = []
        ep_timeout_counts:  list[int]   = []
        ep_break_rates:     list[float] = []

        for _ in range(STEPS_PER_EPISODE):
            action = coder_policy(state)
            result = env.step(action)
            state  = result["state"]

            cr = result["coder_reward"]
            br = result["breaker_reward"]

            ep_coder_rewards.append(cr["total_reward"])
            ep_breaker_rewards.append(br["total_reward"])
            ep_pass_rates.append(cr["pass_rate"])
            ep_fail_counts.append(cr["fail_count"])
            ep_error_counts.append(cr["error_count"])
            ep_timeout_counts.append(cr["error_count"])
            ep_break_rates.append(br["break_rate"])

            if result["done"]:
                break

        # ── Episode summary ────────────────────────────────────────────────
        def avg(lst: list) -> float:
            return round(sum(lst) / len(lst), 4) if lst else 0.0

        ep_summary = {
            "episode":              ep,
            "coder_version":        action.get("coder_version", "unknown"),
            "breaker_tier":         env.breaker.current_tier,
            "avg_coder_reward":     avg(ep_coder_rewards),
            "avg_breaker_reward":   avg(ep_breaker_rewards),
            "avg_pass_rate":        avg(ep_pass_rates),
            "avg_break_rate":       avg(ep_break_rates),
            "steps":                env.step_count,
        }
        episode_history.append(ep_summary)

        # ── Log episode to CSV ─────────────────────────────────────────────
        log_episode(
            episode=ep,
            coder_version=ep_summary["coder_version"],
            breaker_tier=ep_summary["breaker_tier"],
            avg_coder_reward=ep_summary["avg_coder_reward"],
            avg_breaker_reward=ep_summary["avg_breaker_reward"],
            avg_pass_rate=ep_summary["avg_pass_rate"],
            total_fail_count=sum(ep_fail_counts),
            total_error_count=sum(ep_error_counts),
            total_timeout_count=sum(ep_timeout_counts),
            avg_break_rate=ep_summary["avg_break_rate"],
            steps=ep_summary["steps"],
        )

        # ── Accumulate for final summary ───────────────────────────────────
        all_coder_rewards.extend(ep_coder_rewards)
        all_breaker_rewards.extend(ep_breaker_rewards)
        all_pass_rates.extend(ep_pass_rates)
        all_break_rates.extend(ep_break_rates)

        if verbose:
            label = coder_version_label(ep_summary["coder_version"], ep)
            print(
                f"  [Ep {ep:>3}]  Coder: {label:<50}  "
                f"pass={ep_summary['avg_pass_rate']:.2f}  "
                f"reward={ep_summary['avg_coder_reward']:+.2f}  |  "
                f"Breaker: {env.breaker.tier_name:<22}  "
                f"break={ep_summary['avg_break_rate']:.2f}  "
                f"reward={ep_summary['avg_breaker_reward']:+.2f}"
            )

        # ── TRL / Unsloth hook ─────────────────────────────────────────────
        _on_episode_end(ep, ep_summary, memory)

    # ── Final summary JSON ────────────────────────────────────────────────
    update_summary(
        total_episodes=num_episodes,
        coder_version=episode_history[-1]["coder_version"] if episode_history else "unknown",
        final_breaker_tier=env.breaker.current_tier,
        all_coder_rewards=all_coder_rewards,
        all_breaker_rewards=all_breaker_rewards,
        all_pass_rates=all_pass_rates,
        all_break_rates=all_break_rates,
        coach_memory_summary=memory.summary(),
    )

    return {
        "total_episodes":  num_episodes,
        "episode_history": episode_history,
        "memory_summary":  memory.summary(),
    }


# ──────────────────────────────────────────────
# Hook placeholders for future RL framework integration
# ──────────────────────────────────────────────

def _on_episode_end(
    episode: int,
    summary: dict[str, Any],
    memory: CoachMemory,
) -> None:
    """
    Called at end of every episode.

    TODO: Plug in TRL PPOTrainer / Unsloth model updates here.
    E.g.:
        trainer.step(queries, responses, rewards)
        model.save_pretrained(f"models/checkpoint-ep{episode}")
    """
    pass


def _on_step_end(step: int, result: dict[str, Any]) -> None:
    """
    Called after every environment step.

    TODO: Plug in per-step reward logging (W&B, TensorBoard) here.
    """
    pass
