"""Training orchestration for FORGE-v4."""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from agents import coder_version_label, get_coder_code
from config import CHECKPOINT_FILE, DEFAULT_CANDIDATES_PER_STEP, MAX_EPISODES, STEPS_PER_EPISODE, ensure_runtime_dirs
from env import FORGEEnv
from logger import log_episode, update_summary, write_episode_report
from metrics.charts import export_judge_assets
from memory import CoachMemory
from policies.base import CoderPolicy
from policies.factory import build_policy


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
default_policy = build_policy("heuristic", strategy="improving_coder")


# ──────────────────────────────────────────────
# Core training loop
# ──────────────────────────────────────────────

def run_episode(
    env: FORGEEnv,
    coder_policy: Callable[[dict[str, Any]], dict[str, str]] | CoderPolicy,
    max_steps: int = STEPS_PER_EPISODE,
    candidates_per_step: int = DEFAULT_CANDIDATES_PER_STEP,
) -> dict[str, Any]:
    """Run one complete episode and return aggregated metrics."""
    state = env.reset()

    coder_version = "unknown"
    coder_rewards: list[float] = []
    breaker_rewards: list[float] = []
    pass_rates: list[float] = []
    break_rates: list[float] = []
    fail_counts: list[int] = []
    error_counts: list[int] = []
    timeout_counts: list[int] = []
    candidate_counts: list[int] = []
    chosen_candidate_ranks: list[int] = []

    for _ in range(max_steps):
        if hasattr(coder_policy, "generate_candidates"):
            generated = coder_policy.generate_candidates(state, num_candidates=candidates_per_step)  # type: ignore[attr-defined]
            candidate_solutions = [candidate.code for candidate in generated if candidate.code.strip()]
            action = {
                "coder_code": candidate_solutions[0] if candidate_solutions else "",
                "candidate_solutions": candidate_solutions,
                "coder_version": getattr(coder_policy, "name", "policy"),
            }
        else:
            action = coder_policy(state)  # type: ignore[operator]

        result = env.step(action)
        state = result["state"]

        coder_version = action.get("coder_version", coder_version)
        coder = result["coder_reward"]
        breaker = result["breaker_reward"]

        coder_rewards.append(coder["total_reward"])
        breaker_rewards.append(breaker["total_reward"])
        pass_rates.append(coder["pass_rate"])
        break_rates.append(breaker["break_rate"])
        fail_counts.append(coder["fail_count"])
        error_counts.append(coder["error_count"])
        timeout_counts.append(coder.get("timeout_count", 0))
        step_info = result.get("info", {})
        rankings = step_info.get("candidate_rankings", [])
        candidate_counts.append(len(rankings))
        selected_idx = step_info.get("selected_candidate_index", -1)
        rank_value = 1
        if rankings and selected_idx != -1:
            for rank_pos, ranked_item in enumerate(rankings, start=1):
                if ranked_item.get("index") == selected_idx:
                    rank_value = rank_pos
                    break
        chosen_candidate_ranks.append(rank_value)

        _on_step_end(env.step_count, result)

        if result["done"]:
            break

    def avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        "episode": env.episode,
        "coder_version": coder_version,
        "breaker_tier": env.breaker.current_tier,
        "avg_coder_reward": avg(coder_rewards),
        "avg_breaker_reward": avg(breaker_rewards),
        "avg_pass_rate": avg(pass_rates),
        "avg_break_rate": avg(break_rates),
        "total_fail_count": sum(fail_counts),
        "total_error_count": sum(error_counts),
        "total_timeout_count": sum(timeout_counts),
        "avg_candidates_evaluated": avg([float(c) for c in candidate_counts]),
        "avg_chosen_candidate_rank": avg([float(r) for r in chosen_candidate_ranks]),
        "steps": env.step_count,
        "coder_rewards": coder_rewards,
        "breaker_rewards": breaker_rewards,
        "pass_rates": pass_rates,
        "break_rates": break_rates,
    }


def train(
    coder_policy: Callable[[dict[str, Any]], dict[str, str]] | CoderPolicy = default_policy,
    num_episodes: int = MAX_EPISODES,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train defender policy against built-in adversary dynamics."""
    return train_defender(coder_policy=coder_policy, num_episodes=num_episodes, verbose=verbose)


def train_defender(
    coder_policy: Callable[[dict[str, Any]], dict[str, str]] | CoderPolicy = default_policy,
    num_episodes: int = MAX_EPISODES,
    verbose: bool = True,
    candidates_per_step: int = DEFAULT_CANDIDATES_PER_STEP,
) -> dict[str, Any]:
    """Run defender-focused training where environment controls breaker tiers."""
    ensure_runtime_dirs()
    memory = CoachMemory()
    env = FORGEEnv(memory=memory)
    episode_history: list[dict[str, Any]] = []

    all_coder_rewards: list[float] = []
    all_breaker_rewards: list[float] = []
    all_pass_rates: list[float] = []
    all_break_rates: list[float] = []

    for episode_idx in range(1, num_episodes + 1):
        episode_summary = run_episode(
            env=env,
            coder_policy=coder_policy,
            candidates_per_step=candidates_per_step,
        )
        episode_history.append(episode_summary)

        log_episode(
            episode=episode_summary["episode"],
            coder_version=episode_summary["coder_version"],
            breaker_tier=episode_summary["breaker_tier"],
            avg_coder_reward=episode_summary["avg_coder_reward"],
            avg_breaker_reward=episode_summary["avg_breaker_reward"],
            avg_pass_rate=episode_summary["avg_pass_rate"],
            total_fail_count=episode_summary["total_fail_count"],
            total_error_count=episode_summary["total_error_count"],
            total_timeout_count=episode_summary["total_timeout_count"],
            avg_break_rate=episode_summary["avg_break_rate"],
            steps=episode_summary["steps"],
        )

        all_coder_rewards.extend(episode_summary["coder_rewards"])
        all_breaker_rewards.extend(episode_summary["breaker_rewards"])
        all_pass_rates.extend(episode_summary["pass_rates"])
        all_break_rates.extend(episode_summary["break_rates"])

        if verbose:
            label = coder_version_label(episode_summary["coder_version"], episode_idx)
            print(
                f"  [Ep {episode_idx:>3}] Coder: {label:<50} "
                f"pass={episode_summary['avg_pass_rate']:.2f} "
                f"reward={episode_summary['avg_coder_reward']:+.2f} | "
                f"Breaker: {env.breaker.tier_name:<22} "
                f"break={episode_summary['avg_break_rate']:.2f} "
                f"reward={episode_summary['avg_breaker_reward']:+.2f} "
                f"| candidates={episode_summary['avg_candidates_evaluated']:.2f}"
            )

        write_episode_report(episode=episode_idx, payload=episode_summary)

        _on_episode_end(episode_idx, episode_summary, memory)

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
        "mode": "defender",
        "policy": getattr(coder_policy, "name", "callable"),
        "total_episodes": num_episodes,
        "episode_history": episode_history,
        "memory_summary": memory.summary(),
    }


def train_adversary(
    coder_policy: Callable[[dict[str, Any]], dict[str, str]] | CoderPolicy = weak_coder_v1_policy,
    num_episodes: int = MAX_EPISODES,
    verbose: bool = True,
    candidates_per_step: int = DEFAULT_CANDIDATES_PER_STEP,
) -> dict[str, Any]:
    """Adversary-focused routine using a weaker/default defender baseline."""
    return train_defender(
        coder_policy=coder_policy,
        num_episodes=num_episodes,
        verbose=verbose,
        candidates_per_step=candidates_per_step,
    )


def train_with_policy_name(
    policy_name: str,
    num_episodes: int = MAX_EPISODES,
    verbose: bool = True,
    candidates_per_step: int = DEFAULT_CANDIDATES_PER_STEP,
) -> dict[str, Any]:
    """Convenience helper for selecting a policy by name."""
    policy = build_policy(policy_name)
    return train_defender(
        coder_policy=policy,
        num_episodes=num_episodes,
        verbose=verbose,
        candidates_per_step=candidates_per_step,
    )


def run_benchmark_mode(
    policy_name: str,
    episodes: int = 20,
    candidates_per_step: int = DEFAULT_CANDIDATES_PER_STEP,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run evidence benchmark and export judge assets."""
    episode_count = max(20, episodes)
    summary = train_with_policy_name(
        policy_name=policy_name,
        num_episodes=episode_count,
        verbose=verbose,
        candidates_per_step=candidates_per_step,
    )

    rows: list[dict[str, Any]] = []
    for item in summary.get("episode_history", []):
        rows.append(
            {
                "episode": item.get("episode"),
                "pass_rate": item.get("avg_pass_rate", 0.0),
                "defender_reward": item.get("avg_coder_reward", 0.0),
                "adversary_reward": item.get("avg_breaker_reward", 0.0),
                "chosen_candidate_rank": item.get("avg_chosen_candidate_rank", 1.0),
                "tier_progression": item.get("breaker_tier", 0),
            }
        )

    final_report = {
        "mode": "benchmark",
        "policy": policy_name,
        "episodes": episode_count,
        "rows": rows,
        "summary": {
            "avg_pass_rate": round(sum(r["pass_rate"] for r in rows) / len(rows), 4) if rows else 0.0,
            "avg_defender_reward": round(sum(r["defender_reward"] for r in rows) / len(rows), 4) if rows else 0.0,
            "avg_adversary_reward": round(sum(r["adversary_reward"] for r in rows) / len(rows), 4) if rows else 0.0,
            "max_tier": max((r["tier_progression"] for r in rows), default=0),
        },
    }
    assets = export_judge_assets(episodes=rows, final_report=final_report)
    final_report["assets"] = assets
    export_judge_assets(episodes=rows, final_report=final_report)
    return final_report


def run_compare_mode(
    model_policy_name: str = "model",
    episodes: int = 20,
    candidates_per_step: int = DEFAULT_CANDIDATES_PER_STEP,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run baseline heuristic vs model policy comparison with improvement metrics."""
    baseline = run_benchmark_mode(
        policy_name="heuristic",
        episodes=episodes,
        candidates_per_step=candidates_per_step,
        verbose=verbose,
    )
    model = run_benchmark_mode(
        policy_name=model_policy_name,
        episodes=episodes,
        candidates_per_step=candidates_per_step,
        verbose=verbose,
    )

    baseline_summary = baseline.get("summary", {})
    model_summary = model.get("summary", {})

    comparison = {
        "mode": "compare",
        "baseline_policy": "heuristic",
        "model_policy": model_policy_name,
        "episodes": max(20, episodes),
        "baseline": baseline_summary,
        "model": model_summary,
        "improvement": {
            "pass_rate_delta": round(model_summary.get("avg_pass_rate", 0.0) - baseline_summary.get("avg_pass_rate", 0.0), 4),
            "defender_reward_delta": round(
                model_summary.get("avg_defender_reward", 0.0) - baseline_summary.get("avg_defender_reward", 0.0),
                4,
            ),
            "adversary_reward_delta": round(
                model_summary.get("avg_adversary_reward", 0.0) - baseline_summary.get("avg_adversary_reward", 0.0),
                4,
            ),
        },
    }

    export_judge_assets(episodes=model.get("rows", []), final_report=comparison)
    return comparison


def save_checkpoint(path: str = CHECKPOINT_FILE, payload: dict[str, Any] | None = None) -> str:
    """Persist lightweight training state for resume workflows."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = payload or {}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    return path


def load_checkpoint(path: str = CHECKPOINT_FILE) -> dict[str, Any]:
    """Load checkpoint payload if available, otherwise return empty state."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return loaded if isinstance(loaded, dict) else {}


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
