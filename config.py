"""Central configuration for FORGE-v4.

This module intentionally keeps plain constants so hackathon iteration remains
fast and transparent.
"""

import os

# Sandbox settings
SANDBOX_TIMEOUT_SECONDS = 5
SANDBOX_MAX_OUTPUT_CHARS = 4096
SANDBOX_MAX_CODE_CHARS = 20000
SANDBOX_MEMORY_LIMIT_MB = 256

# Task and environment settings
MAX_ARRAY_SIZE = 20
MIN_ARRAY_SIZE = 3
ARRAY_VALUE_RANGE = (-100, 100)
NUM_HIDDEN_TESTS = 5

# Reward settings
CODER_PASS_REWARD = 1.0
CODER_FAIL_PENALTY = -0.5
CODER_ERROR_PENALTY = -0.9
CODER_TIMEOUT_PENALTY = -1.2
CODER_PERFECT_RUN_BONUS = 0.5

BREAKER_BREAK_REWARD = 1.0
BREAKER_ERROR_BREAK_BONUS = 0.2
BREAKER_TIMEOUT_BREAK_BONUS = 0.4
BREAKER_FAIL_PENALTY = -0.3

# Coder agent versions
CODER_VERSIONS = ["weak_coder_v1", "weak_coder_v2", "improving_coder"]
IMPROVING_CODER_TIER1_UNTIL = 3
IMPROVING_CODER_TIER2_UNTIL = 6

# Breaker tier system
BREAKER_TIER_NAMES = {
    1: "Tier-1 (basic)",
    2: "Tier-2 (edge cases)",
    3: "Tier-3 (stress)",
    4: "Tier-4 (boundary/extreme)",
}
BREAKER_TIER_UNLOCK_RATE = 0.1  # Fast unlock
BREAKER_TIER3_MIN_EPISODE = 1   # Available immediately if breaker performs
BREAKER_TIER4_MIN_EPISODE = 1   # Available immediately if breaker performs

# Tier thresholds for display
TIER_THRESHOLDS = {
    "novice": (0.0, 0.4),
    "intermediate": (0.4, 0.7),
    "advanced": (0.7, 0.9),
    "expert": (0.9, 1.01),
}

# Paths and persistence
MEMORY_FILE = "data/coach_memory.json"
MEMORY_MAX_LESSONS = 1000
LOG_DIR = "logs"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
CHECKPOINT_FILE = "models/forge_checkpoint.json"
ADAPTERS_DIR = "models/adapters"
CHECKPOINTS_DIR = "models/checkpoints"
REWARD_GRAPHS_DIR = "outputs/reward_graphs"
RUN_ARTIFACTS_DIR = "outputs/run_artifacts"

LOG_REWARDS_FILE = "logs/rewards.json"
LOG_EPISODES_FILE = "logs/episodes.csv"
LOG_SUMMARY_FILE = "logs/summary.json"

# Training defaults
MAX_EPISODES = 100
STEPS_PER_EPISODE = 3
DEFAULT_CANDIDATES_PER_STEP = 3
GLOBAL_RANDOM_SEED = 42

# LLM provider configuration (future-ready)
LLM_PROVIDER = "nim"  # mock | openrouter | hf_api | huggingface_local | nim
LLM_MODEL = "qwen/qwen2.5-coder-0.5b-instruct"
HF_LOCAL_MODEL_ID = "qwen/qwen2.5-coder-0.5b-instruct"
NIM_MODEL = "meta/llama-3.1-405b-instruct"  # High-performance NIM choice
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = "" # Or set OPENROUTER_API_KEY env var
HF_TOKEN = ""           # Or set HF_TOKEN env var
NVIDIA_API_KEY = "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"


def ensure_runtime_dirs() -> None:
    """Create runtime directories required for logs, checkpoints, and artifacts."""
    paths = [
        LOG_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        ADAPTERS_DIR,
        CHECKPOINTS_DIR,
        REWARD_GRAPHS_DIR,
        RUN_ARTIFACTS_DIR,
        os.path.dirname(MEMORY_FILE) or ".",
    ]
    for path in paths:
        os.makedirs(path, exist_ok=True)
