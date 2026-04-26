"""Central configuration for FORGE-v4.

This module intentionally keeps plain constants so hackathon iteration remains
fast and transparent. Secrets load only from environment variables or a local
`.env` file (never commit real keys).
"""

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

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

# LLM provider configuration (legacy + local policies)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")  # mock | openrouter | hf_api | huggingface_local | nim
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen2.5-coder-0.5b-instruct")
HF_LOCAL_MODEL_ID = os.getenv("HF_LOCAL_MODEL_ID", "qwen/qwen2.5-coder-0.5b-instruct")

# Router: Hugging Face custom (base + LoRA adapter from hub)
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
HF_ADAPTER_REPO = os.getenv("HF_MODEL_ID", os.getenv("HF_ADAPTER_REPO", "sanjay7676/forge-qwen-final"))

# NVIDIA NIM (OpenAI-compatible)
NIM_API_KEY = os.getenv("NIM_API_KEY", os.getenv("NVIDIA_API_KEY", ""))
NVIDIA_API_KEY = NIM_API_KEY
NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_MODEL = os.getenv("NIM_MODEL", "meta/llama-3.1-8b-instruct")

# OpenRouter
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen2.5-coder-7b-instruct")

HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("HUGGING_FACE_HUB_TOKEN", ""))

# Policy "model" routing: auto | custom_hf | nim | openrouter | mock
CODE_PROVIDER_MODE = os.getenv("CODE_PROVIDER_MODE", "auto")

# Per-provider HTTP / wrapped inference timeouts (seconds)
ROUTER_HF_TIMEOUT_SEC = float(os.getenv("ROUTER_HF_TIMEOUT_SEC", "360"))
ROUTER_NIM_TIMEOUT_SEC = float(os.getenv("ROUTER_NIM_TIMEOUT_SEC", "90"))
ROUTER_OPENROUTER_TIMEOUT_SEC = float(os.getenv("ROUTER_OPENROUTER_TIMEOUT_SEC", "90"))

# Dataset / evidence settings
DPO_DATASET_FILE = "data/dpo_dataset.jsonl"
MIN_DPO_PAIRS_TARGET = 480


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
