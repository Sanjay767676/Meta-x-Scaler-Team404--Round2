# config.py
# Central configuration constants for FORGE-v4

# ──────────────────────────────────────────────
# Sandbox settings
# ──────────────────────────────────────────────
SANDBOX_TIMEOUT_SECONDS = 5          # Max time allowed for code execution
SANDBOX_MAX_OUTPUT_CHARS = 4096      # Truncate stdout/stderr beyond this length

# ──────────────────────────────────────────────
# Task / environment settings
# ──────────────────────────────────────────────
MAX_ARRAY_SIZE = 20                  # Max length of generated integer arrays
MIN_ARRAY_SIZE = 3                   # Min length of generated integer arrays
ARRAY_VALUE_RANGE = (-100, 100)      # (min, max) integers in generated arrays
NUM_HIDDEN_TESTS = 5                 # Number of hidden test cases per task

# ──────────────────────────────────────────────
# Reward settings
# ──────────────────────────────────────────────
# Coder reward weights
CODER_PASS_REWARD = 1.0              # Reward per passing hidden test
CODER_FAIL_PENALTY = -0.5            # Penalty per failing hidden test
CODER_ERROR_PENALTY = -1.0           # Penalty when code raises an error

# Breaker reward weights
BREAKER_BREAK_REWARD = 1.0           # Reward when breaker's test breaks coder
BREAKER_FAIL_PENALTY = -0.3          # Penalty when breaker's test does NOT break coder

# ──────────────────────────────────────────────
# Tier thresholds (coder skill levels)
# ──────────────────────────────────────────────
TIER_THRESHOLDS = {
    "novice":       (0.0,  0.4),     # pass-rate range [low, high)
    "intermediate": (0.4,  0.7),
    "advanced":     (0.7,  0.9),
    "expert":       (0.9,  1.01),
}

# ──────────────────────────────────────────────
# Memory / logging
# ──────────────────────────────────────────────
MEMORY_FILE = "data/coach_memory.json"   # Persistent memory path
LOG_DIR = "logs/"                        # Directory for episode logs
MODELS_DIR = "models/"                   # Saved model checkpoints
OUTPUTS_DIR = "outputs/"                 # Generated code outputs

# ──────────────────────────────────────────────
# Training placeholders
# ──────────────────────────────────────────────
MAX_EPISODES = 100                   # Default training episode count
STEPS_PER_EPISODE = 10               # Steps per episode
