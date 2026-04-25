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
CODER_PASS_REWARD   =  1.0           # Reward per passing hidden test
CODER_FAIL_PENALTY  = -0.5           # Penalty per failing hidden test
CODER_ERROR_PENALTY = -1.0           # Penalty when code raises an error/timeout

BREAKER_BREAK_REWARD = 1.0           # Reward when breaker's test breaks coder
BREAKER_FAIL_PENALTY = -0.3          # Penalty when coder survives a breaker test

# ──────────────────────────────────────────────
# Coder agent versions
# ──────────────────────────────────────────────
CODER_VERSIONS = ["weak_coder_v1", "weak_coder_v2", "improving_coder"]

# improving_coder tier-up thresholds (episode numbers)
IMPROVING_CODER_TIER1_UNTIL = 3      # Episodes 1–3 → uses weak strategy
IMPROVING_CODER_TIER2_UNTIL = 6      # Episodes 4–6 → uses mid strategy

# ──────────────────────────────────────────────
# Breaker tier system
# ──────────────────────────────────────────────
BREAKER_TIER_NAMES = {
    1: "Tier-1 (basic)",
    2: "Tier-2 (edge cases)",
    3: "Tier-3 (stress)",
    4: "Tier-4 (boundary/extreme)",
}

# Minimum break_rate to unlock next tier
BREAKER_TIER_UNLOCK_RATE  = 0.6      # 60% break rate needed to promote
# Minimum episode before tier 3 unlocks (regardless of break rate)
BREAKER_TIER3_MIN_EPISODE = 4
BREAKER_TIER4_MIN_EPISODE = 7

# ──────────────────────────────────────────────
# Tier thresholds (coder skill levels — for display/labelling)
# ──────────────────────────────────────────────
TIER_THRESHOLDS = {
    "novice":       (0.0,  0.4),
    "intermediate": (0.4,  0.7),
    "advanced":     (0.7,  0.9),
    "expert":       (0.9,  1.01),
}

# ──────────────────────────────────────────────
# Memory / logging
# ──────────────────────────────────────────────
MEMORY_FILE   = "data/coach_memory.json"
LOG_DIR       = "logs/"
MODELS_DIR    = "models/"
OUTPUTS_DIR   = "outputs/"

# Log file paths (within LOG_DIR)
LOG_REWARDS_FILE  = "logs/rewards.json"
LOG_EPISODES_FILE = "logs/episodes.csv"
LOG_SUMMARY_FILE  = "logs/summary.json"

# ──────────────────────────────────────────────
# Training placeholders
# ──────────────────────────────────────────────
MAX_EPISODES      = 100
STEPS_PER_EPISODE = 3                # Kept short for fast demo runs
