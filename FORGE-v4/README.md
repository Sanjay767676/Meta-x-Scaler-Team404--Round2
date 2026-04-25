# FORGE-v4

**Adversarial Code Generation Environment for Reinforcement Learning**

A hackathon project built on an **OpenEnv-style** reinforcement learning framework where two competing agents — a Coder and a Breaker — are trained adversarially on Python sorting tasks.

---

## Overview

FORGE-v4 pits two agents against each other:

| Agent | Role |
|-------|------|
| **Coder** | Writes Python code to solve integer array sorting tasks |
| **Breaker** | Generates adversarial test cases to expose flaws in the Coder's solution |

Each episode the Coder earns rewards for passing hidden tests; the Breaker earns rewards for breaking the Coder's solution. A **Coach Memory** module accumulates lessons learned across episodes to guide future training.

The skeleton is designed to be **drop-in ready for TRL / Unsloth fine-tuning** and **Hugging Face deployment**.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   FORGEEnv (env.py)              │
│                                                  │
│  ┌──────────────┐        ┌──────────────────┐   │
│  │  Coder Agent  │        │  Breaker Agent    │   │
│  │  (policy fn) │        │  (policy fn)      │   │
│  └──────┬───────┘        └────────┬──────────┘   │
│         │ code (str)              │ test cases    │
│         ▼                         ▼               │
│  ┌──────────────────────────────────────────┐    │
│  │           Sandbox (sandbox.py)           │    │
│  │  subprocess · timeout · pass/fail/error  │    │
│  └──────────────────┬───────────────────────┘    │
│                     │ results                     │
│                     ▼                             │
│  ┌──────────────────────────────────────────┐    │
│  │         Rewards (rewards.py)             │    │
│  │  coder_reward() · breaker_reward()       │    │
│  └──────────────────┬───────────────────────┘    │
│                     │                             │
│                     ▼                             │
│  ┌──────────────────────────────────────────┐    │
│  │       Coach Memory (memory.py)           │    │
│  │  JSON-backed · lessons · summary()       │    │
│  └──────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

---

## File Structure

```
FORGE-v4/
├── app.py           # CLI entry point — runs one demo episode
├── env.py           # FORGEEnv: reset() / step() / get_state()
├── tasks.py         # Task generator + hidden test sampler
├── rewards.py       # coder_reward() and breaker_reward()
├── sandbox.py       # Safe subprocess code execution with timeout
├── memory.py        # CoachMemory: JSON-backed lessons store
├── trainer.py       # Training loop + TRL/Unsloth hook placeholders
├── config.py        # All constants (timeout, rewards, tier thresholds)
├── requirements.txt # Dependencies
├── README.md        # This file
├── data/            # coach_memory.json (auto-created)
├── logs/            # Episode logs
├── models/          # Saved model checkpoints
└── outputs/         # Generated code outputs
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The core skeleton has minimal dependencies. ML packages (TRL, Unsloth, PyTorch) are commented out in `requirements.txt` — uncomment them when adding LLM training.

### 2. Run a demo episode

```bash
python app.py
```

This runs a single episode with placeholder Coder and Breaker policies (the Coder always uses `sorted()`, the Breaker sends fixed edge cases). You should see per-step reward output and a coach memory summary.

### 3. Optional: override step count

```bash
python app.py --steps 3
```

---

## Configuration

Edit `config.py` to adjust environment constants:

| Constant | Default | Description |
|----------|---------|-------------|
| `SANDBOX_TIMEOUT_SECONDS` | `5` | Max execution time per code run |
| `MAX_ARRAY_SIZE` | `20` | Largest generated array |
| `NUM_HIDDEN_TESTS` | `5` | Hidden test cases per task |
| `CODER_PASS_REWARD` | `1.0` | Reward per passing test |
| `BREAKER_BREAK_REWARD` | `1.0` | Reward per test that breaks coder |
| `MAX_EPISODES` | `100` | Default training episode count |

---

## Extending with LLM Agents

Replace the placeholder policies in `trainer.py`:

```python
# trainer.py
def my_coder_policy(state: dict) -> str:
    prompt = state["task_prompt"]
    # Call your LLM here (TRL model, OpenAI API, Unsloth, etc.)
    return generated_code

def my_breaker_policy(state: dict) -> list[dict]:
    prompt = state["task_prompt"]
    # Call your adversarial LLM here
    return [{"input": arr} for arr in generated_arrays]
```

Then run:

```python
from trainer import train
summary = train(
    coder_policy=my_coder_policy,
    breaker_policy=my_breaker_policy,
    num_episodes=50,
)
```

---

## TRL / Unsloth Integration (Future)

Hook points are prepared in `trainer.py`:

- `_on_episode_end()` — plug in `PPOTrainer.step()` or `GRPOTrainer` updates
- `_on_step_end()` — plug in per-step reward logging (W&B, TensorBoard)

```python
# Example (uncomment in trainer.py after installing TRL):
# from trl import PPOTrainer, PPOConfig
# trainer = PPOTrainer(config=PPOConfig(...), model=model, ...)
# trainer.step(queries, responses, rewards)
```

---

## Google Colab

1. Clone or upload the project to Colab.
2. Install Unsloth:
   ```
   !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   ```
3. Mount Drive and set `MEMORY_FILE` / `MODELS_DIR` in `config.py` to paths under `/content/drive/MyDrive/`.
4. Run `python app.py` or import and call `train()` directly.

---

## Hugging Face Deployment

After training, push your model with:

```python
model.push_to_hub("your-username/forge-v4-coder")
tokenizer.push_to_hub("your-username/forge-v4-coder")
```

The repo structure (`models/`, `outputs/`) maps directly to HF Hub conventions.

---

## License

MIT
