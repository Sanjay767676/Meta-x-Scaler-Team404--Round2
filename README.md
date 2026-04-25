---
title: FORGE-v4 Adversarial Robust Code Gen
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.0.0
python_version: 3.11
app_file: app.py
pinned: false
---

# FORGE-v4: Adversarial Robust Code Generation

[![Hackathon: Meta OpenEnv](https://img.shields.io/badge/Hackathon-Meta%20OpenEnv-blueviolet)](https://github.com/Meta-OpenEnv)
[![Demo: Hugging Face](https://img.shields.io/badge/Demo-Hugging%20Face-orange)](https://huggingface.co/spaces/sanjay7676/Team404_FORGE)

**FORGE-v4** (Framework for Objective Robustness & Generation Evaluation) is a high-fidelity adversarial benchmark environment designed to harden code-generation models against real-world edge cases, boundary values, and malicious logic traps.

## 🎯 Problem Statement
Large Language Models (LLMs) often excel at writing standard algorithms but fail silently when faced with adversarial inputs like negative values, extreme duplicates, or large-scale stress tests. In production, these "silent failures" lead to security vulnerabilities and system crashes.

**FORGE-v4** solves this by implementing an **Adversarial Red-Teaming loop** where a model's robustness is continuously challenged by an evolving Breaker agent.

## ⚔️ The Adversarial Loop
FORGE-v4 operates as a two-agent zero-sum game:
1.  **The Defender (Coder)**: Generates Python code to solve algorithmic tasks.
2.  **The Adversary (Breaker)**: Discovers and escalates adversarial test cases across 4 Tiers of difficulty.
3.  **CoachMemory**: A persistent feedback loop where the model "learns" from past failures to generate more robust solutions in subsequent episodes.

## 🚀 Benchmark Results
| Metric | Baseline (Heuristic) | FORGE Model | Delta |
| :--- | :--- | :--- | :--- |
| **Avg Pass Rate** | 91.00% | 100.00% | **+9.00%** |
| **Avg Reward** | 10.90 | 13.00 | **+2.10** |
| **Max Tier Reached** | Tier 4 | Tier 4 | **Sustained** |

### Visual Evidence
<div align="center">
  <img src="outputs/reward_curve.png" width="45%" />
  <img src="outputs/pass_rate.png" width="45%" />
</div>

## 🛠️ Technology Stack
- **Core Engine**: Python 3.11+
- **Sandbox**: Secure subprocess-based execution with resource limits.
- **API Standard**: OpenEnv compliant (FastAPI).
- **UI**: Gradio (Interactive Demo).
- **Optimization**: 10x speedup via batch evaluation and caching.

## 📡 API Endpoints
FORGE-v4 is fully programmable via its FastAPI server (`api_server.py`):
- `POST /reset`: Initialize a new adversarial episode.
- `POST /step`: Submit code candidates and receive rewards/diagnostics.
- `GET /state`: Retrieve current environment metrics and memory summary.

## 💻 Local Setup
```bash
# Clone the repo
git clone https://github.com/Sanjay767676/Meta-x-Scaler-Team404--Round2.git
cd FORGE

# Install dependencies
pip install -r requirements.txt

# Run the interactive demo
python app.py

# Run the training benchmark
python train_colab.py --compare --episodes 20
```

## 🧠 Authentic RL Training
FORGE-v4 generates real-world preference data for model alignment:

1.  **Generate Dataset**: Run a benchmark with a model policy. The environment evaluates multiple candidates and saves "Chosen" vs "Rejected" pairs into `data/dpo_dataset.jsonl`.
    ```bash
    python train_colab.py --compare --episodes 20
    ```
2.  **Fine-tune with Unsloth**: Apply the captured adversarial feedback using the integrated DPO trainer.
    ```bash
    bash setup_colab.sh
    bash run_dpo.sh
    ```

## 🗺️ Future Roadmap
- [ ] **Multi-Language Support**: Support for C++, Java, and Rust sandboxes.
- [ ] **Complex Task Generation**: Beyond sorting, into graph algorithms and dynamic programming.
- [x] **True LLM Finetuning**: Integrated with LoRA/Unsloth for RL-based weight optimization.
- [ ] **Web Dashboard**: Advanced analytics for multi-model comparisons.

---
*Developed by Team 404 for the Meta OpenEnv Hackathon.*
