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

# FORGE-v4

**OpenEnv-style RL environment for adversarial code generation** — a Defender writes Python; a Breaker escalates tiered stress tests; rewards and memory come from real sandbox runs, not vibes.

[![Hugging Face Space](https://img.shields.io/badge/Live-Demo-yellow)](https://huggingface.co/spaces/sanjay7676/Team404_FORGE)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-black)](https://github.com/Sanjay767676/Meta-x-Scaler-Team404--Round2)
[![Training Notebook](https://img.shields.io/badge/Colab-Notebook-orange)](https://colab.research.google.com/github/Sanjay767676/Meta-x-Scaler-Team404--Round2/blob/main/FORGE_Training_Colab.ipynb)
[![Hackathon](https://img.shields.io/badge/Hackathon-Meta%20OpenEnv-0a66c2)](https://openenv.devpost.com)

---

## Why this exists (the story in one breath)

Benchmarks love the happy path. Production does not. FORGE-v4 is a small arena where **pretty answers lose to executed answers**: the Breaker hunts negatives, duplicates, bad shapes, and large inputs while the Defender is scored by verifiers and timeouts. **CoachMemory** keeps a structured trace of *how* things failed, so later episodes are not amnesiac rerolls. That loop is the difference between a static test file and something that behaves like **self-improvement under pressure**.

---

## Rubric fit (what judges are looking for)

| Track | How FORGE-v4 answers it |
| :-- | :-- |
| **Self-improvement (primary)** | Persistent **CoachMemory** (`memory.py`), Breaker **curriculum tiers**, trajectory export to **`data/dpo_dataset.jsonl`**, optional **Unsloth + TRL** post-training (`train_unsloth.py`). |
| **Multi-agent (secondary)** | Explicit **Defender vs Breaker** loop each step (`agents.py`, `rewards.py`), not a single monolithic policy. |
| **OpenEnv / environment** | **`openenv.yaml`** manifest; **`FORGEEnv`** in `env.py` with `reset`, `step`, **`get_state`**; **FastAPI** mirror at **`POST /reset`**, **`POST /step`**, **`GET /state`**, **`GET /health`** (`api_server.py`). |

**Novelty in one line:** we combine **adversarial tier escalation + executable verification + memory-backed feedback + a training export path** in one reproducible repo, instead of a leaderboard snapshot or a chat-only demo.

---

## Tech stack (concrete)

| Layer | Choices |
| :-- | :-- |
| **Language / runtime** | Python **3.11** |
| **UI / hosting** | **Gradio** (`app.py`) on **Hugging Face Spaces** |
| **API** | **FastAPI** + **Uvicorn** (`api_server.py`, port **8000**) |
| **Env spec** | **`openenv.yaml`** (version, agents, routes, artifact paths) |
| **Execution** | **Sandbox** with timeouts and caps (`sandbox.py`, `config.py`, `services/`) |
| **Training** | **Unsloth**, **TRL**, **DPO** / **GRPO** modes, **LoRA** on **Qwen2.5-Coder** (4-bit default in trainer) |
| **Repro** | **`FORGE_Training_Colab.ipynb`**, **`train_colab.py`**, **`setup_colab.sh`** (Colab GPU stack) |
| **Inference hooks** | **`policies/`** — heuristic, API, local HF, **NIM-ready** via env (`config.py`) |

---

## Evidence (numbers you can re-run)

After `python train_colab.py --compare --episodes 20`, inspect **`outputs/final_report.json`** and the plots below. The table is a **sample** from one local compare (your run may differ; the pipeline is deterministic given seed and config).

| Metric | Baseline (heuristic) | Model policy | Delta |
| :-- | --: | --: | --: |
| Avg pass rate | 91.00% | 100.00% | +9.00% |
| Avg Defender reward | 10.90 | 13.00 | +2.10 |
| Max Breaker tier reached | 4 | 4 | — |

<div align="center">
  <img src="outputs/reward_curve.png" alt="Reward curve" width="32%" />
  <img src="outputs/pass_rate.png" alt="Pass rate" width="32%" />
  <img src="outputs/loss_curve.png" alt="Training loss" width="32%" />
</div>

---

## Reproduce (fast paths)

**Colab (recommended):** [open the notebook](https://colab.research.google.com/github/Sanjay767676/Meta-x-Scaler-Team404--Round2/blob/main/FORGE_Training_Colab.ipynb) → enable **GPU** if you want real Unsloth imports → run cells top to bottom (clone → `pip` → compare → train → checks → plots).

**Local:**

```bash
git clone https://github.com/Sanjay767676/Meta-x-Scaler-Team404--Round2.git
cd Meta-x-Scaler-Team404--Round2
python -m venv .venv && .venv\Scripts\activate   # Windows; use source .venv/bin/activate on Unix
pip install -r requirements.txt
python app.py                                    # Gradio UI
python api_server.py                             # OpenEnv API :8000
python train_colab.py --compare --episodes 20
python train_unsloth.py --mode dpo             # or --mode grpo
```

**Training fidelity:** `train_unsloth.py` runs full Unsloth/TRL when those libs and a **GPU** are available; otherwise it can fall back to a **simulation path** that still writes adapter metadata and **`outputs/TRAINING_REPORT.md`** so the repo stays runnable on CPU-only machines. For hackathon evidence of real fine-tuning, prefer Colab GPU + `setup_colab.sh`.

---

## Roadmap (what we will do next)

- **Multi-task robustness** beyond the current controlled benchmark framing in `openenv.yaml`
- **Security-style verifier loops** and richer exploit-shaped failures
- **Long-horizon coding agents** (edit sequences, not single-shot snippets)
- **Stronger isolation** for public deployment (containers on top of the current sandbox)

---

## Team

**Team404** · Meta OpenEnv Hackathon

---

## Judge checklist (links + artifacts)

| | |
| :-- | :-- |
| **Live demo** | [Hugging Face Space](https://huggingface.co/spaces/sanjay7676/Team404_FORGE) |
| **Source** | [GitHub](https://github.com/Sanjay767676/Meta-x-Scaler-Team404--Round2) |
| **Colab** | [FORGE_Training_Colab.ipynb](https://colab.research.google.com/github/Sanjay767676/Meta-x-Scaler-Team404--Round2/blob/main/FORGE_Training_Colab.ipynb) |
| **Story / blog** | [MINI_BLOG.md](MINI_BLOG.md) |
| **Commands & security** | [guide.md](guide.md) |
| **Key outputs** | `outputs/final_report.json`, `outputs/*.png`, `logs/summary.json`, `logs/episodes.csv`, `data/dpo_dataset.jsonl` |

**Scope (honest):** Python-first benchmark; GPU recommended for authentic Unsloth training; sandbox is hackathon-grade—harden further for untrusted code in production.
