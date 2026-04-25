# FORGE-v4

Adversarial code-generation environment for hackathons.

FORGE-v4 simulates a two-agent loop:
- Defender (Coder) submits Python sorting code.
- Adversary (Breaker) generates valid adversarial tests.
- Sandbox executes code safely with timeout limits.
- Rewards, memory, logs, and charts track learning progress.

## Why This Project

This repo is designed for a practical hackathon flow:
- Runs locally with no API keys required.
- Uses modular Python files that can be upgraded fast.
- Supports real LLM provider switching (OpenRouter, Hugging Face local, mock fallback).
- Produces evidence artifacts: logs and trend charts.

## Competitive Phase Features

- Real model provider integration:
	- Hugging Face local model provider
	- OpenRouter API provider
	- Mock fallback provider
- Best-candidate ranking from multi-sample generation per step
- Benchmark evidence mode (20+ episodes)
- Before-vs-after compare mode (baseline heuristic vs model policy)
- Judge asset export:
	- outputs/reward_curve.png
	- outputs/pass_rate.png
	- outputs/final_report.json

## Controlled Benchmark Story

FORGE-v4 intentionally uses integer sorting as a controlled adversarial benchmark.
This narrow domain is a feature, not a limitation: it provides a measurable,
repeatable environment for robust code-generation research before expanding to
broader programming tasks.

## Current Architecture

- app.py: CLI runner for demo episodes and chart generation
- env.py: OpenEnv-style environment (`reset`, `step`, `get_state`)
- tasks.py: sorting task and hidden test generation
- sandbox.py: guarded subprocess execution and structured result parsing
- rewards.py: realistic defender/adversary reward shaping
- memory.py: persistent JSON memory with weighted lessons
- trainer.py: episode loop, train entrypoints, checkpoint helpers
- logger.py: JSON/CSV/summary logging
- metrics/: trend chart generation into outputs/reward_graphs
- llm_agent.py: provider-ready abstraction (OpenRouter, Hugging Face local, mock)
- policies/: pluggable defender policies (heuristic, api, local, mock, model)
- services/: candidate ranking/evaluation services
- storage/: run artifact and checkpoint storage helpers
- openenv.yaml: OpenEnv metadata profile

## Quickstart

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Run a demo episode

```bash
python app.py
```

3. Run with custom options

```bash
python app.py --policy heuristic --coder improving_coder --candidates 4 --steps 5 --charts
```

## CLI Options

- `--coder VERSION`: `weak_coder_v1` | `weak_coder_v2` | `improving_coder`
- `--policy NAME`: `heuristic` | `api` | `local` | `mock` | `model`
- `--candidates N`: number of candidate solutions evaluated per step
- `--steps N`: override episode steps for this run
- `--charts`: render trend charts into `outputs/`
- `--benchmark N`: run benchmark mode for N episodes (minimum 20)
- `--compare`: run baseline heuristic vs model policy comparison
- `--help`: print usage

Benchmark examples:

```bash
python app.py --policy model --benchmark 25
python app.py --compare
```

## Training API

Use `trainer.py` for programmatic training:

```python
from trainer import train_defender, train_adversary, save_checkpoint

summary = train_defender(num_episodes=20, verbose=True)
save_checkpoint(payload=summary)
```

Key trainer functions:
- `run_episode(env, coder_policy, max_steps)`
- `train_defender(coder_policy, num_episodes, verbose)`
- `train_adversary(coder_policy, num_episodes, verbose)`
- `train_with_policy_name(policy_name, ...)`
- `save_checkpoint(path, payload)`
- `load_checkpoint(path)`

## Logs and Artifacts

FORGE writes:
- `logs/rewards.json`: step-level metrics
- `logs/episodes.csv`: episode rollups
- `logs/summary.json`: training summary snapshot
- `data/coach_memory.json`: persistent lessons
- `outputs/reward_graphs/*.png`: trend charts
- `outputs/run_artifacts/*.json`: deterministic episode reports
- `models/checkpoints/*.json`: checkpoints
- `models/adapters/`: adapter-ready folder for PEFT/LoRA exports

## LLM Provider Readiness

`llm_agent.py` plus `policies/` provide production-ready structure and safe defaults:
- `mock`: deterministic no-key fallback
- `openrouter`: live API integration when OPENROUTER_API_KEY is set
- `huggingface_local`: local model inference with fallback to mock provider

Policy classes are pluggable through interfaces/classes:
- `HeuristicPolicy`
- `APIModelPolicy`
- `LocalModelPolicy`

Set provider-related flags in `config.py`:
- `LLM_PROVIDER`
- `LLM_MODEL`
- `HF_LOCAL_MODEL_ID`
- `OPENROUTER_*`

Default lightweight coding model profile:
- qwen/qwen2.5-coder-0.5b-instruct

## Security Notes

Sandbox currently uses isolated Python subprocess execution with:
- strict timeout enforcement
- structured payload framing and parse validation
- temporary script cleanup
- optional memory guard on Unix (`resource`)

For production internet-facing usage, add container-level isolation (Docker/Firecracker)
and strict syscall/network controls.

## Deployment Readiness

This repository is optimized for hackathon deployment:
- Local CLI works immediately.
- Artifacts are deterministic and easy to demo.
- Structure is compatible with future Colab and Hugging Face workflows.

Hugging Face Space readiness:
- requirements include runtime, plotting, YAML, and HTTP dependencies
- optional transformers/torch lines are provided for local model inference mode
- models/adapters, models/checkpoints, outputs/run_artifacts are prestructured

## License

MIT
