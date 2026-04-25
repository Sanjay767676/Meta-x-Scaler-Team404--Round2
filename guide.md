# FORGE-v4 Mentor Guide

## 1) Project Summary

FORGE-v4 is a controlled adversarial benchmark for robust code generation.

The system runs two competing agents:
- Defender (Coder): generates Python code for sorting integer arrays.
- Adversary (Breaker): generates hard but valid test cases to break the coder.

A sandbox executes generated code safely, rewards are computed, learning traces are stored in memory/logs, and charts/reports are exported for judging.

Why sorting?
- It is intentionally narrow and measurable.
- It gives a deterministic benchmark to prove robustness and iteration quality before scaling to broader coding tasks.

## 2) What We Built

### Core loop
1. Generate task and hidden tests.
2. Defender policy generates multiple candidate solutions.
3. Candidate evaluator ranks candidates by quality.
4. Best candidate is selected.
5. Sandbox executes against hidden and adversarial tests.
6. Defender and adversary rewards are computed.
7. Breaker tier updates based on break performance.
8. Logs, memory, and artifacts are written.

### Production upgrades implemented
- Reliable timeout accounting and deterministic metric writing.
- Pluggable policy system with class-based interfaces.
- Multi-candidate evaluation and ranking (GRPO-ready style).
- Hardened sandbox flow (timeouts, structured errors, temp cleanup, optional memory cap).
- Benchmark mode (20+ episodes) and compare mode (heuristic vs model).
- Judge assets exported automatically.
- Hugging Face-ready folder layout for adapters/checkpoints/artifacts.

## 3) Technology Stack

Language/runtime:
- Python 3.10+

Core libraries:
- pandas
- matplotlib
- requests
- tqdm
- PyYAML

Optional model stack:
- transformers (optional, commented in requirements)
- torch (optional, commented in requirements)

## 4) Repository Structure (Key Files)

Top-level execution:
- app.py: CLI runner (demo, benchmark, compare)
- trainer.py: training and benchmark orchestration
- env.py: OpenEnv-style environment

Evaluation and rewards:
- sandbox.py: code execution sandbox
- rewards.py: defender/adversary reward shaping
- services/candidate_evaluator.py: candidate scoring and ranking

Policy architecture:
- policies/base.py: policy interface
- policies/heuristic.py: deterministic baseline
- policies/api_model.py: OpenRouter-backed policy
- policies/local_model.py: Hugging Face local-backed policy
- policies/mock_model.py: stable fallback policy
- policies/factory.py: provider/policy selection

LLM providers:
- llm_agent.py: provider implementations and fallback behavior

Observability and persistence:
- logger.py: JSON/CSV/summary logging
- memory.py: persistent lesson memory
- storage/artifact_store.py: run artifacts persistence
- metrics/charts.py: chart generation and judge exports

Metadata:
- openenv.yaml: benchmark framing and environment metadata

## 5) Policy Providers and Model Switching

Configured in config.py:
- LLM_PROVIDER: mock | openrouter | huggingface_local
- LLM_MODEL
- HF_LOCAL_MODEL_ID
- OPENROUTER_API_KEY / OPENROUTER_BASE_URL

Provider behavior:
- mock: deterministic offline fallback for guaranteed runs.
- openrouter: real API call path via requests when key is set.
- huggingface_local: local Transformers pipeline; falls back to mock if unavailable.

Default lightweight model profile:
- qwen/qwen2.5-coder-0.5b-instruct

This is intentionally easy to replace through config flags without changing business logic.

## 6) Candidate Ranking Logic

Each candidate is scored using:
- pass rate on hidden tests
- average runtime
- robustness score (penalizes timeout/errors)
- static code quality heuristic

Composite score selects the best candidate per step.

Rankings are exposed in environment info and logged for auditability.

## 7) Reward Logic

Defender reward uses:
- pass reward
- fail penalty
- error penalty
- timeout penalty
- perfect-run bonus

Adversary reward uses:
- break reward scaled by defender baseline strength
- extra bonus for timeout/error breaks
- penalty for non-breaking tests

This creates realistic, non-fake metrics with clear signal direction.

## 8) Sandbox Safety Model

Current protections:
- strict execution timeout
- isolated interpreter mode (-I)
- structured output framing
- temporary runner files with cleanup
- optional memory guard on Unix (resource)
- explicit structured exception typing

Production note:
- For public internet deployment, add container-level isolation (Docker/Firecracker), syscall controls, and network egress restrictions.

## 9) Benchmark and Compare Modes

### Benchmark mode
Command:
- python app.py --policy model --benchmark 20

Outputs episode rows including:
- episode number
- pass rate
- defender reward
- adversary reward
- chosen candidate rank
- tier progression

### Compare mode
Command:
- python app.py --compare

Runs:
1. baseline heuristic policy
2. model policy (config-driven)

Exports improvement deltas in final report.

## 10) Judge Asset Exports

Automatically generated:
- outputs/reward_curve.png
- outputs/pass_rate.png
- outputs/final_report.json

Additional artifacts:
- outputs/reward_graphs/*.png
- outputs/run_artifacts/episode_XXXX.json

## 11) Hugging Face Submission Readiness

Prepared structure:
- models/adapters/
- models/checkpoints/
- outputs/reward_graphs/
- outputs/run_artifacts/

requirements.txt includes stable runtime dependencies.
Optional model dependencies are documented for local inference mode in Spaces.

## 12) How To Run

Install:
- pip install -r requirements.txt

Default run:
- python app.py

Policy run:
- python app.py --policy model --candidates 3 --steps 3

Benchmark run:
- python app.py --policy model --benchmark 20

Compare run:
- python app.py --compare

## 13) Suggested Mentor Q&A Answers

Q: What is novel here?
A: We combine adversarial test generation with multi-candidate defender ranking under a controlled benchmark and export deterministic evidence artifacts for reproducibility.

Q: How do you prove learning/performance?
A: Benchmark mode runs 20+ episodes and exports reward/pass-rate curves plus structured final_report.json and per-episode artifacts.

Q: Why not a broader coding benchmark?
A: Sorting is a constrained benchmark chosen for measurable reliability. The architecture is intentionally modular so broader tasks can be plugged in after benchmark validation.

Q: How production-ready is this?
A: It is hackathon-production ready for demo and evaluation pipelines, with clear next hardening steps for public untrusted execution.

Q: How can this scale?
A: Swap policy providers/models through config, expand tasks/test generators, and integrate full RL updates using trainer hooks.

## 14) Current Limitations and Next Steps

Current limitations:
- HF local model path depends on optional transformers/torch install.
- OpenRouter path requires valid API key and network.
- Sandbox is not yet container-isolated.

Next steps:
- Add containerized sandbox runtime.
- Add richer benchmarks beyond sorting while preserving deterministic scoring.
- Integrate full GRPO/PPO update loop in trainer hooks.

## 15) One-Line Pitch

FORGE-v4 is a competitive, evidence-driven adversarial code-generation benchmark where model policies are tested, ranked, and compared with reproducible artifacts ready for hackathon judging.
