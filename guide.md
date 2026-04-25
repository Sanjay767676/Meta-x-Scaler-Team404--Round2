# FORGE-v4: Adversarial Code Generation & Robustness Benchmark

## 1. Project Overview
**FORGE-v4** (Framework for Observability, Robustness, and Generation Evaluation) is a state-of-the-art adversarial benchmark designed to evaluate and improve the reliability of LLM-generated code. 

The project implements a **Competitive Agentic Loop** where two primary entities interact in a high-stakes "Red vs. Blue" scenario:
- **The Defender (Coder Agent):** Responsible for generating optimized Python code to solve algorithmic challenges (currently focusing on integer sorting as a controlled variable).
- **The Adversary (Breaker Agent):** Responsible for identifying edge cases, bottlenecks, and logic flaws by generating valid but difficult test cases designed to "break" the defender's code.

### Why this approach?
Most LLM benchmarks focus on static correctness. FORGE-v4 focuses on **Robustness**. By forcing the coder to iterate against a tiered adversary, we can measure and improve the model's ability to handle edge cases, performance constraints, and adversarial inputs.

---

## 2. Technology Stack
We have utilized a modern, modular Python stack to ensure high performance and auditability:

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core logic and scripting. |
| **Data Analysis** | Pandas | Processing logs and generating performance metrics. |
| **Visualization** | Matplotlib | Generating real-time trend charts and reward curves. |
| **Networking** | Requests | Integrating with external LLM providers (OpenRouter, etc.). |
| **Persistence** | JSON / CSV | Storing coach memory, logs, and episode artifacts. |
| **Configuration** | PyYAML | Environment framing and metadata management. |
| **LLM Integration** | Transformers / OpenRouter | Supporting both local (Hugging Face) and cloud-based models. |

---

## 3. Detailed Architecture
The codebase is organized into several key modules, each handling a specific part of the adversarial loop:

### A. Environment & Task Management (`env.py`, `tasks.py`)
- **FORGEEnv:** A custom OpenEnv-compliant environment that manages the state, rewards, and step logic.
- **Task Generators:** Modules that create problems and maintain a hidden bank of standard test cases.

### B. Sandbox Execution (`sandbox.py`)
- **Safety First:** Executes generated code in an isolated subprocess.
- **Constraint Enforcement:** Implements strict timeouts and memory limits to prevent runaway processes.
- **Structured Errors:** Translates Python tracebacks into actionable feedback for the coder agent.

### C. Policy Framework (`policies/`)
- Supports multiple policy types: `Heuristic`, `API Model`, `Local Model`, and `Mock`.
- **Multi-Candidate Ranking:** Generates multiple solutions per step and uses a `CandidateEvaluator` to select the most robust one based on pass rates and performance.

### D. Reward Engine (`rewards.py`)
- **Defender Rewards:** Balanced between correctness (pass rate), speed (latency), and robustness (escaping the breaker).
- **Adversary Rewards:** Tiered rewards based on the difficulty of the test cases that successfully break the defender.

### E. Persistence & Observability (`logger.py`, `memory.py`, `metrics/`)
- **Coach Memory:** A persistent JSON-based memory that stores "lessons learned" from previous episodes.
- **Logging:** Detailed JSON/CSV logs for every step, used for downstream analysis.
- **Charts:** Automatic generation of reward trends and pass-rate progressions.

---

## 4. Key Features for Mentors
When discussing this project with mentors, focus on these highlights:

1. **Adversarial Hardening:** Unlike simple code-gen tasks, our coder must survive against increasingly difficult "Breaker" tiers.
2. **GRPO-Inspired Ranking:** We implement a candidate selection phase that mirrors modern reinforcement learning techniques (like Group Relative Policy Optimization) by sampling multiple solutions and picking the "best of N."
3. **Deterministic Benchmarking:** We include a `--benchmark` mode that runs 20+ episodes to provide statistically significant data on model performance.
4. **Production Readiness:** The system includes structured logging, artifact exports, and a pluggable architecture ready for any LLM backend.
5. **Measurable Improvement:** Using the `--compare` mode, we can quantify exactly how much better a model-based policy performs compared to a heuristic baseline.

---

## 5. How to Run the Project

### Prerequisites
```bash
pip install -r requirements.txt
```

### Standard Demo
Runs a single episode with the default improving coder.
```bash
python app.py --charts
```

### Benchmark Mode
Runs 25 episodes to generate a comprehensive report.
```bash
python app.py --policy model --benchmark 25 --charts
```

### Comparative Analysis
Compares the baseline heuristic against the current model policy.
```bash
python app.py --compare
```

---

## 6. Mentor Q&A (Preparation)

**Q: Why did you choose sorting as the task?**
*A: Sorting provides a mathematically controlled environment where correctness is binary and performance is easily measurable. This allows us to focus on the adversarial mechanics and robustness without the "noise" of complex business logic.*

**Q: How do you handle safety when running generated code?**
*A: We use a multi-layered sandbox approach: execution in an isolated interpreter mode, strict timeouts, and structured exception handling to ensure that even "bad" code cannot affect the host system.*

**Q: What is the benefit of the "Coach Memory"?**
*A: The Coach Memory allows the model to "learn" from its failures. By storing specific adversarial cases that broke previous versions, the model can avoid repeating the same mistakes in future iterations.*

---

## 7. Future Roadmap
- **Dynamic Task Expansion:** Moving beyond sorting into more complex algorithmic tasks like pathfinding or data structure management.
- **Full RL integration:** Implementing a direct PPO/GRPO update loop to fine-tune local models based on the rewards generated in the environment.
- **De-localized Sandbox:** Moving from local subprocesses to containerized (Docker/Firecracker) execution environments.

---
**FORGE-v4 Team** | *Building robust agents for a safer coding future.*
