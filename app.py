import os
import json
import gradio as gr
import pandas as pd
from typing import Any, Dict
from fastapi import FastAPI
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from trainer import run_benchmark_mode, run_compare_mode
from memory import CoachMemory
from metrics.charts import generate_charts
from config import LOG_SUMMARY_FILE, REWARD_GRAPHS_DIR, OUTPUTS_DIR
from api_server import app as api_app

# Handle missing directories
os.makedirs(REWARD_GRAPHS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def get_current_metrics() -> Dict[str, Any]:
    """Load latest metrics from summary.json if it exists."""
    if os.path.exists(LOG_SUMMARY_FILE):
        try:
            with open(LOG_SUMMARY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {}

def get_memory_lessons() -> str:
    """Get top lessons from CoachMemory."""
    memory = CoachMemory()
    summary = memory.summary()
    top_lessons = summary.get("top_lessons", [])
    if not top_lessons:
        return "No lessons recorded yet."

    output = ""
    for idx, lesson in enumerate(top_lessons):
        note = lesson.get("coach_note") or ""
        w = lesson.get("reward_weight", 0.0)
        output += f"{idx + 1}. {note} (Weight: {w})\n"
    return output


def _cuda_ready() -> bool:
    try:
        import torch  # noqa: PLC0415
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _ui_provider_options() -> list[str]:
    # GPU Space: lead with local HF (real weights on T4). CPU: lead with offline so demos stay instant.
    if _cuda_ready():
        return ["custom_hf", "auto", "nim", "openrouter", "offline"]
    return ["offline", "auto", "nim", "openrouter", "custom_hf"]


FORGE_PROVIDER_OPTIONS = _ui_provider_options()


def default_forge_ui_provider() -> str:
    override = os.getenv("FORGE_DEFAULT_PROVIDER", "").strip().lower()
    if override in FORGE_PROVIDER_OPTIONS:
        return override
    return "custom_hf" if _cuda_ready() else "offline"


def _benchmark_episode_cap() -> int:
    return 30 if _cuda_ready() else 5


def _ui_candidates_per_step() -> int:
    """Gradio-only: fewer generations per step so `custom_hf` returns while the queue is still open."""
    return max(1, min(8, int(os.getenv("FORGE_UI_CANDIDATES", "1"))))


def _ui_max_steps_for_gradio() -> int | None:
    """Gradio-only: cap steps per episode (`FORGE_UI_STEPS`). Use full, default, or 0 for global config.STEPS_PER_EPISODE."""
    raw = os.getenv("FORGE_UI_STEPS", "2").strip().lower()
    if raw in ("full", "default", "0"):
        return None
    try:
        return max(1, min(10, int(raw)))
    except ValueError:
        return 2


def run_benchmark_ui(episodes, forge_provider_label: str):
    """Gradio wrapper for benchmark mode."""
    ep_count = min(int(episodes), _benchmark_episode_cap())
    mode = forge_provider_label if forge_provider_label in (
        "auto", "custom_hf", "nim", "openrouter", "offline", "mock"
    ) else "offline"
    report = run_benchmark_mode(
        policy_name="model",
        episodes=ep_count,
        verbose=False,
        forge_provider=mode,
        candidates_per_step=_ui_candidates_per_step(),
        max_steps=_ui_max_steps_for_gradio(),
    )
    
    summary = report.get("summary", {})
    generate_charts() # Update trends too
    lessons = get_memory_lessons()
    
    # Paths for Gradio (as requested by user)
    reward_path = os.path.join(OUTPUTS_DIR, "reward_curve.png")
    pass_rate_path = os.path.join(OUTPUTS_DIR, "pass_rate.png")
    
    return (
        f"{summary.get('avg_pass_rate', 0.0):.2f}",
        f"{summary.get('avg_defender_reward', 0.0):+.2f}",
        f"{summary.get('avg_adversary_reward', 0.0):+.2f}",
        f"{summary.get('max_tier', 1)}",
        reward_path if os.path.exists(reward_path) else None,
        pass_rate_path if os.path.exists(pass_rate_path) else None,
        lessons
    )

def run_compare_ui(episodes, forge_provider_label: str):
    """Gradio wrapper for compare mode."""
    ep_count = min(int(episodes), 10 if _cuda_ready() else 3)
    mode = forge_provider_label if forge_provider_label in (
        "auto", "custom_hf", "nim", "openrouter", "offline", "mock"
    ) else "offline"
    report = run_compare_mode(
        model_policy_name="model",
        episodes=ep_count,
        verbose=False,
        forge_provider=mode,
        candidates_per_step=_ui_candidates_per_step(),
        max_steps=_ui_max_steps_for_gradio(),
    )
    
    model_summary = report.get("model", {})
    generate_charts()
    lessons = get_memory_lessons()
    
    # Paths for Gradio (as requested by user)
    reward_path = os.path.join(OUTPUTS_DIR, "reward_curve.png")
    pass_rate_path = os.path.join(OUTPUTS_DIR, "pass_rate.png")
    
    return (
        f"{model_summary.get('avg_pass_rate', 0.0):.2f}",
        f"{model_summary.get('avg_defender_reward', 0.0):+.2f}",
        f"{model_summary.get('avg_adversary_reward', 0.0):+.2f}",
        f"{model_summary.get('max_tier', 1)}",
        reward_path if os.path.exists(reward_path) else None,
        pass_rate_path if os.path.exists(pass_rate_path) else None,
        lessons
    )

# --- Gradio UI Layout ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# FORGE-v4: Adversarial Robust Code Generation Environment")
    
    # Pre-load data
    initial_lessons = get_memory_lessons()
    initial_reward = os.path.join(OUTPUTS_DIR, "reward_curve.png")
    initial_pass = os.path.join(OUTPUTS_DIR, "pass_rate.png")
    
    with gr.Tab("1. Project Summary"):
        gr.Markdown("""
        ### Adversarial Code-Generation Benchmarking
        FORGE-v4 is an environment for training and evaluating code-generation models against adversarial pressure.
        
        **Key Features:**
        - **Two-Agent Interaction**: Defender (Coder) vs. Adversary (Breaker).
        - **Tiered Red-Teaming**: The Breaker escalates difficulty (negatives, duplicates, large arrays) as the Defender improves.
        - **CoachMemory Feedback**: Models learn from past failures to generate more robust solutions.
        - **OpenEnv Compliant**: Standardized API for LLM agent integration.
        """)
        
    with gr.Tab("2. Training & Evaluation"):
        with gr.Row():
            episodes_input = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Episodes (Limited for Demo)")
            provider_input = gr.Dropdown(
                choices=FORGE_PROVIDER_OPTIONS,
                value=default_forge_ui_provider(),
                label="Inference provider",
                info=(
                    "**custom_hf** = local PyTorch + Hub weights on this machine (default on **GPU**). "
                    "**auto** = NIM → OpenRouter → optional local HF if **HF_TOKEN** is set → else offline. "
                    "**offline** = no external APIs (CPU-friendly fallback). "
                    "Gradio uses **`FORGE_UI_CANDIDATES`** (default 1) and **`FORGE_UI_STEPS`** (default 2 steps/episode; set `full` for config default). CLI/training use full settings."
                ),
            )
        
        with gr.Row():
            btn_benchmark = gr.Button("Run Model Benchmark", variant="primary")
            btn_compare = gr.Button("Compare Baseline vs Model", variant="secondary")
            
        gr.Markdown("### Latest Evaluation Results")
        with gr.Row():
            m_pass = gr.Textbox(label="Pass Rate", placeholder="0.00")
            m_def_reward = gr.Textbox(label="Defender Reward", placeholder="+0.0")
            m_adv_reward = gr.Textbox(label="Adversary Reward", placeholder="+0.0")
            m_tier = gr.Textbox(label="Max Tier Reached", placeholder="1")
            
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Reward Trend")
                plot_reward = gr.Image(value=initial_reward if os.path.exists(initial_reward) else None, label="Reward Curve", type="filepath")
            with gr.Column():
                gr.Markdown("#### Pass Rate Trend")
                plot_pass = gr.Image(value=initial_pass if os.path.exists(initial_pass) else None, label="Pass Rate Curve", type="filepath")
                
        gr.Markdown("### Coach Memory: Top Lessons Learned")
        memory_output = gr.Textbox(value=initial_lessons, lines=5, label="Strategic Improvements", placeholder="Run training to see lessons...")

    with gr.Tab("3. API Endpoints"):
        gr.Markdown("""
        ### OpenEnv API Standard
        FORGE-v4 serves **Gradio at `/`** and the OpenEnv JSON routes at the **same origin** (`/health`, `/reset`, `/step`, `/state`). Locally, `python api_server.py` serves **API-only** on **`:8000`**; `python app.py` serves UI **+** API on **`:7860`**. On this Space, use your **`*.hf.space`** base URL (no `/start` — use **`POST /reset`** then **`POST /step`**).

        - **`GET /health`**: Liveness / version check.
        - **`POST /reset`**: Starts a new episode and returns the initial state (new random task each time unless Space secret **`FORGE_DETERMINISTIC_RESET=1`**).
        - **`POST /step`**: JSON body: `coder_code`, `coder_version`, optional `candidate_solutions` (array of strings). Returns rewards and updated state.
        - **`GET /state`**: Current environment snapshot.

        **Example (replace `BASE` with your Space `https://….hf.space` host):**  
        `curl -sS "$BASE/health"` → `curl -sS -X POST "$BASE/reset" -H "Content-Type: application/json"` → `curl -sS -X POST "$BASE/step" -H "Content-Type: application/json" -d '{"coder_code":"def solution(arr):\\n    return sorted(list(arr))","coder_version":"demo"}'`
        """)

    # Event handlers
    btn_benchmark.click(
        run_benchmark_ui,
        inputs=[episodes_input, provider_input],
        outputs=[m_pass, m_def_reward, m_adv_reward, m_tier, plot_reward, plot_pass, memory_output],
    )
    btn_compare.click(
        run_compare_ui,
        inputs=[episodes_input, provider_input],
        outputs=[m_pass, m_def_reward, m_adv_reward, m_tier, plot_reward, plot_pass, memory_output],
    )

# Mount Gradio at "/" so Hugging Face Spaces (hub iframe + *.hf.space) load assets and
# websockets from the same root. OpenEnv routes on api_app are registered before this mount
# and keep precedence over the Gradio catch-all.
app = gr.mount_gradio_app(
    api_app,
    demo,
    path="/",
    ssr_mode=False,
)

# HF Spaces (and other reverse proxies) terminate TLS and set X-Forwarded-Proto. Without this,
# Gradio's slash redirects emit http://… which the browser blocks inside https iframes → blank UI.
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
