import os
import json
import gradio as gr
import pandas as pd
from typing import Any, Dict
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

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
        output += f"{idx+1}. Episode {lesson.get('episode')}: {lesson.get('coach_note')} (Weight: {lesson.get('reward_weight')})\n"
    return output

FORGE_PROVIDER_OPTIONS = ["auto", "custom_hf", "nim", "openrouter", "mock"]


def run_benchmark_ui(episodes, forge_provider_label: str):
    """Gradio wrapper for benchmark mode."""
    # Limit episodes for demo stability on CPU
    ep_count = min(int(episodes), 5)
    mode = forge_provider_label if forge_provider_label in (
        "auto", "custom_hf", "nim", "openrouter", "mock"
    ) else "auto"
    report = run_benchmark_mode(
        policy_name="model",
        episodes=ep_count,
        verbose=False,
        forge_provider=mode,
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
    ep_count = min(int(episodes), 3)  # Very small for demo
    mode = forge_provider_label if forge_provider_label in (
        "auto", "custom_hf", "nim", "openrouter", "mock"
    ) else "auto"
    report = run_compare_mode(
        model_policy_name="model",
        episodes=ep_count,
        verbose=False,
        forge_provider=mode,
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
                value="auto",
                label="Inference provider",
                info="Auto tries HF adapter, then NIM, then OpenRouter, then deterministic mock.",
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
        FORGE-v4 exposes a FastAPI server on the **same origin** as this UI: routes live at the **site root**, while Gradio is under **`/ui`**. Locally, `python api_server.py` serves on **`:8000`**; on this Space, use your **`*.hf.space`** base URL (no separate `/start` — use **`POST /reset`** then **`POST /step`**).

        - **`GET /health`**: Liveness / version check.
        - **`POST /reset`**: Starts a new episode and returns the initial state.
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

space_app = gr.mount_gradio_app(api_app, demo, path="/ui")


@space_app.get("/")
async def root_redirect() -> RedirectResponse:
    return RedirectResponse(url="/ui", status_code=302)


app = space_app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
