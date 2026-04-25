import os
import json
import gradio as gr
import pandas as pd
from typing import Any, Dict

from trainer import run_benchmark_mode, run_compare_mode
from memory import CoachMemory
from metrics.charts import generate_charts
from config import LOG_SUMMARY_FILE, REWARD_GRAPHS_DIR, OUTPUTS_DIR

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

def run_benchmark_ui(episodes):
    """Gradio wrapper for benchmark mode."""
    # Limit episodes for demo stability on CPU
    ep_count = min(int(episodes), 5) 
    report = run_benchmark_mode(policy_name="model", episodes=ep_count, verbose=False)
    
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

def run_compare_ui(episodes):
    """Gradio wrapper for compare mode."""
    ep_count = min(int(episodes), 3) # Very small for demo
    report = run_compare_mode(model_policy_name="model", episodes=ep_count, verbose=False)
    
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
        FORGE-v4 exposes a FastAPI server (available at `:8000` when running locally) with the following endpoints:
        
        - **`POST /reset`**: Initializes a new episode and returns the problem description.
        - **`POST /step`**: Receives code candidates, evaluates them, and returns rewards/diagnostics.
        - **`GET /state`**: Returns current environment status and memory summary.
        
        These endpoints allow external agents to interface with FORGE-v4 programmatically.
        """)

    # Event handlers
    btn_benchmark.click(
        run_benchmark_ui, 
        inputs=[episodes_input], 
        outputs=[m_pass, m_def_reward, m_adv_reward, m_tier, plot_reward, plot_pass, memory_output]
    )
    btn_compare.click(
        run_compare_ui, 
        inputs=[episodes_input], 
        outputs=[m_pass, m_def_reward, m_adv_reward, m_tier, plot_reward, plot_pass, memory_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
