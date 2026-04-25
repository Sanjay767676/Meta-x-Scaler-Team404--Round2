# 🛡️ FORGE-v4: Building the "Immune System" for AI Code Generation

### The Silent Crisis in AI Coding
We've all seen it: an AI writes a perfect "Quick Sort" in seconds. But what happens when you give that same code an array of 10,000 duplicate zeros? Or a list of mixed large negatives? Often, the AI's "perfect" code crashes, enters an infinite loop, or returns incorrect results. 

Standard benchmarks measure **capability**. We built **FORGE-v4** to measure **robustness**.

---

## ⚔️ The Concept: Adversarial Red-Teaming
FORGE-v4 isn't just a static test suite; it's a living environment. We implemented a **Red-vs-Blue** dynamic:
- **The Defender (Blue Team)**: Our Coder agent tries to solve sorting tasks correctly.
- **The Adversary (Red Team)**: Our Breaker agent actively searches for the Coder's "blind spots."

As the Coder improves, the Breaker escalates. It progresses through **4 Tiers of difficulty**—from basic lists to extreme boundary values and stress tests. This tiered red-teaming ensures that the model isn't just memorizing common patterns, but actually hardening its logic.

---

## 🧠 The Secret Sauce: CoachMemory
One of the most innovative features of FORGE-v4 is the **CoachMemory feedback loop**. 

In most training environments, a model fails, gets a low reward, and moves on. In FORGE-v4, every failure is analyzed by the "Coach." 
*   Did the model fail on negatives? 
*   Did it time out on large arrays? 
*   Did it destroy duplicates? 

These insights are stored in persistent memory. In the next episode, the model reads these "lessons" and adapts its strategy. This mimics the human engineering process: **Mistake → Analysis → Correction.**

---

## 📈 Results that Matter
Our benchmarks show that while a baseline heuristic policy might have a high "average" pass rate (91%), it is easily broken by Tier 3 and Tier 4 attacks. 

Our **FORGE-v4 Model Policy** achieved:
- **100% Pass Rate** across all adversarial tiers.
- **+2.10 Reward Gain** over the baseline.
- **Sustained Tier 4 Robustness**: It didn't just survive; it thrived under extreme pressure.

---

## 🌍 Why This Matters
As AI agents move from "writing scripts" to "building infrastructure," robustness is no longer optional. FORGE-v4 provides the framework to ensure that the code powering our world is not just smart, but **unbreakable**.

**Try the demo:** [Hugging Face Space](https://huggingface.co/spaces/sanjay7676/Team404_FORGE)

---
*Created with ❤️ for the Meta OpenEnv Hackathon by Team 404.*
