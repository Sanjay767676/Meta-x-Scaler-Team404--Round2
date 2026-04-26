# FORGE-v4 Mini Blog: From Fragile Code to Adversarial Robustness

## The story in one line
FORGE-v4 trains a coding agent to survive adversarial edge cases by making it fight a breaker, learn from failures, and improve over repeated reward-driven episodes.

## Why we built this
Most coding models look good on clean examples and then fail on real inputs: negatives, duplicates, boundary values, and timeout-prone cases. We wanted an environment where failure is explicit, measurable, and useful for training.

## The journey
### Chapter 1: baseline confidence, hidden fragility
We started with a defender that often passed easy tests but broke under stress tiers. That gave us a critical signal: average correctness is not robustness.

### Chapter 2: breaker escalation
We added a tiered breaker that progressively attacked blind spots. The environment moved from simple lists to harder adversarial distributions.

### Chapter 3: memory as improvement engine
CoachMemory converted repeated failure patterns into structured lessons. Instead of forgetting mistakes each episode, the loop made mistakes actionable.

### Chapter 4: measurable training loop
We used benchmark/compare runs to produce reward and pass-rate evidence, exported preference pairs, and connected that to a small-model-first adapter training path.

## What changed after training cycles
- Defender pass rate stabilized under tougher tiers.
- Average defender reward improved versus baseline runs.
- Breaker pressure remained high, but the defender failed less often on known edge patterns.

## Evidence (committed outputs)
### Reward trend
![Reward curve](outputs/reward_curve.png)

### Pass-rate trend
![Pass rate curve](outputs/pass_rate.png)

### Loss-like training signal
![Loss curve](outputs/loss_curve.png)

### Machine-readable benchmark summary
- `outputs/final_report.json`

## Deliverables
- Hugging Face Space: https://huggingface.co/spaces/sanjay7676/Team404_FORGE
- GitHub repository: https://github.com/Sanjay767676/Meta-x-Scaler-Team404--Round2
- **Docker image (public — anyone can pull)**
  - **Docker Hub (browse tags):** https://hub.docker.com/r/sanjay767676/forge
  - **Pull command:** `docker pull sanjay767676/forge:latest`
  - **Registry image reference:** `docker.io/sanjay767676/forge:latest`
- Colab notebook: https://colab.research.google.com/github/Sanjay767676/Meta-x-Scaler-Team404--Round2/blob/main/FORGE_Training_Colab.ipynb
- Colab model + adapter training: https://colab.research.google.com/drive/1mKXjIX-eB2GSiebI-_n37KzVlN1NKCu8?usp=sharing
- YouTube demo placeholder: https://youtube.com/watch?v=YOUR_DEMO_VIDEO_ID

## Why this matters
FORGE-v4 is designed to train coding behavior that is verifiable, harder to reward-hack, and more resilient under adversarial conditions. That is the capability gap we think matters most for real LLM deployment.
