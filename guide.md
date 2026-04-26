# FORGE-v4 Submission Guide

## Colab-First Commands

1. Benchmark with model policy:

```bash
python train_colab.py --benchmark --policy model --episodes 20
```

2. Compare baseline vs model:

```bash
python train_colab.py --compare --episodes 20
```

3. Top up authentic DPO dataset to 480 pairs:

```bash
python train_colab.py --benchmark --policy model --episodes 20 --topup-dpo --target-pairs 480
```

4. Verify pair count:

```bash
python -c "import pathlib; p=pathlib.Path('data/dpo_dataset.jsonl'); print(sum(1 for _ in p.open('r',encoding='utf-8')) if p.exists() else 0)"
```

## Security Notes

- API keys must be set via environment variables.
- No secrets should be hardcoded in source files.
- Sandbox enforces timeout, memory cap (where supported), blocked risky builtins, and temp cleanup.
- For public deployment, add container isolation.

## What Judges Should See

- outputs/reward_curve.png
- outputs/loss_curve.png
- outputs/pass_rate.png
- outputs/final_report.json
- data/dpo_dataset.jsonl with target pair count
