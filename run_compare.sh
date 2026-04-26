#!/bin/bash
# Run FORGE-v4 Benchmark Comparison
echo "[FORGE] Starting Comparison Benchmark (Baseline vs Model)..."
if command -v python >/dev/null 2>&1; then
  python train_colab.py --compare --episodes 20
else
  py train_colab.py --compare --episodes 20
fi

