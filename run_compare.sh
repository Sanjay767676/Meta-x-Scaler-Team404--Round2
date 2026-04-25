#!/bin/bash
# Run FORGE-v4 Benchmark Comparison
echo "[FORGE] Starting Comparison Benchmark (Baseline vs Model)..."
python train_colab.py --compare --episodes 50
