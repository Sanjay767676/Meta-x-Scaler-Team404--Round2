#!/bin/bash
# Run FORGE-v4 GRPO Fine-tuning
echo "[FORGE] Starting GRPO Training Mode..."
python train_unsloth.py --mode grpo --episodes 10
