#!/bin/bash
# Run FORGE-v4 DPO Fine-tuning
echo "[FORGE] Starting DPO Training Mode..."
python train_unsloth.py --mode dpo --episodes 20
