#!/bin/bash
# Run FORGE-v4 DPO Fine-tuning
echo "[FORGE] Starting DPO Training Mode..."
if command -v python >/dev/null 2>&1; then
  python train_unsloth.py --mode dpo --episodes 20
else
  py train_unsloth.py --mode dpo --episodes 20
fi

