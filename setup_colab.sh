#!/bin/bash
# FORGE-v4: Colab Environment Setup Script
# Automated installation for Unsloth, TRL, and Adversarial red-teaming.

echo "------------------------------------------------------------"
echo "  FORGE-v4: Adversarial Robustness Setup"
echo "------------------------------------------------------------"

echo "[*] Updating pip..."
python -m pip install --upgrade pip

echo "[*] Installing core requirements..."
pip install -r requirements.txt

echo "[*] Installing Unsloth (Colab Optimized)..."
# Using the specific Unsloth Colab installation command
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

echo "[*] Verifying GPU acceleration..."
if command -v nvidia-smi &> /dev/null
then
    nvidia-smi
else
    echo "[!] WARNING: No NVIDIA GPU detected. Training will be slow or unavailable."
fi

echo "[*] Creating runtime directories..."
mkdir -p data models outputs logs

echo "------------------------------------------------------------"
echo "  [OK] FORGE-v4 setup complete."
echo "  You can now run: bash run_compare.sh"
echo "------------------------------------------------------------"
