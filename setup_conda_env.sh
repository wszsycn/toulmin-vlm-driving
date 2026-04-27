#!/usr/bin/env bash
# Setup conda environment for Cosmos-Reason2-8B training
# Single GPU (RTX 6000 Ada), CUDA 12.4, no torchrun/DDP
set -euo pipefail

ENV_NAME="cosmos2"

echo "=== Creating conda environment: ${ENV_NAME} ==="
conda create -y -n "${ENV_NAME}" python=3.10

echo "=== Activating environment ==="
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "=== Installing PyTorch 2.4.0 (CUDA 12.4) ==="
conda install -y pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    pytorch-cuda=12.4 -c pytorch -c nvidia

echo "=== Installing ffmpeg from conda-forge ==="
conda install -y -c conda-forge ffmpeg

echo "=== Verifying mpeg4 codec ==="
ffmpeg -codecs 2>/dev/null | grep -i mpeg4 || echo "[WARN] mpeg4 codec not found"

echo "=== Installing core Python packages ==="
pip install --upgrade pip

pip install \
    "transformers>=4.45.0" \
    "trl>=0.11.0" \
    "peft>=0.13.0" \
    "accelerate>=0.34.0" \
    datasets \
    bitsandbytes \
    einops \
    qwen-vl-utils \
    opencv-python \
    openai \
    tensorboard \
    numpy \
    Pillow

echo "=== Installing decord 0.6.0 ==="
pip install decord==0.6.0

echo "=== Installing flash-attn (may take 10-20 min to compile) ==="
# Must be installed AFTER torch. If compile fails, add attn_implementation="eager" to model loading.
pip install flash-attn --no-build-isolation || {
    echo "[WARN] flash-attn failed to compile — skipping."
    echo "[WARN] Add attn_implementation='eager' to model loading calls."
}

echo ""
echo "=== Setup complete ==="
echo "Activate with: conda activate ${ENV_NAME}"
echo "Verify with:   python verify_env.py"
