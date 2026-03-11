#!/bin/bash
# Azure ML Environment Initialization Script
# This script MUST run before train_dpo_azure.py to set cache directories to /tmp

set -e

echo "=========================================="
echo "  Azure ML Environment Initialization"
echo "=========================================="
echo ""

# Set HuggingFace cache to /tmp (large disk)
export HF_HOME="/tmp/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="/tmp/huggingface_cache"
export TRANSFORMERS_CACHE="/tmp/huggingface_cache"
export HF_HUB_CACHE="/tmp/huggingface_cache"
export TORCH_HOME="/tmp/torch_cache"

# Create cache directories
mkdir -p "$HF_HOME"
mkdir -p "$TORCH_HOME"

echo "✓ HuggingFace cache: $HF_HOME"
echo "✓ Torch cache: $TORCH_HOME"
echo ""

# Show disk space
echo "Disk space:"
df -h / /tmp
echo ""

# Check for existing cached models
echo "Checking for cached models..."
if [ -d "/tmp/huggingface_cache/models--Wan-AI--Wan2.2-T2V-A14B" ]; then
    echo "✓ Found cached Wan2.2-T2V-A14B model"
fi
if [ -d "/tmp/huggingface_cache/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers" ]; then
    echo "✓ Found cached Wan2.2-T2V-A14B-Diffusers model"
fi

echo ""
echo "Environment ready for training!"
echo "=========================================="
