#!/bin/bash
# Azure ML Training Wrapper Script
# This script initializes the environment and then runs the training

set -e

# Initialize environment (set cache dirs to /tmp)
source azure/init_environment.sh

# Export environment variables so Python can see them
export HF_HOME="/tmp/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="/tmp/huggingface_cache"
export TRANSFORMERS_CACHE="/tmp/huggingface_cache"
export HF_HUB_CACHE="/tmp/huggingface_cache"
export TORCH_HOME="/tmp/torch_cache"

# Run the training script with all passed arguments
echo ""
echo "Starting training..."
python azure/train_dpo_azure.py "$@"
