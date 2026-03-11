#!/bin/bash
# Quick submission script for Azure ML DPO training

# Azure ML Configuration (from your Azure ML workspace)
SUBSCRIPTION_ID="fc1f3270-30de-4538-b720-3f2ac5377083"
RESOURCE_GROUP="chemEng"
WORKSPACE_NAME="ChemEngTraining"

# Compute Configuration
COMPUTE_NAME="gpu-h100-2x"
VM_SIZE="Standard_ND96isr_H100_v5"  # H100 GPU (same as your video generation)
MAX_NODES=1
NODE_COUNT=1

# Data Configuration
DATA_PATH=""  # Leave empty to upload from local
LOCAL_DATA_DIR="/Users/hivamoh/cs234Proj/azureml_outputs_final"
PREF_FILE="video_rankings3_pairwise.json"
SKIP_UPLOAD=true  # Set to true if data is already uploaded to Azure

# Model Configuration
MODEL="Wan-AI/Wan2.2-TI2V-5B-Diffusers"  # Base video generation model (T2V-A14B for text-only, TI2V-5B for text+image)
TEXT_ENCODER="Qwen/Qwen3-VL-2B-Instruct"  # Custom text encoder (leave empty for default)

# Training Hyperparameters
BETA=0.1
LR=1e-4  # Increased to 1e-4 - rewards show policy not diverging from reference at 5e-5
BATCH_SIZE=1
GRAD_ACCUM=4
EPOCHS=5
N_FRAMES=8
NUM_INFERENCE_STEPS=20

# Weights & Biases Configuration
USE_WANDB=false  # Temporarily disabled to test submission
WANDB_PROJECT="wan22-dpo"
WANDB_RUN_NAME=""  # Leave empty for auto-generated name

# Experiment Configuration
EXPERIMENT_NAME="wan22-dpo-training-$(date +%Y%m%d-%H%M%S)"

echo "=========================================="
echo "  Azure ML DPO Training Submission"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Workspace: $WORKSPACE_NAME"
echo "  Compute: $COMPUTE_NAME ($VM_SIZE)"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Epochs: $EPOCHS"
echo "  Beta: $BETA"
echo "  Learning Rate: $LR"
echo "=========================================="
echo ""

# Check if data exists locally
if [ -z "$DATA_PATH" ]; then
    if [ ! -d "$LOCAL_DATA_DIR" ]; then
        echo "ERROR: Local data directory not found: $LOCAL_DATA_DIR"
        echo "Please download data first or specify --data-path"
        exit 1
    fi
    if [ ! -f "$PREF_FILE" ]; then
        echo "ERROR: Preference file not found: $PREF_FILE"
        exit 1
    fi
    echo "✓ Local data found, will upload to Azure"
else
    echo "✓ Using existing Azure data path: $DATA_PATH"
fi

# Build upload flag
SKIP_UPLOAD_FLAG=""
if [ "$SKIP_UPLOAD" = true ]; then
    SKIP_UPLOAD_FLAG="--skip-upload"
    echo "✓ Skipping data upload (already on Azure)"
fi

# Build model flag
MODEL_FLAG=""
if [ ! -z "$MODEL" ]; then
    MODEL_FLAG="--model $MODEL"
    echo "✓ Using model: $MODEL"
fi

# Build text encoder flag
TEXT_ENCODER_FLAG=""
if [ ! -z "$TEXT_ENCODER" ]; then
    TEXT_ENCODER_FLAG="--text-encoder $TEXT_ENCODER"
    echo "✓ Using custom text encoder: $TEXT_ENCODER"
fi

# Build wandb flags
WANDB_FLAGS=""
if [ "$USE_WANDB" = true ]; then
    WANDB_FLAGS="--use-wandb --wandb-project $WANDB_PROJECT"
    if [ ! -z "$WANDB_RUN_NAME" ]; then
        WANDB_FLAGS="$WANDB_FLAGS --wandb-run-name $WANDB_RUN_NAME"
    fi
    echo "✓ W&B enabled: Project=$WANDB_PROJECT"
fi

# Submit job
python azure/submit_job.py \
    --subscription-id "$SUBSCRIPTION_ID" \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE_NAME" \
    --experiment-name "$EXPERIMENT_NAME" \
    --compute-name "$COMPUTE_NAME" \
    --vm-size "$VM_SIZE" \
    --max-nodes "$MAX_NODES" \
    --node-count "$NODE_COUNT" \
    --local-data-dir "$LOCAL_DATA_DIR" \
    --pref-file "$PREF_FILE" \
    $MODEL_FLAG \
    $TEXT_ENCODER_FLAG \
    $WANDB_FLAGS \
    --beta "$BETA" \
    --lr "$LR" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --epochs "$EPOCHS" \
    --n-frames "$N_FRAMES" \
    --num-inference-steps "$NUM_INFERENCE_STEPS" \
    $SKIP_UPLOAD_FLAG

echo ""
echo "Job submitted! Monitor progress in Azure ML Studio."
