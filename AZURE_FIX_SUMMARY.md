# Azure ML DPO Training - Fixes Applied

## Issues Fixed

### 1. Wrong Model Format
**Problem**: Training script was using `Wan-AI/Wan2.2-T2V-A14B` which doesn't have the diffusers structure required for DPO training.

**Fix**: Changed default model to `Wan-AI/Wan2.2-T2V-A14B-Diffusers` which has the proper structure with `vae/`, `text_encoder/`, and `transformer/` subfolders.

**File**: `azure/train_dpo_azure.py` line 689

### 2. DataReferenceConfiguration Path Resolution
**Problem**: Azure ML was passing `DataReferenceConfiguration` objects instead of actual file paths to the training script, causing `FileNotFoundError`.

**Fix**:
- Added `path_on_compute` parameter to specify mount points
- Used `.as_mount()` and `.as_download()` methods to get proper path placeholders
- Azure ML now correctly resolves these to actual paths at runtime

**Files**: `azure/submit_job.py` lines 171-198

### 3. Data Upload Disabled
**Problem**: The actual data upload code was commented out, so videos and preference files were never uploaded to Azure Blob Storage.

**Fix**: Uncommented the `datastore.upload()` call to enable data uploads.

**File**: `azure/submit_job.py` lines 92-103

### 4. Qwen3-VL Integration
**Added**: Full support for using Qwen3-VL-2B as a custom text encoder with learned projection layer.

**Files**:
- `azure/train_dpo_azure.py`: Added Qwen3-VL loading and projection layer
- `azure/submit_job.py`: Added `--text-encoder` parameter
- `azure/submit_job.sh`: Added `TEXT_ENCODER` configuration

## How to Run

### With Qwen3-VL-2B (Recommended)

The submission script is already configured to use Qwen3-VL-2B:

```bash
cd /Users/hivamoh/cs234Proj/CS234-Project-RL
bash azure/submit_job.sh
```

This will:
1. Upload your videos and preference data to Azure Blob Storage
2. Create/connect to GPU compute cluster
3. Submit training job with:
   - Video generation: Wan-AI/Wan2.2-T2V-A14B-Diffusers
   - Text encoding: Qwen/Qwen3-VL-2B-Instruct
   - Learned projection layer: 896 → 4096 dimensions

### Without Qwen3-VL (Default Text Encoder)

Edit `azure/submit_job.sh` and set:
```bash
TEXT_ENCODER=""  # Empty = use default from Wan2.2
```

Then run:
```bash
bash azure/submit_job.sh
```

## Configuration Options

### In azure/submit_job.sh

```bash
# Azure ML workspace settings
SUBSCRIPTION_ID="fc1f3270-30de-4538-b720-3f2ac5377083"
RESOURCE_GROUP="chemEng"
WORKSPACE_NAME="ChemEngTraining"

# Compute settings
COMPUTE_NAME="gpu-h100-2x"
VM_SIZE="Standard_ND96isr_H100_v5"  # H100 GPU

# Data settings
LOCAL_DATA_DIR="/Users/hivamoh/cs234Proj/azureml_outputs_final"
PREF_FILE="video_rankings3_pairwise.json"
SKIP_UPLOAD=true  # Set to false if data needs to be uploaded

# Model settings
TEXT_ENCODER="Qwen/Qwen3-VL-2B-Instruct"  # Custom text encoder

# Training hyperparameters
BETA=0.1          # DPO temperature
LR=1e-6           # Learning rate
BATCH_SIZE=1      # Batch size per GPU
GRAD_ACCUM=4      # Gradient accumulation steps
EPOCHS=10         # Training epochs
N_FRAMES=8        # Frames per video
```

## What Gets Trained

When using Qwen3-VL-2B:
- ✅ **Wan2.2 Transformer** (~14B params): Policy network for DPO
- ✅ **Text Embedding Projector** (~3M params): Maps Qwen embeddings to Wan2.2 dimension
- ❌ **Qwen3-VL** (2B params): Frozen - used only for text encoding
- ❌ **VAE**: Frozen - used for video encoding

Total trainable: **~14.3B parameters**

## Output Structure

After training completes, outputs are saved to Azure Blob Storage:

```
wan22_dpo_outputs/
├── checkpoint-500/
│   ├── config.json
│   ├── diffusion_pytorch_model.safetensors
│   └── text_projector.pt              # If using Qwen3-VL
├── checkpoint-1000/
│   └── ...
├── best_model/
│   ├── config.json
│   ├── diffusion_pytorch_model.safetensors
│   └── text_projector.pt
└── final_model/
    ├── config.json
    ├── diffusion_pytorch_model.safetensors
    └── text_projector.pt
```

## Monitoring Training

1. Go to [Azure ML Studio](https://ml.azure.com)
2. Navigate to your workspace: `ChemEngTraining`
3. Click on "Experiments" in the left sidebar
4. Find your experiment: `wan22-dpo-training-<timestamp>`
5. Monitor:
   - Real-time logs
   - Metrics (loss, accuracy, reward margin)
   - GPU utilization
   - Training progress

## Downloading Results

After training completes, download outputs:

```bash
# Using Azure CLI
az ml job download-output \
  --name <job-name> \
  --output-name outputs \
  --download-path ./trained_models

# Or download from Azure ML Studio UI
# Experiments → Your Job → Outputs + logs → Download
```

## Troubleshooting

### "No cached model found"
This is normal. Models will be downloaded from HuggingFace to `/tmp/huggingface_cache` on the compute node.

### "Out of memory"
Reduce `BATCH_SIZE=1` and `N_FRAMES=4`, increase `GRAD_ACCUM=8`

### "Data not found"
Set `SKIP_UPLOAD=false` in `submit_job.sh` to upload data

### "Compute cluster not found"
The script will automatically create the compute cluster if it doesn't exist

## Memory Requirements

- **Wan2.2-T2V-A14B-Diffusers**: ~28GB
- **Qwen3-VL-2B**: ~4GB
- **Training overhead**: ~8-16GB
- **Total**: ~40-48GB VRAM

Recommended GPU: A100 40GB or H100 80GB

## Next Steps

1. **Submit training**: `bash azure/submit_job.sh`
2. **Monitor in Azure ML Studio**: Check logs and metrics
3. **Download trained model**: After job completes
4. **Generate videos**: Use trained model for inference
5. **Evaluate**: Test on held-out videos

See `RUN_WITH_QWEN.md` for detailed usage guide.
