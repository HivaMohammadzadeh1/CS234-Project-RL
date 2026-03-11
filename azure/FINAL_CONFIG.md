# Azure ML DPO Training - Final Configuration

## Model Format Clarification

### Why Wan2.2-T2V-A14B-Diffusers?

The training script **requires** the Diffusers format because it loads model components separately:

```python
vae = AutoencoderKLWan.from_pretrained(model, subfolder="vae")
pipe = WanPipeline.from_pretrained(model)
transformer = pipe.transformer
text_encoder = pipe.text_encoder
```

**Two Model Formats:**

| Model | Format | Structure | Use Case |
|-------|--------|-----------|----------|
| `Wan-AI/Wan2.2-T2V-A14B` | Original | Monolithic `.pth` files | Video generation with native Wan2.2 code |
| `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | Diffusers | Separate `vae/`, `text_encoder/`, `transformer/` subfolders | DPO training with HuggingFace Diffusers |

**What you have cached:**
- `/tmp/huggingface_cache/models--Wan-AI--Wan2.2-T2V-A14B/` - Original format (used for video generation)
- This **cannot** be used for DPO training without conversion

**What will be downloaded:**
- `Wan-AI/Wan2.2-T2V-A14B-Diffusers` (~120GB) to `/tmp/huggingface_cache/`
- Will download once and be cached for subsequent training runs on same compute node

---

## Training Configuration

### Key Parameters

```bash
# Model
--model Wan-AI/Wan2.2-T2V-A14B-Diffusers  # Diffusers format (required)

# Data
--data video_rankings3_pairwise.json
--videos-dir /path/to/videos

# DPO Hyperparameters
--beta 0.1                 # DPO temperature
--lr 1e-6                  # Learning rate
--batch-size 1             # Batch size per GPU
--grad-accum 4             # Gradient accumulation steps (effective batch size = 4)
--epochs 10                # Number of epochs

# Training Settings
--n-frames 8               # Frames per video for training
--num-inference-steps 20   # Diffusion steps (for evaluation/generation)
--max-grad-norm 1.0        # Gradient clipping
--save-every 500           # Save checkpoint every N steps

# Train/Test Split
--train-only               # Use only training videos (exclude test set)
--test-indices "0,1,8,15,21,25,27,28,45,48,50,52"  # Test video indices
```

### Effective Training Setup

- **Effective batch size**: 1 × 4 = 4 (with gradient accumulation)
- **Train videos**: 48 (60 total - 12 test)
- **Frames per video**: 8
- **Diffusion steps**: 20 (for generation/evaluation, not used during training)

---

## Quick Start

### Submit Training Job to Azure ML

```bash
# From project root
bash azure/submit_job.sh
```

This will:
1. Upload videos and preference file to Azure Blob Storage
2. Download Wan2.2-T2V-A14B-Diffusers model to `/tmp` (~120GB, one-time download)
3. Train with all available data (or use `--train-only` for training set only)
4. Save checkpoints every 500 steps
5. Log metrics to Azure ML

### Training Only on Training Set

Edit `azure/submit_job.py` and add to arguments list (around line 196):

```python
arguments = [
    "--data", pref_input,
    "--videos-dir", videos_input,
    "--output-dir", output,
    "--beta", beta,
    "--lr", lr,
    "--batch-size", batch_size,
    "--grad-accum", grad_accum,
    "--epochs", epochs,
    "--n-frames", n_frames,
    "--train-only",              # ADD THIS LINE
    "--num-inference-steps", 20, # ADD THIS LINE
]
```

---

## Expected Output

### 1. Environment Setup
```
======================================================================
  Configuring HuggingFace Cache Location
======================================================================
✓ HuggingFace cache: /tmp/huggingface_cache
✓ Torch cache: /tmp/torch_cache

Disk space before model loading:
Filesystem      Size  Used Avail Use% Mounted on
/dev/sdb1       251G  XXG  XXXG  XX% /tmp
```

### 2. Model Loading
```
Checking for cached models...
NOTE: DPO training requires Wan2.2-T2V-A14B-Diffusers format
  (has vae/, text_encoder/, transformer/ subfolders)
  Cannot use the original Wan2.2-T2V-A14B format

⚠ No cached model found. Will download from HuggingFace to /tmp
  Model: Wan-AI/Wan2.2-T2V-A14B-Diffusers

[Downloads model ~120GB to /tmp]

✓ Pipeline loaded
Creating reference transformer (frozen)...
Trainable parameters: 14,288,491,584
```

### 3. Dataset Loading
```
======================================================================
  Loading Preference Dataset
======================================================================
Found X video files in /mnt/azureml/.../videos
Found Y prompt groups in preference file

Filtering dataset by 48 indices...  # If --train-only
  Pairs before filtering: 1,400
  Pairs after filtering: 1,120

Dataset Statistics:
  Total pairs in JSON: 1,400
  Valid pairs (both videos exist): 1,120
```

### 4. Training
```
======================================================================
  Starting DPO Training
======================================================================
Epochs: 10
Batch size: 1 (effective: 4)
Beta: 0.1

Epoch 1/10:  25%|████▎           | 70/280 [10:30<31:30, 9.00s/it, loss=0.6934, acc=0.625, margin=0.152]
```

---

## Troubleshooting

### "404 Not Found" for Wan-AI/Wan2.2-T2V-A14B
- **Cause:** Using original model format instead of Diffusers
- **Fix:** Already fixed - default is now `Wan-AI/Wan2.2-T2V-A14B-Diffusers`

### "No space left on device"
- **Cause:** Model downloading to root disk instead of /tmp
- **Fix:** Already fixed - cache redirected to /tmp

### "0 preference pairs loaded"
- **Cause:** Video files not accessible or Azure datastore not mounted
- **Fix:** Already fixed - datasets properly mounted as inputs

### Model downloading instead of using cache
- **Cause:** Cache from original format (Wan2.2-T2V-A14B) doesn't work for Diffusers format
- **Expected:** First run will download Diffusers model (~120GB) to /tmp
- **Note:** Subsequent runs on same compute node will reuse cached model

---

## Performance Estimates

Based on H100 GPU (93.1 GB VRAM):

- **Model loading**: ~2-3 minutes (first time: +30-60 minutes for download)
- **Training speed**: ~9-15 seconds per step (depends on video resolution)
- **Steps per epoch**: ~280 (1,120 pairs ÷ 4 effective batch size)
- **Epoch time**: ~40-70 minutes
- **Total training (10 epochs)**: ~7-12 hours

---

## Files Reference

- **Training script**: `azure/train_dpo_azure.py`
- **Submission script**: `azure/submit_job.py`
- **Submission wrapper**: `azure/submit_job.sh`
- **Preference data**: `video_rankings3_pairwise.json`
- **Videos**: `/Users/hivamoh/cs234Proj/azureml_outputs_final/` (local)
- **Azure output**: `wan22_dpo_outputs/` (Azure Blob Storage)

---

## Next Steps

1. **Submit training job**: `bash azure/submit_job.sh`
2. **Monitor in Azure ML Studio**: Check portal URL in output
3. **Check for successful model download**: Look for "✓ Pipeline loaded" in logs
4. **Verify dataset loading**: Should see >0 valid pairs
5. **Monitor training metrics**: Loss, accuracy, reward margin
6. **Download checkpoints**: After training completes from Azure Blob Storage
