# Azure ML Disk Space Fix

## Problem

The Wan2.2 model download was failing with "No space left on device" because HuggingFace was trying to download ~120GB of model files to `/root/.cache/` (root disk with only 73GB free) instead of `/tmp/` (which has 238GB free).

## Solution

The training script has been updated to automatically:
1. Set all HuggingFace cache directories to `/tmp/huggingface_cache`
2. Check for existing cached models in `/tmp` and reuse them
3. Show disk space before loading models

## How to Apply the Fix

### Option 1: Resubmit with Updated Script (Recommended)

Simply resubmit your training job. The updated `azure/train_dpo_azure.py` will automatically use `/tmp` for all downloads:

```bash
# From project root
bash azure/submit_job.sh
```

The script now includes:
- Automatic cache directory configuration
- Cached model detection
- Disk space reporting

### Option 2: Use Wrapper Script (Extra Safe)

If you want to be extra sure, use the wrapper script that sets environment variables before Python starts:

**Edit `azure/submit_job.py` line 177:**

```python
# Change from:
script="azure/train_dpo_azure.py",

# To:
script="azure/run_training.sh",
```

Then resubmit:

```bash
bash azure/submit_job.sh
```

### Option 3: Pre-download Model to /tmp

If your Azure job has persistent `/tmp` storage, you can pre-download the model once and reuse it:

1. SSH into your Azure compute node or run a setup job
2. Run:
   ```bash
   export HF_HOME=/tmp/huggingface_cache
   python -c "from diffusers import WanPipeline; WanPipeline.from_pretrained('Wan-AI/Wan2.2-T2V-A14B-Diffusers', cache_dir='/tmp/huggingface_cache')"
   ```
3. All subsequent training jobs will find and reuse this cached model

## Verification

After resubmitting, check the Azure ML logs for these messages:

```
======================================================================
  Configuring HuggingFace Cache Location
======================================================================
✓ HuggingFace cache: /tmp/huggingface_cache
✓ Torch cache: /tmp/torch_cache

Disk space before model loading:
Filesystem      Size  Used Avail Use% Mounted on
overlay         124G   59G   66G  48% /
/dev/sdb1       251G  XXG  XXXG  XX% /tmp

Checking for cached models...
✓ Found cached model at: /tmp/huggingface_cache/models--Wan-AI--Wan2.2-T2V-A14B/...
```

If you see these messages, the fix is working correctly!

## Files Changed

- `azure/train_dpo_azure.py` - Main training script (lines 264-331)
- `azure/init_environment.sh` - Environment initialization script (new)
- `azure/run_training.sh` - Training wrapper script (new)

## Why This Happened

1. HuggingFace defaults to `~/.cache/huggingface/` for model downloads
2. In Azure ML, this maps to `/root/.cache/` on the root filesystem
3. The root filesystem only has ~73GB free
4. The Wan2.2 model requires ~120GB for download + extraction
5. `/tmp/` is mounted on a separate disk with 238GB free

The fix redirects all HuggingFace operations to use `/tmp/` instead.
