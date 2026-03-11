# Azure ML Training Fixes - Summary

## Issues Fixed

### 1. Disk Space Error ✓ FIXED
**Problem:** Model download failing with "No space left on device" because HuggingFace was downloading to `/root/.cache/` (73GB available) instead of `/tmp/` (238GB available).

**Solution:**
- Set environment variables to redirect HuggingFace cache to `/tmp`:
  - `HF_HOME=/tmp/huggingface_cache`
  - `HUGGINGFACE_HUB_CACHE=/tmp/huggingface_cache`
  - `TRANSFORMERS_CACHE=/tmp/huggingface_cache`
  - `HF_HUB_CACHE=/tmp/huggingface_cache`
  - `TORCH_HOME=/tmp/torch_cache`
- Added automatic detection of cached models to avoid re-downloading
- Added disk space reporting before model loading

**Files Changed:**
- `azure/train_dpo_azure.py` (lines 264-331)

---

### 2. Dataset Loading Error (0 preference pairs) ✓ FIXED
**Problem:** Dataset was loading 0 pairs because Azure ML datastore URIs (`azureml://...`) were not being mounted/downloaded properly.

**Solution:**
- Updated `submit_job.py` to properly mount datasets as inputs using Azure ML Dataset API
- Videos are mounted (as_mount) for fast access
- Preference JSON is downloaded (as_download) to ensure file availability
- Updated training script to handle mounted paths correctly

**Files Changed:**
- `azure/submit_job.py` (data handling section, lines 140-193)
- `azure/train_dpo_azure.py` (dataset class, lines 66-76)

---

### 3. Wrong Model Being Used ✓ FIXED
**Problem:** Script was defaulting to `Wan-AI/Wan2.2-T2V-A14B-Diffusers` instead of the standard `Wan-AI/Wan2.2-T2V-A14B`.

**Solution:**
- Changed default model to `Wan-AI/Wan2.2-T2V-A14B`
- Added smart cache detection that checks for both model variants
- Will use whichever model is already cached to avoid re-downloading

**Files Changed:**
- `azure/train_dpo_azure.py` (line 580, model default argument)
- `azure/train_dpo_azure.py` (lines 289-321, cache detection logic)

---

### 4. Train/Test Split Support ✓ ADDED
**Problem:** No way to filter dataset to use only training indices (excluding test set).

**Solution:**
- Added `--train-only` flag to automatically exclude test indices
- Added `--test-indices` argument to specify which videos are test set (default: 0,1,8,15,21,25,27,28,45,48,50,52)
- Dataset now supports `filter_indices` parameter to only load specific videos
- Automatically computes train indices as (0-59) - test_indices

**Usage:**
```bash
python azure/train_dpo_azure.py \
  --data preference.json \
  --videos-dir /path/to/videos \
  --train-only  # Only use training data
```

**Files Changed:**
- `azure/train_dpo_azure.py` (lines 606-623, argument parser)
- `azure/train_dpo_azure.py` (lines 113-125, dataset filtering)
- `azure/train_dpo_azure.py` (line 456, pass filter_indices to dataset)

---

### 5. Preference File Upload ✓ FIXED
**Problem:** `upload_data_to_azure` only uploaded .mp4 files, not the preference JSON file.

**Solution:**
- Updated upload function to also copy and upload the preference JSON file
- Returns both videos path and preference file path

**Files Changed:**
- `azure/submit_job.py` (lines 63-102, upload function)

---

## How to Use the Fixed Code

### Option 1: Submit New Training Job (Recommended)
```bash
# From project root
bash azure/submit_job.sh
```

The script will:
1. Upload videos and preference file to Azure Blob Storage
2. Create mounted datasets for fast access
3. Use /tmp for model caching (avoiding disk space issues)
4. Train with all available data

### Option 2: Train Only on Training Set
Edit `azure/submit_job.sh` and add to the arguments:
```bash
python azure/submit_job.py \
  ... existing args ... \
  --train-only
```

Or when calling directly:
```bash
python azure/train_dpo_azure.py \
  --data /path/to/preference.json \
  --videos-dir /path/to/videos \
  --train-only \
  --test-indices "0,1,8,15,21,25,27,28,45,48,50,52"
```

---

## Verification Checklist

After submitting the job, check Azure ML logs for these success indicators:

### ✓ Disk Space Fix Working
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

### ✓ Model Loading Working
```
Checking for cached models...
✓ Found cached model at: /tmp/huggingface_cache/models--Wan-AI--Wan2.2-T2V-A14B/...
✓ Pipeline loaded
```

### ✓ Dataset Loading Working
```
======================================================================
  Loading Preference Dataset
======================================================================
Preference file: /tmp/azureml_runs/.../pref_data/video_rankings3_pairwise.json
Videos directory: /mnt/azureml/.../videos

Found X prompt groups in preference file
Found Y video files in /mnt/azureml/.../videos

Dataset Statistics:
  Total pairs in JSON: N
  Valid pairs (both videos exist): M
```

### ✓ Training Started
```
======================================================================
  Starting DPO Training
======================================================================
Epochs: 10
Batch size: 1 (effective: 4)
Beta: 0.1

Epoch 1/10: [progress bar]
```

---

## Common Issues and Solutions

### Issue: "Preference file not found"
- **Cause:** Preference file not uploaded to Azure or path incorrect
- **Fix:** Check that `video_rankings3_pairwise.json` exists in project root before submitting

### Issue: "0 preference pairs loaded"
- **Cause:** Video files don't match names in JSON, or aren't uploaded
- **Fix:** Check `LOCAL_DATA_DIR` in `azure/submit_job.sh` points to correct directory with videos

### Issue: "No space left on device" (still happening)
- **Cause:** Old version of training script running
- **Fix:** Re-submit job to deploy updated script

### Issue: Model downloading instead of using cache
- **Cause:** Cache directory doesn't persist between jobs, or using different compute node
- **Fix:** Normal behavior for first run on new compute node. Subsequent runs will be faster.

---

## Files Modified Summary

1. **azure/train_dpo_azure.py** - Main training script
   - HuggingFace cache redirection
   - Cached model detection
   - Dataset filtering by indices
   - Azure mounted path handling
   - Model default changed to Wan2.2-T2V-A14B

2. **azure/submit_job.py** - Job submission script
   - Preference file upload
   - Dataset mounting (proper Azure ML data handling)
   - Import cleanup

3. **azure/init_environment.sh** - NEW - Environment initialization
4. **azure/run_training.sh** - NEW - Training wrapper script
5. **azure/DISK_SPACE_FIX.md** - NEW - Disk space fix documentation
6. **azure/FIXES_SUMMARY.md** - NEW - This file

---

## Testing Locally (Before Azure Submission)

To test the training script locally with your data:

```bash
python azure/train_dpo_azure.py \
  --data video_rankings3_pairwise.json \
  --videos-dir /Users/hivamoh/cs234Proj/azureml_outputs_final \
  --output-dir ./dpo_output \
  --epochs 1 \
  --batch-size 1 \
  --train-only
```

This will:
- Use the standard Wan2.2-T2V-A14B model
- Load only training videos (excluding test set)
- Run for 1 epoch
- Save outputs to ./dpo_output
